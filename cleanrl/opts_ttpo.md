# OPTS_TTPO 设计文档

## 1. 算法概述

OPTS_TTPO（On-policy Parallel Tree Search + Tree Trajectory Policy Optimization）将树搜索与 PPO 策略梯度相结合。每个环境在 rollout 过程中维护一棵或多棵搜索树，episode 终止时通过 TUCT 选择已有树的分支点进行回溯扩展，或开启新树。所有树上的数据统一进行 TreeGAE 优势估计和 branch_weight 校正后的 PPO 更新。

### 与 PPO 的区别

| 特性 | PPO | OPTS_TTPO |
|------|-----|-----------|
| 轨迹结构 | 线性 | 树形，多棵树共存 |
| 终止处理 | reset | TUCT 选择分支点或开新树 |
| 优势估计 | GAE | TreeGAE，分支节点取子节点优势均值回传 |
| 策略梯度 | 标准 | branch_weight_factor 加权校正 |
| 跨迭代 | 每次全部 reset | 未终止 episode 自然延续到下一迭代 |

### 环境要求

需要环境支持状态快照（clone_state / restore_state）。MuJoCo 通过 MuJoCoStateSnapshotWrapper 保存 qpos、qvel、派生量和所有 wrapper 状态；Atari 通过 AtariStateSnapshotWrapper 保存 ALE state 和 wrapper 状态。wrapper 状态包括 TimeLimit 的 elapsed_steps、RecordEpisodeStatistics 的累计 return/length、NormalizeObservation/NormalizeReward 的 running statistics 等。


## 2. 超参数

在 PPO 基础参数之外，新增以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| alpha | 1.0 | mean_return 和 std_return 的 EMA 平滑系数 |
| max_search_per_tree | 4 | 每棵树每迭代最大搜索次数 |
| c | 1.0 | TUCT 中 exploration 项系数 |
| beta | 0.0 | 树跳过阈值系数，当树的最大 return 超过 mean_return + beta * std_return 时跳过 |
| tail_length | 2 | TUCT 选中最优路径末端 tail_length 个节点时跳过该树 |


## 3. 数据结构

### 树结构编码

树结构通过 parent_indices 张量（形状 num_steps x num_envs）隐式编码。每个节点对应 rollout 中的一个 step，parent_indices 记录其父节点的 step 索引。负值表示父节点是某个根状态。

tree_indices 张量记录每个节点所属的树 ID，该值等于该树根节点在 parent_indices 中的负数值。节点的 tree_id 继承规则：若自身的 parent 为负数则 tree_id 等于该负数，否则继承父节点的 tree_id。

state_branches 张量记录每个节点被分支的次数，初始为 1。当 TUCT 选择从某节点分支时，该节点的 state_branches 加 1。

### 根节点管理

root_states 是每个环境的列表，新根通过 insert(0, ...) 插入列表头部。parent_indices 中的负值通过 Python 负数索引映射到 root_states 列表：parent=-1 对应 root_states[-1]（最早创建的根），parent=-2 对应 root_states[-2]（第二棵树的根），依此类推。开新树时 current_parent 设为 -len(root_states)。

root_branch_counts 字典记录每个根状态被分支的次数（即从同一根出发的一级节点数），用于计算根节点的 branch_weight。

### 每迭代辅助数据

search_count 记录每棵树在本迭代已被搜索的次数，达到 max_search_per_tree 时不再对该树搜索。tree_max_returns 记录每棵树在本迭代的最大 episodic return，用于 TUCT 中的树跳过判断。episodic_return_info 记录本迭代所有终止 episode 的 return、tree_id、step 和 env_idx，用于计算 aggregated_returns。

### 环境状态快照

env_states 是二维列表（num_steps x num_envs），保存每步执行动作后的环境状态快照。用于 TUCT 选择分支点后恢复环境到该点的父状态。

### 树结构示例

假设某环境有两棵树。树 A 最先创建，根状态存于 root_states[-1]，tree_id=-1。树 A 有一条从根出发的主路径 step 0→1→2（step 2 终止），之后 TUCT 选择从根分支，产生 step 3→4。树 B 后创建，根状态存于 root_states[-2]，tree_id=-2，有路径 step 5→6。

此时 parent_indices 为 [-1, 0, 1, -1, 3, -2, 5]，tree_indices 为 [-1, -1, -1, -1, -1, -2, -2]。root_branch_counts 中 -1 的值为 2（根 A 出发了两条分支），-2 的值为 1。


## 4. 训练流程

### 4.1 迭代初始化

迭代开始时，只对上一迭代最后一步处于终止状态的环境执行 reset，未终止的环境保留 next_obs 和环境内部状态自然延续。所有环境都 clone 当前状态作为本迭代的根状态。然后重置 current_parent 为 -1、清零 advantages 和 dones、重置 parent_indices 为 -1、重置 state_branches 为 1、清零 tree_indices。

延续的环境在第一步的 parent_indices 为 -1，因此 tree_id 也为 -1，与正常开新树的行为一致。延续 episode 的 RecordEpisodeStatistics 累计值在 wrapper 中自然保持，终止时报告完整的 episodic return。

### 4.2 采样循环

每步操作按以下顺序执行：

**保存观测和采样动作。** 将 next_obs 存入 obs[step]，通过 agent 采样得到 action、logprob 和 value。

**记录树结构。** 将 current_parent 写入 parent_indices[step]，根据父节点值推导 tree_indices[step]。若父节点为负数（根节点），更新 root_branch_counts。然后将 current_parent 更新为当前 step。

**执行动作。** 所有环境执行 action，收集 next_obs、reward、done，保存状态快照到 env_states[step]。若 episode 终止，记录 episodic return 并更新 tree_max_returns。

**处理终止环境。** 对所有终止的环境依次执行：先调用 TreeGAE 从终止节点回溯更新 advantages；再调用 TUCT 选择下一步操作；最后根据选择结果恢复状态。

若 TUCT 返回负值（开新树），reset 环境，将新根插入 root_states 头部，设置 current_parent 为对应的负索引。

若 TUCT 返回非负 step 索引（分支搜索），恢复到该 step 的父状态。若父状态是根则从 root_states 恢复，否则从 env_states 恢复并将该父节点的 state_branches 加 1。将 next_obs 设为被选中 step 的观测（从同一状态重新采样新动作），current_parent 设为该父节点。

### 4.3 采样结束后

对所有 current_parent >= 0 的环境（存在未终止轨迹），用 next_obs 的 value 作为 bootstrap value 调用 TreeGAE。

计算 returns = advantages + values。

计算 branch_weight_factors。

按 (env_idx, tree_id) 分组，对本迭代所有终止 episode 的 return 进行 branch_weight 加权平均，得到 aggregated_returns，进而计算 mean_return 和 std_return。通过 EMA 更新跨迭代的 prev_mean_return 和 prev_std_return，供下一迭代 TUCT 使用。

执行 PPO 更新，policy loss 和 value loss 均用 branch_weight 加权。


## 5. 关键算法

### 5.1 TreeGAE

从指定节点沿 parent_indices 回溯到根，逐节点计算优势。

对于每个节点 t，先找到其所有子节点（在 terminal_step 范围内 parent_indices 等于 t 的节点）。若无子节点且该节点已终止，V_next 为 0；若无子节点且未终止，V_next 为 bootstrap value；若有子节点，V_next 取第一个子节点的 value（所有子节点观测相同，value 相同）。

TD 误差 delta = reward + gamma * V_next - V_current。叶节点的 advantage 直接等于 delta。分支节点的 advantage 等于 delta + gamma * lambda * 所有子节点 advantage 的均值。这一均值回传自然实现了不同分支轨迹的优势聚合。

每次 episode 终止时触发一次 TreeGAE（next_value=0），迭代结束时对未终止轨迹再触发一次（带 bootstrap value）。advantage 是 in-place 更新，分支节点的值随着更多子轨迹完成而持续修正。

### 5.2 TUCT（Tree Upper Confidence for Trees）

episode 终止时，为该环境跨所有树全局选择最佳分支点。

对每棵树依次处理。首先跳过已达 max_search 次数的树和最大 return 超过 mean_return + beta * std_return 的树。

对于剩余的树，找到其最优路径：从 advantage 最大的根节点出发，每步贪心选择 advantage 最大的子节点，直到叶节点。

沿最优路径计算 TUCT 值。exploitation 是 backward cumulative mean，即节点 k 的 exploitation 等于从 k 到路径末端所有 advantage 的均值，反映从该点开始的子路径整体质量。exploration 等于 (sibling_count - 1) * max_abs_exploitation，其中 sibling_count 是与该节点共享父节点的兄弟数量，max_abs_exploitation 是整条路径上 exploitation 绝对值的最大值。TUCT = exploitation + c * exploration。

取路径上 TUCT 最小的节点作为候选分支点。若该节点位于路径末端 tail_length 个节点内，跳过这棵树。这一过滤的原因是：当 TUCT 最小值出现在末端时，说明整条最优路径的前半段表现尚可或已被充分探索，问题集中在末端，而末端的差表现通常是状态质量问题而非动作选择问题，从末端附近分支无法改善根本方向，且会浪费 search 预算。

遍历所有树后，选择全局 TUCT 最小的分支点。若无可选树，开新树。

### 5.3 Branch Weight Factor

branch_weight 校正树形结构下的策略梯度，使其保持无偏。

根节点的 weight 等于 root_branch_counts 中该根的分支数，即从同一根状态出发的一级节点总数。非根节点的 weight 等于父节点的 weight 乘以父节点的 state_branches 值，即从根到该节点路径上所有祖先分支数的累乘。

直觉：一个节点被经过的次数越多（因为其祖先被多次分支），它在数据中出现的频率就越高，需要除以 weight 来消除重复采样的偏差。

### 5.4 Aggregated Returns 与 EMA

每迭代结束时，按 (env_idx, tree_id) 对本迭代终止的 episode 分组。同组内各 episode 的 return 用 branch_weight 倒数加权平均，得到每组的 aggregated_return。对所有组求均值和标准差得到 mean_return 和 std_return。

通过 EMA 跨迭代平滑：prev_mean_return = alpha * mean_return + (1 - alpha) * prev_mean_return，std_return 同理。平滑后的值用于下一迭代 TUCT 中的树跳过阈值判断。

### 5.5 策略梯度（TTPO）

PPO 的 clipped surrogate loss 和 value loss 均除以 branch_weight 后加权求和，而非简单均值：

```
pg_loss = sum(clip_loss_i / W_i) / sum(1 / W_i)
v_loss  = sum(value_loss_i / W_i) / sum(1 / W_i)
```

这确保了被多次经过的节点不会在梯度中被过度表示。


## 6. 实现要点

不能使用 SyncVectorEnv，因为需要单独操控每个环境的 clone/restore 状态，使用独立环境列表。

环境创建时最外层包裹 StateSnapshotWrapper（MuJoCo 或 Atari 版本），在其内部自动查找并定位 TimeLimit、RecordEpisodeStatistics、NormalizeObservation、NormalizeReward 等 wrapper，确保 clone/restore 时完整保存和恢复所有 wrapper 状态。
