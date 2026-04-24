# OPTS_TTPO 设计文档

## 1. 算法概述

OPTS_TTPO（On-policy Parallel Tree Search + Tree Trajectory Policy Optimization）将树搜索与 PPO 策略梯度相结合。每个环境在 rollout 过程中维护一棵或多棵搜索树，episode 终止时通过 OTRC 选择已有树的分支点进行回溯扩展，或开启新树。所有树上的数据统一进行 TreeGAE 优势估计和 branch_weight 校正后的 PPO 更新。

### 与 PPO 的区别

| 特性 | PPO | OPTS_TTPO |
|------|-----|-----------|
| 轨迹结构 | 线性 | 树形，多棵树共存 |
| 终止处理 | reset | OTRC 选择分支点或开新树 |
| 优势估计 | GAE | TreeGAE，分支节点取子节点优势均值回传 |
| 策略梯度 | 标准 | branch_weight 加权校正 |
| 跨迭代 | 每次全部 reset | 未终止 episode 自然延续到下一迭代 |

### 环境要求

需要环境支持状态快照（clone_state / restore_state）。MuJoCo 通过 MuJoCoStateSnapshotWrapper 保存 qpos、qvel、派生量和所有 wrapper 状态；Atari 通过 AtariStateSnapshotWrapper 保存 ALE state 和 wrapper 状态。wrapper 状态包括 TimeLimit 的 elapsed_steps、RecordEpisodeStatistics 的累计 return/length、NormalizeObservation/NormalizeReward 的 running statistics 等。


## 2. 超参数

在 PPO 基础参数之外，新增以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_search_per_tree | 4 | 每棵树每迭代最大搜索次数 |
| c | 1.0 | OTRC 中 exploration 项系数 |


## 3. 数据结构

### 树结构编码

树结构通过 parent_indices 张量（形状 num_steps x num_envs）隐式编码。每个节点对应 rollout 中的一个 step，parent_indices 记录其父节点的 step 索引。负值表示父节点是某个根状态。

tree_indices 张量记录每个节点所属的树 ID，该值等于该树根节点在 parent_indices 中的负数值。节点的 tree_id 继承规则：若自身的 parent 为负数则 tree_id 等于该负数，否则继承父节点的 tree_id。

state_branches 张量记录每个节点被分支的次数，初始为 1。当 OTRC 选择从某节点分支时，该节点的 state_branches 加 1。

### 根节点管理

root_states 是每个环境的列表，新根通过 insert(0, ...) 插入列表头部。parent_indices 中的负值通过 Python 负数索引映射到 root_states 列表：parent=-1 对应 root_states[-1]（最早创建的根），parent=-2 对应 root_states[-2]（第二棵树的根），依此类推。开新树时 current_parent 设为 -len(root_states)。

### 每迭代辅助数据

以下数据在每迭代开始时全部重新创建：

- root_branch_counts：每个环境的字典，记录各根状态被分支的次数（即从同一根出发的一级节点数），用于计算根节点的 branch_weight。
- search_count：每个环境的字典，记录每棵树在本迭代已被搜索的次数，达到 max_search_per_tree 时不再对该树搜索。
- tree_max_returns：每个环境的字典，记录每棵树在本迭代的最大 episodic return，用于 OTRC 中的树跳过判断。
- episodic_return_info：全局列表，记录本迭代所有终止 episode 的 (return, tree_id, step, env_idx) 四元组，用于计算 aggregated_returns。

### 环境状态快照

env_states 是二维列表（num_steps x num_envs），保存每步执行动作后的环境状态快照。用于 OTRC 选择分支点后恢复环境到该点的父状态。

### 树结构示例

假设某环境有两棵树。树 A 最先创建，根状态存于 root_states[-1]，tree_id=-1。树 A 有一条从根出发的主路径 step 0→1→2（step 2 终止），之后 OTRC 选择从根分支，产生 step 3→4。树 B 后创建，根状态存于 root_states[-2]，tree_id=-2，有路径 step 5→6。

此时 parent_indices 为 [-1, 0, 1, -1, 3, -2, 5]，tree_indices 为 [-1, -1, -1, -1, -1, -2, -2]。root_branch_counts 中 -1 的值为 2（根 A 出发了两条分支），-2 的值为 1。


## 4. 训练流程

### 4.1 迭代初始化

迭代开始时，只对上一迭代最后一步处于终止状态的环境执行 reset，未终止的环境保留 next_obs 和环境内部状态自然延续。所有环境都 clone 当前状态作为本迭代的根状态。然后重置树结构张量（current_parent 为 -1、parent_indices 为 -1、tree_indices 为 0、state_branches 为 1、advantages 为 0、next_done 为 0），并重新创建所有每迭代辅助数据（root_branch_counts、search_count、tree_max_returns、episodic_return_info）。

延续的环境在第一步的 parent_indices 为 -1，因此 tree_id 也为 -1，与正常开新树的行为一致。延续 episode 的 RecordEpisodeStatistics 累计值在 wrapper 中自然保持，终止时报告完整的 episodic return。

### 4.2 采样循环

每步操作按以下顺序执行：

**保存观测和采样动作。** 将 next_obs 存入 obs[step]，通过 agent 采样得到 action、logprob 和 value。

**记录树结构。** 将 current_parent 写入 parent_indices[step]，根据父节点值推导 tree_indices[step]。若父节点为负数（根节点），更新 root_branch_counts。然后将 current_parent 更新为当前 step。

**执行动作。** 所有环境执行 action，收集 next_obs、reward、done，保存状态快照到 env_states[step]。若 episode 终止，记录 episodic return 并更新 tree_max_returns。

**处理终止环境。** 分三阶段处理所有终止的环境：

1. **TreeGAE 阶段**：对每个终止环境分别调用 TreeGAE，从终止节点回溯更新 advantages（next_value=0）。
2. **OTRC 选择阶段**：将所有终止环境的索引批量传入 select_next_states，一次性为每个环境选择下一步操作。
3. **状态恢复阶段**：逐个根据选择结果恢复环境状态。

若 OTRC 返回负值（开新树），reset 环境，将新根插入 root_states 头部，设置 current_parent 为对应的负索引。

若 OTRC 返回非负 step 索引（分支搜索），恢复到该 step 的父状态。若父状态是根则从 root_states 恢复，否则从 env_states 恢复并将该父节点的 state_branches 加 1。将 next_obs 设为被选中 step 的观测（从同一状态重新采样新动作），current_parent 设为该父节点。

### 4.3 采样结束后

对所有 current_parent >= 0 的环境（存在未终止轨迹），用 next_obs 的 value 作为 bootstrap value 调用 TreeGAE。

计算 returns = advantages + values。

计算 branch_weight。

按 (env_idx, tree_id) 分组，对本迭代所有终止 episode 的 return 进行 branch_weight 加权平均，得到 aggregated_returns，进而计算 mean_return。将 mean_return 直接作为下一迭代 OTRC 的 return_threshold（prev_mean_return = mean_return）。

执行 PPO 更新，policy loss 和 value loss 均用 branch_weight 加权。


## 5. 关键算法

### 5.1 TreeGAE

从指定节点沿 parent_indices 回溯到根，逐节点计算优势。

对于每个节点 t，先找到其所有子节点（在 terminal_step 范围内 parent_indices 等于 t 的节点）。若无子节点且该节点已终止，V_next 为 0；若无子节点且未终止，V_next 为 bootstrap value；若有子节点，V_next 取第一个子节点的 value（所有子节点观测相同，value 相同）。

TD 误差 delta = reward + gamma * V_next - V_current。叶节点的 advantage 直接等于 delta。分支节点的 advantage 等于 delta + gamma * lambda * 所有子节点 advantage 的均值。这一均值回传自然实现了不同分支轨迹的优势聚合。

每次 episode 终止时触发一次 TreeGAE（next_value=0），迭代结束时对未终止轨迹再触发一次（带 bootstrap value）。advantage 是 in-place 更新，分支节点的值随着更多子轨迹完成而持续修正。

### 5.2 OTRC（Tree Upper Confidence for Trees）

episode 终止时，为该环境跨所有树全局选择最佳分支点。

首先处理两种直接开新树的情况：若当前是最后一步（skip_search=True），或 return_threshold 为 None（首次迭代，尚无 mean_return），则直接开新树。

对每棵树依次处理。首先跳过已达 max_search 次数的树和最大 return 超过 return_threshold（即上一迭代的 mean_return）的树。

对于剩余的树，找到其最优路径：从 advantage 最大的根节点出发，每步贪心选择 advantage 最大的子节点，直到叶节点。

沿最优路径计算 OTRC 值。

#### 5.2.1 exploitation（期望改善量）的数学推导

**目标**：估计从节点 k 分支的期望改善 V^π(s_k) - G_k。

**第一步：精确 telescoping 恒等式（纯代数，零近似）**

定义 TD error δ_t = r_t + γ V(s_{t+1}) - V(s_t)。对 V(s_k) - G_k 在每步加减 γ^{t-k} V(s_t)：

```
V(s_k) - G_k = [V(s_k) - r_k - γV(s_{k+1})] + γ[V(s_{k+1}) - r_{k+1} - γV(s_{k+2})] + ...
             = -δ_k - γδ_{k+1} - γ²δ_{k+2} - ...
             = -sum_{t=k}^{T} γ^{t-k} δ_t + γ^{T-k+1} V(s_{T+1})
```

若 s_{T+1} 为终止状态，V(s_{T+1})=0，则 V(s_k) - G_k = -sum_{t=k}^{T} γ^{t-k} δ_t。此式对任意 V 成立，无需近似。

**第二步：δ 与 A^π 的关系**

真实优势 A^π(s,a) = Q^π(s,a) - V^π(s)。TD error δ_t = r_t + γV(s_{t+1}) - V(s_t)。对于具体转移，δ_t 是 A^π 的无偏但高方差的单样本估计：E[δ_t | s_t, a_t] = A^π(s_t, a_t)（当 V = V^π 时）。

**第三步：用 GAE 替代 δ 提高稳定性**

因为 V̂ ≠ V^π，单步 δ̂_t 依赖两个不准确的 V̂ 值做差，偏差大。GAE Â_t = sum_l (γλ)^l δ_{t+l} 通过多步加权平均平滑了 V̂ 的误差，是 A^π 更稳定的估计器。

将目标量分解为优势的加权和后，用 GAE 逐项替代：

```
V^π(s_k) - G_k = -sum_{t=k}^{T} γ^{t-k} A^π(s_t, a_t)
               ≈ -sum_{t=k}^{T} γ^{t-k} Â_t^{GAE}
```

这不是声称代数恒等式对 GAE 成立，而是：目标量分解为各步优势的加权和，用最好的优势估计器（GAE）逐项逼近，得到更稳定的估计。

**下游 advantage 的含义**：A_{k+1}, A_{k+2}, ... 不是在预测"分支后会经过什么状态"，而是在度量"原轨迹从该点开始比策略期望差了多少"。新分支的期望回报为 V^π(s_k)（按定义），原轨迹实际回报为 G_k，差值 V^π(s_k) - G_k 就是期望改善量，它恰好等于下游优势的折扣加权和的负值。

**实际计算**：从后向前迭代 discounted_sum = -Â_k + γ * discounted_sum，再除以后续长度 (n-k) 得到单位预算期望改善。exploitation 为正值表示有改善空间。

#### 5.2.2 exploration（搜索惩罚）

等于 (sibling_count - 1) * max_abs_exploitation，其中 sibling_count 是共享同一父节点的全部子节点数（包含节点自身），因此 (sibling_count - 1) 表示该父节点已被分支探索的额外次数。max_abs_exploitation 是整条路径上 exploitation 绝对值的最大值（若为 0 则设为 1.0），用于将 exploration 项标准化到与 exploitation 同一量级。

#### 5.2.3 OTRC 与选择

**OTRC = exploitation - c * exploration**。取路径上 OTRC 最大的节点（期望改善最大且未被过度搜索）作为候选分支点。

遍历所有树后，选择全局 OTRC 最大的分支点。只要存在任何可搜索的树（未被跳过且搜索次数未满），就一定选择分支，不检查 OTRC 值正负。仅当所有树都被跳过（搜索次数已满或 max return 超过阈值）时才开新树。

### 5.3 Branch Weight Factor

branch_weight 校正树形结构下的策略梯度，使其保持无偏。

根节点的 weight 等于 root_branch_counts 中该根的分支数，即从同一根状态出发的一级节点总数。非根节点的 weight 等于父节点的 weight 乘以父节点的 state_branches 值，即从根到该节点路径上所有祖先分支数的累乘。

直觉：一个节点被经过的次数越多（因为其祖先被多次分支），它在数据中出现的频率就越高，需要除以 weight 来消除重复采样的偏差。

### 5.4 Aggregated Returns

每迭代结束时，按 (env_idx, tree_id) 对本迭代终止的 episode 分组。同组内各 episode 的 return 用 branch_weight 倒数加权平均，得到每组的 aggregated_return。对所有组求均值得到 mean_return。

mean_return 直接赋值给 prev_mean_return，作为下一迭代 OTRC 中的树跳过阈值（return_threshold）。首次迭代时 prev_mean_return 为 None，此时所有终止 episode 直接开新树，不进行树搜索。

### 5.5 策略梯度（TTPO）

PPO 的 clipped surrogate loss 和 value loss 均除以 branch_weight 后加权求和，而非简单均值：

```
pg_loss = sum(clip_loss_i / W_i) / sum(1 / W_i)
v_loss  = sum(value_loss_i / W_i) / sum(1 / W_i)
```

entropy_loss 不进行 branch_weight 加权，直接取简单均值：

```
entropy_loss = mean(entropy_i)
loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
```

这确保了被多次经过的节点不会在策略梯度和价值回归中被过度表示，而熵正则化保持对所有节点的均匀鼓励。


## 6. 实现要点

### 6.1 环境管理

不能使用 SyncVectorEnv，因为树搜索需要单独操控每个环境的 clone/restore 状态。使用独立环境列表 `envs = [make_env(...)() for i in range(num_envs)]`，所有环境循环在 Python 层逐个执行。

### 6.2 StateSnapshotWrapper

环境创建时最外层包裹 StateSnapshotWrapper（Atari 或 MuJoCo 版本），提供 `clone_state()` 和 `restore_state(state)` 接口。wrapper 初始化时沿 `env` 链自动查找并缓存内部各层 wrapper 的引用，确保 clone/restore 时完整保存和恢复所有 wrapper 状态。

#### episode 统计值的快照时序问题

RecordEpisodeStatistics 在 episode 终止时先将累计值写入 `info['episode']`，然后立即将内部计数器 `episode_returns` / `episode_lengths` 归零。如果在终止后直接读取计数器，得到的是归零后的值而非真实累计值。

StateSnapshotWrapper 通过在自身的 `step()` 方法中维护 `_episode_return_snapshot` / `_episode_length_snapshot` 解决此问题：
- episode 未终止时，从 `RecordEpisodeStatistics.episode_returns[0]` 读取当前累计值。
- episode 终止时，从 `info['episode']` 中提取终止前的真实累计值。
- `clone_state()` 保存的是 snapshot 值而非直接读计数器，确保恢复后累计值正确。
- `reset()` 时将 snapshot 归零。

#### Atari（AtariStateSnapshotWrapper）

保存和恢复以下状态：

| 组件 | 保存内容 |
|------|----------|
| ALE | `ale.cloneState()` / `ale.restoreState()`，包含 RAM、寄存器等全部模拟器状态 |
| FrameStack | `frames` deque 中所有帧的深拷贝，恢复时 clear 后逐帧 append |
| MaxAndSkipEnv | `_obs_buffer` 数组的深拷贝 |
| EpisodicLifeEnv | `lives` 和 `was_real_done` 标志 |
| TimeLimit | `_elapsed_steps` |
| RecordEpisodeStatistics | 通过 snapshot 机制保存的累计 return/length |

Atari 环境 wrapper 链（从内到外）：`gym.make` → `RecordEpisodeStatistics` → `NoopResetEnv` → `MaxAndSkipEnv(skip=4)` → `EpisodicLifeEnv` → `FireResetEnv`（若适用）→ `ClipRewardEnv` → `ResizeObservation(84,84)` → `GrayScaleObservation` → `FrameStack(4)` → `AtariStateSnapshotWrapper`。

#### MuJoCo（MuJoCoStateSnapshotWrapper）

保存和恢复以下状态：

| 组件 | 保存内容 |
|------|----------|
| mjData 核心 | `qpos`、`qvel`、`time` |
| mjData 可选输入 | `act`、`qacc`、`ctrl`、`qfrc_applied`、`xfrc_applied` 等（存在且非空时保存） |
| mjData 派生量 | `xpos`、`xquat`、`xmat`、`xipos`、`site_xpos`、`subtree_com`、`cinert`、`cvel` 等 |
| Mocap 体 | `mocap_pos`、`mocap_quat`（Reacher 等环境使用） |
| 环境 Python 属性 | `_last_x_position`（HalfCheetah）、`_last_position`、`_init_obs` 等影响奖励/观测的属性 |
| 目标 | `env.goal`（Reacher 等目标条件环境） |
| RNG 状态 | `np_random.bit_generator.state`（含随机目标的环境需要） |
| TimeLimit | `_elapsed_steps` |
| RecordEpisodeStatistics | 通过 snapshot 机制保存的累计 return/length |
| NormalizeObservation | `obs_rms` 的 mean、var、count |
| NormalizeReward | `return_rms` 的 mean、var、count 以及 `returns` 累加器 |

**派生量的恢复顺序**：Gymnasium MuJoCo 环境在 `_get_obs()` 中使用的是 `mj_step` 积分前计算的"陈旧"派生量（如 `site_xpos`、`cfrc_ext`）。恢复时的三步操作确保观测一致性：

1. 先恢复所有 mjData 字段（qpos、qvel、ctrl 等）
2. 调用 `mujoco.mj_forward(model, data)` 重算派生量
3. 再用保存的派生量覆盖 `mj_forward` 的结果，还原到保存时的"陈旧"值

MuJoCo 环境 wrapper 链（从内到外）：`gym.make` → `FlattenObservation` → `RecordEpisodeStatistics` → `ClipAction` → `NormalizeObservation` → `TransformObservation(clip ±10)` → `NormalizeReward(gamma)` → `TransformReward(clip ±10)` → `MuJoCoStateSnapshotWrapper`。

### 6.3 两个实现变体的差异

核心算法（TreeGAE、OTRC、branch_weight、aggregated_returns、加权 PPO 更新）在 Atari 和 MuJoCo 两个实现中完全一致，差异仅在于环境接口适配层：

| 差异项 | Atari (`opts_ttpo_atari.py`) | MuJoCo (`opts_ttpo_continuous_action.py`) |
|--------|------------------------------|-------------------------------------------|
| 动作空间 | Discrete，Categorical 分布采样 | Box，Normal 分布采样（actor_mean + actor_logstd） |
| 网络结构 | CNN（3 层 Conv2d → Linear(512)）| MLP（2 层 Linear(64) + Tanh） |
| 观测预处理 | 网络内 `x / 255.0` 归一化 | 环境层 NormalizeObservation + clip ±10 |
| 奖励预处理 | ClipRewardEnv（clip 到 {-1,0,1}）| NormalizeReward + clip ±10 |
| 快照实现 | ALE cloneState/restoreState | mjData 字段深拷贝 + mj_forward + 派生量覆盖 |
| 需保存的 wrapper 状态 | FrameStack、MaxAndSkipEnv、EpisodicLifeEnv | NormalizeObservation、NormalizeReward（含 running statistics） |

### 6.4 结果保存

每迭代结束后将 aggregated_returns 的统计量（mean_return、max_return、min_return）追加写入 JSON 文件，路径为 `./results/{num_envs}_{num_steps}/{algorithm_name}/{env_id}_{seed}.json`。`algorithm_name` 格式为 `{exp_name}_{YYYYMMDD}`。同时通过 TensorBoard（SummaryWriter）记录 episodic_return、episodic_length、loss 等指标，可选启用 wandb 同步。
