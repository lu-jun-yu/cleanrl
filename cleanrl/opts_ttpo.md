# OPTS_TTPO for CleanRL 设计文档

## 1. 算法概述

OPTS_TTPO（On-Policy Parallel Tree Search + Tree Trajectory Policy Optimization）是一种将树搜索与策略梯度优化相结合的强化学习算法，基于 PPO 修改而来。

### 1.1 核心思想

**OPTS（同策略并行树搜索）**
- 每个环境维护一棵独立的搜索树
- 环境终止时，通过 TUCT 选择树中最优状态-动作对进行回溯扩展
- 支持恢复到早期状态重新探索

**TTPO（树轨迹策略优化）**
- 使用 TreeGAE 计算树结构上的优势估计
- 使用 branch_weight_factor 校正策略梯度，保证无偏估计

### 1.2 与标准 PPO 的主要区别

| 特性 | PPO | OPTS_TTPO |
|------|-----|-----------|
| 轨迹结构 | 线性 | 树形 |
| 终止处理 | 自动 reset | TUCT 选择 + 状态恢复 |
| 优势估计 | GAE | TreeGAE |
| 策略梯度 | 标准 | branch_weight_factor 校正 |

### 1.3 环境要求

需要环境支持状态快照：
- **Atari**：使用 ALE 的 `cloneState()` / `restoreState()`
- **MuJoCo**：使用 `get_state()` / `set_state()`


## 2. 参数配置

在 PPO 基础上新增以下参数：

```python
@dataclass
class Args:
    # ... PPO 原有参数 ...

    # OPTS_TTPO 新增参数
    root_tuct: float = 0.5
    """根节点的 TUCT 基准值，用于与树中节点竞争"""
```

**参数说明：**
- `root_tuct`：根节点 TUCT 的基准值，实际值为 `max(mean(advantages), root_tuct)`


## 3. 数据结构

使用矩阵 + 索引维护树结构，每列代表一棵树（对应一个环境）。
**root 信息单独存储，env_states 存储 next_states（执行动作后的状态）**。

### 3.1 根节点数据（单独存储）

| 变量名 | 类型/形状 | 说明 |
|--------|-----------|------|
| root_states | List[Any], 长度 num_envs | 根节点环境状态快照 |

### 3.2 核心张量

| 张量名 | 形状 | 说明 |
|--------|------|------|
| obs | (num_steps, num_envs, *obs_shape) | 执行动作前的观测（与 PPO 一致） |
| actions | (num_steps, num_envs, *action_shape) | 动作 |
| rewards | (num_steps, num_envs) | 执行动作后的奖励 |
| dones | (num_steps, num_envs) | 执行动作前的终止标志 |
| values | (num_steps, num_envs) | V(obs[t])，当前状态的价值估计 |
| logprobs | (num_steps, num_envs) | 动作对数概率 |
| parent_indices | (num_steps, num_envs) | 父节点索引，-1 表示父节点是 root |
| state_branches | (num_steps, num_envs) | 分支数，初始为 1 |
| tid | (num_steps, num_envs) | 轨迹 ID，用于计算 N_subtree |
| advantages | (num_steps, num_envs) | 优势估计 |
| returns | (num_steps, num_envs) | 回报 |

### 3.3 非张量数据

| 变量名 | 类型 | 说明 |
|--------|------|------|
| env_states | List[List[Any]] | next_states 的环境快照，形状 (num_steps, num_envs) |
| children_indices | List[List[List[int]]] | 子节点索引列表，形状 (num_steps, num_envs) |
| tree_branches | List[Dict[int, int]] | 每棵树的轨迹分支数，tree_branches[env][tid] = 分支数 |
| init_weights | List[Dict[int, int]] | 第一层节点的初始权重，init_weights[env][tid] = 权重 |
| next_tid | List[int] | 每棵树的下一个可用 tid，长度 num_envs |

### 3.4 树结构示例

```
数据对应关系：
    root_state → actions[t] → env_states[t] / obs[t]
    即：actions[t] 导致到达 env_states[t]，rewards[t] 是该转移的奖励

假设 num_steps=5, num_envs=1，树结构如下：

        root
       /    \
    [0]     [3] (回溯到 root)
     |        |
    [1]     [4]
     |
    [2] (终止，触发回溯)

对应的数据：
    parent_indices[:, 0] = [-1, 0, 1, -1, 3]
    state_branches[:, 0] = [1, 1, 1, 1, 1]

    children_indices[0, 0] = [1]
    children_indices[1, 0] = [2]
    children_indices[2, 0] = []  (终止节点)
    children_indices[3, 0] = [4]
    children_indices[4, 0] = []
```


## 4. 训练流程

### 4.1 整体流程图

```
┌─────────────────────────────────────────────────────────────────┐
│  for iteration in iterations:                                    │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │  1. 初始化：reset 所有环境，保存根状态                    │  │
│    └─────────────────────────────────────────────────────────┘  │
│                            ↓                                     │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │  2. 采样循环：for step in range(num_steps):              │  │
│    │     a. 所有环境同步执行一步                               │  │
│    │     b. 保存状态快照                                       │  │
│    │     c. 检查终止的环境                                     │  │
│    │     d. 对终止环境：TreeGAE → TUCT 选择 → 状态恢复         │  │
│    └─────────────────────────────────────────────────────────┘  │
│                            ↓                                     │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │  3. 更新：计算 branch_weight_factor，更新策略和价值网络   │  │
│    └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 采样循环伪代码

```
初始化：
    所有环境 reset
    保存 root_states, root_obs, root_value
    current_obs = root_obs  # 当前观测
    current_parent = [-1] * num_envs  # 当前父节点索引
    pending_action = [None] * num_envs  # 待执行的动作（回溯时传递）
    N_total = [0] * num_envs  # 每棵树的终止轨迹数

    # tid 相关初始化
    tree_branches = [{} for _ in range(num_envs)]  # 每棵树的轨迹分支数
    next_tid = [0] * num_envs  # 下一个可用 tid
    current_tid = [0] * num_envs  # 当前轨迹的 tid
    # 初始轨迹
    for env in range(num_envs):
        tree_branches[env][0] = 1
        next_tid[env] = 1

for step in range(num_steps):

    # 1. 【并行】采样或使用传递的动作
    for each env (并行):
        if pending_action[env] is not None:
            action[env] = pending_action[env]
            pending_action[env] = None
        else:
            action[env] = agent.sample(current_obs[env])

    # 2. 【并行】执行动作
    next_obs, rewards, dones = envs.step(actions)

    # 3. 【并行】保存数据
    for each env (并行):
        保存 actions[step], obs[step]=next_obs, rewards[step], dones[step]
        保存 values[step], logprobs[step]
        保存 env_states[step] = env.get_state()
        parent_indices[step, env] = current_parent[env]

        # 保存 tid
        if current_parent[env] == -1:
            tid[step, env] = current_tid[env]  # 第一层节点
        else:
            tid[step, env] = tid[current_parent[env], env]  # 继承父节点

        # 更新父节点的 children_indices
        if current_parent[env] != -1:
            children_indices[current_parent[env], env].append(step)

    # 4. 处理终止的环境
    terminated_envs = 找出 done=True 的环境

    if terminated_envs 非空:
        # a. 【并行】更新 TreeGAE
        for each terminated_env (并行):
            N_total[env] += 1
            从终止节点回溯到 root，更新 advantages

        # b. 【并行】TUCT 选择（内部计算 branch_weight_factor）
        for each terminated_env (并行):
            selected = TUCT 选择最优节点  # -1 表示 root

        # c. 【并行】状态恢复和 tid 维护
        for each terminated_env (并行):
            if selected == -1:  # 选中 root
                env.set_state(root_states[env])
                current_obs[env] = root_obs[env]
                current_parent[env] = -1
                pending_action[env] = None  # 需要重新采样

                # 创建新轨迹
                current_tid[env] = next_tid[env]
                tree_branches[env][current_tid[env]] = 1
                next_tid[env] += 1
            else:
                # 恢复到 selected 的父状态
                parent = parent_indices[selected, env]
                if parent == -1:
                    env.set_state(root_states[env])
                    current_obs[env] = root_obs[env]
                else:
                    env.set_state(env_states[parent, env])
                    current_obs[env] = obs[parent, env]
                current_parent[env] = parent
                pending_action[env] = actions[selected, env]  # 传递动作
                state_branches[selected, env] += 1

                # 继承 tid 并增加分支数
                current_tid[env] = tid[selected, env]
                tree_branches[env][current_tid[env]] += 1

    # 5. 更新未终止环境的状态
    for each non_terminated_env:
        current_obs[env] = next_obs[env]
        current_parent[env] = step
```


## 5. 关键公式

### 5.1 索引对应关系

```
索引 t 对应的数据（与 PPO 一致）：
    obs[t] = 执行动作前的观测
    values[t] = V(obs[t])
    actions[t] = 在 obs[t] 下采样的动作
    rewards[t] = 执行 actions[t] 后的奖励
    dones[t] = 执行 actions[t] 后是否终止
    env_states[t] = 执行 actions[t] 后到达的状态
```

### 5.2 TreeGAE

从终止节点回溯到 root，计算树结构上的优势估计。对于分支节点，使用子节点优势的均值进行回传。

```
从终止节点开始，沿 parent_indices 回溯到 root（parent=-1）：

V_next 的计算：
    若无子节点且 dones[t] = True：V_next = 0（终止状态）
    若无子节点且 dones[t] = False：V_next = bootstrap_value（未终止叶节点）
    若有子节点：V_next = values[children[0]]（所有子节点状态相同，取第一个即可）

δ_t = rewards[t] + γ * V_next - values[t]

优势计算：
    若无子节点（叶节点）：A_t = δ_t
    若有子节点（分支节点）：A_t = δ_t + γ * λ * mean(children_advs)

说明：
    - 分支节点的子节点代表从同一状态-动作对出发的不同后续轨迹
    - 通过对子节点优势取均值，自然实现了不同轨迹的优势聚合
    - 这与策略梯度中通过 branch_weight_factor 的聚合效果一致
```

### 5.3 TUCT（Tree UCT）

```
TUCT(t) = exploitation * exploration

其中：
    exploitation = A_t / W_t
    exploration = sqrt(log(N_total + 1)) / N_subtree

    W_t = branch_weight_factor（见 5.4）
    N_total = 当前树的终止轨迹总数
    N_subtree = tree_branches[env][tid[t, env]]（通过 tid 映射）

动作聚合：
    相同状态-动作对可能有多个索引（不同后续轨迹），需要聚合后再选择：
    - 相同父节点（parent != -1）的节点是相同动作
    - parent == -1 且相同 tid 的节点是相同动作

    聚合方式：
    1. 按动作身份分组：key = (is_root, tid) if parent == -1 else (is_child, parent)
    2. 计算每组的平均 TUCT
    3. 选择平均 TUCT 最大的组，返回该组第一个索引

根节点的 TUCT：
    root_tuct_value = max(mean(所有节点的 A), root_tuct 参数)

选择规则：
    选择聚合后 TUCT 值最大的动作
    若 root_tuct_value 最大，则选择根节点（selected = -1）
```

### 5.4 tid 和 tree_branches 维护

```
初始化：
    tree_branches[env] = {}  # 空字典
    next_tid[env] = 0

TUCT 选择后：
    if selected == -1:  # 选中根节点
        tid_value = next_tid[env]
        next_tid[env] += 1
        tree_branches[env][tid_value] = 1
    else:  # 选中非根节点
        tid_value = tid[selected, env]
        tree_branches[env][tid_value] += 1

每次 step 后（保存数据时）：
    if parent == -1:
        # 第一层节点，使用当前轨迹的 tid
        tid[step, env] = current_tid[env]
    else:
        # 继承父节点的 tid
        tid[step, env] = tid[parent, env]

计算 N_subtree：
    N_subtree[t] = tree_branches[env][tid[t, env]]
```

### 5.5 Branch Weight Factor

用于 TUCT 选择和策略梯度校正，保证无偏估计：

```
对于 parent_indices[t] == -1 的节点（第一层节点）：
    W_t = init_weights[env][tid[t]]

对于其他节点：
    W_t = W_{parent} * state_branches[parent]

即：W_t = init_weights[tid] * 从第一层节点到 t 路径上所有祖先节点的 state_branches 累乘

init_weights 维护：
    初始化：init_weights[env][0] = 1
    选中 root 时：init_weights[env][new_tid] = 1
    选中第一层节点（parent=-1）时：init_weights[env][tid] += 1
```

### 5.6 策略梯度

```
标准 PPO：
    ∇J = E[∇log π(a|s) * A]

TTPO（树轨迹策略优化）：
    ∇J = E[∇log π(a|s) * A / W]

损失聚合：
    loss = sum(loss_per_step / W) / sum(1 / W)
```


## 6. 实现要点

### 6.1 环境管理

由于需要单独操作每个环境的状态，**不能使用 SyncVectorEnv**：

```python
# 错误：SyncVectorEnv 不支持单独设置状态
envs = gym.vector.SyncVectorEnv([...])

# 正确：使用独立环境列表
envs = [make_env(i)() for i in range(num_envs)]
```

需要包装环境以支持状态快照：
- Atari: `env.unwrapped.ale.cloneState()` / `restoreState()`
- MuJoCo: `env.unwrapped.data.qpos/qvel` 的保存和恢复

### 6.2 并行化策略

| 操作 | 并行方式 |
|------|----------|
| 环境 step | ThreadPoolExecutor |
| 状态快照保存 | ThreadPoolExecutor |
| TreeGAE 更新 | 不同环境可并行 |
| TUCT 选择 | 不同环境可并行 |
| 状态恢复 | ThreadPoolExecutor |

### 6.3 注意事项

1. **状态恢复逻辑**：
   - 选中 root：恢复到 root_state，重新采样动作，创建新轨迹（新 tid）
   - 选中节点 t：恢复到 t 的父状态，传递 actions[t] 到下一循环执行，继承 tid

2. **pending_action**：回溯时传递动作，下一循环直接执行而非重新采样

3. **state_branches 更新**：TUCT 选择后，对被选中的非 root 节点 +1

4. **tid 维护**：
   - 初始化：每个环境从 tid=0 开始，tree_branches[env][0] = 1
   - 选中 root：创建新 tid，tree_branches[new_tid] = 1
   - 选中节点：继承节点的 tid，tree_branches[tid] += 1
   - 保存数据：第一层节点使用 current_tid，其他节点继承父节点的 tid

5. **内存管理**：env_states 存储状态快照，Atari 约几 KB/状态，MuJoCo 约几百字节/状态


## 7. 与 LLM 版本的主要区别

| 特性 | LLM 版本 | CleanRL 版本 |
|------|----------|--------------|
| 树的数量 | 动态创建，全局选择 | 每个环境一棵树 |
| 状态恢复 | token 拼接 | 环境状态快照 |
| 采样轮次 | g 轮，每轮 n 条轨迹 | num_steps 步 |
| 终止处理 | 轨迹自然结束 | 环境 done 信号 |
| 并行方式 | batch 并行 | 环境并行 + 线程池 |
