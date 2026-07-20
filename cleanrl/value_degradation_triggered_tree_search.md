# 基于价值退化触发的树搜索策略梯度

## 1. 核心思想

给定一批由当前策略采样的初始轨迹，我们持续估计每条轨迹从初始状态 \(s_0\) 出发、截至当前位置 \(t\) 已经体现出的预期表现。

当第 \(i\) 条轨迹的前缀估计首次跌破其他轨迹的平均终止表现时，认为当前 rollout 相对于批次参考水平发生了退化。此时不直接丢弃原轨迹，也不只保留搜索到的最优后缀，而是从当前状态展开多个新的 on-policy continuation，形成树轨迹，并通过 TreeGAE 和加权树策略梯度联合学习全部分支。

整体逻辑为：

\[
\text{前缀价值估计}
\rightarrow
\text{检测相对表现退化}
\rightarrow
\text{局部展开搜索}
\rightarrow
\text{构造树轨迹}
\rightarrow
\text{TreeGAE}
\rightarrow
\text{加权树策略梯度更新}.
\]

该方法的重点不是寻找“理论上的最大方差位置”，而是将额外采样预算集中到已经表现出价值退化的轨迹前缀上，搜索可能恢复回报的替代 continuation。

---

## 2. 基于优势缓存的前缀价值估计

### 2.1 TreeGAE 优势

设第 \(i\) 条初始轨迹为

\[
\tau_i=(s_{i,0},a_{i,0},r_{i,0},\ldots,s_{i,T_i}).
\]

在尚未产生分支时，TreeGAE 退化为标准 GAE。记缓存的优势为

\[
\hat A^{\mathrm{TreeGAE}}_{i,t}.
\]

定义第 \(i\) 条轨迹在位置 \(t\) 的前缀价值估计：

\[
\boxed{
M_{i,t}
=
V(s_{i,0})
+
\lambda
\left[
\hat A^{\mathrm{TreeGAE}}_{i,0}
-
(\gamma\lambda)^t
\hat A^{\mathrm{TreeGAE}}_{i,t}
\right].
}
\]

该形式适合实际实现，因为训练框架通常已经缓存每个位置的 advantage，无需重新显式累加所有 TD residual。

对于链式轨迹，

\[
\hat A_{i,0}
-
(\gamma\lambda)^t\hat A_{i,t}
=
\sum_{k=0}^{t-1}(\gamma\lambda)^k\delta_{i,k},
\]

其中

\[
\delta_{i,k}
=
r_{i,k}
+
\gamma V(s_{i,k+1})
-
V(s_{i,k}).
\]

因此

\[
M_{i,t}
=
V(s_{i,0})
+
\sum_{k=0}^{t-1}\gamma^k\lambda^{k+1}\delta_{i,k}.
\]

也就是说，尽管 \(M_{i,t}\) 可以通过缓存的 TreeGAE 优势计算，它实际上只包含从 \(0\) 到 \(t\) 的前缀信息。

### 2.2 终止估计

在终止位置 \(T_i\)，令

\[
\hat A^{\mathrm{TreeGAE}}_{i,T_i}=0,
\]

则

\[
\boxed{
M_{i,T_i}
=
V(s_{i,0})
+
\lambda\hat A^{\mathrm{TreeGAE}}_{i,0}.
}
\]

它表示利用完整轨迹信息得到的初始状态价值估计。

---

## 3. 留一参考基线

对于第 \(i\) 条轨迹，使用其他轨迹的终止估计构造留一参考：

\[
\boxed{
M_T^{(-i)}
=
\frac{1}{N-1}
\sum_{j\neq i}M_{j,T_j}.
}
\]

使用 leave-one-out reference 有两个作用：

1. 避免第 \(i\) 条轨迹自身的未来终止结果进入其分支判定；
2. 将当前轨迹的前缀表现与同批次其他轨迹的平均完整表现进行比较。

\(M_T^{(-i)}\) 可以理解为当前策略在该批任务或初始状态分布上的经验参考水平。

---

## 4. 价值退化触发规则

定义第 \(i\) 条轨迹的分支位置为

\[
\boxed{
\tau_i
=
\inf
\left\{
t:
M_{i,t}<M_T^{(-i)}
\right\}.
}
\]

若不存在满足条件的位置，则该轨迹不产生额外分支。

由于选择的是首次跌破位置，因此在 \(\tau_i>0\) 时：

\[
M_{i,t}\ge M_T^{(-i)},
\qquad
\forall t<\tau_i,
\]

而

\[
M_{i,\tau_i}<M_T^{(-i)}.
\]

因此，\(\tau_i\) 表示该轨迹的前缀价值估计第一次从参考水平之上跌到参考水平之下的位置。

等价的 crossing 表达为

\[
M_{i,t-1}\ge M_T^{(-i)},
\qquad
M_{i,t}<M_T^{(-i)}.
\]

这里检测到的不是某一步的即时奖励下降，而是截至当前位置累计前缀信息所反映出的整体 continuation 退化。

### 可调阈值

更一般地，可以使用

\[
\boxed{
\tau_i
=
\inf
\left\{
t:
M_{i,t}<M_T^{(-i)}+\beta
\right\}.
}
\]

其中：

- \(\beta=0\)：标准的均值跌破规则；
- \(\beta>0\)：提前触发，在真正跌破平均水平之前开始搜索；
- \(\beta<0\)：更保守，只在明显退化后触发。

---

## 5. 从退化位置展开局部搜索

在触发状态 \(s_{i,\tau_i}\) 处，从当前策略独立采样 \(K\) 条新 continuation：

\[
\tau_{i,\tau_i}^{(1)},\ldots,
\tau_{i,\tau_i}^{(K)}
\sim
\pi_\theta(\cdot\mid s_{i,\tau_i}).
\]

搜索的目的不是直接选择回报最高的分支，而是增加该状态下不同动作和后续轨迹的覆盖：

\[
\text{degraded continuation}
\rightarrow
\text{multiple alternative continuations}.
\]

原始 continuation 必须保留，并与新增分支共同构成树轨迹。所有搜索分支也必须保留，不能只训练其中回报最高的分支，否则会引入 best-of-\(K\) selection bias。

在固定搜索预算下，可以采用：

- 单次展开：每条根轨迹只在首次退化位置展开一次；
- 递归展开：对新增分支继续应用同一触发规则，直到达到最大深度、最大节点数或 rollout 预算。

---

## 6. TreeGAE

设树中的状态节点为 \(x\)，其状态为 \(s_x\)。节点 \(x\) 有 \(m(x)\) 条出边，第 \(j\) 条边对应

\[
(s_x,a_{x,j},r_{x,j},s_{x'_j}).
\]

为每条出边指定不参与梯度回传的聚合权重

\[
\alpha_{x,j}\ge 0,
\qquad
\sum_{j=1}^{m(x)}\alpha_{x,j}=1.
\]

### 6.1 边级 TD residual

\[
\boxed{
\delta_{x,j}
=
r_{x,j}
+
\gamma V(s_{x'_j})
-
V(s_x).
}
\]

### 6.2 边级 TreeGAE

从叶节点向根节点反向折叠：

\[
\boxed{
\hat A^{\mathrm{TreeGAE}}_{x,j}
=
\delta_{x,j}
+
\gamma\lambda
\hat A^{\mathrm{TreeGAE}}_{x'_j}.
}
\]

其中叶节点满足

\[
\hat A^{\mathrm{TreeGAE}}_{x}=0
\]

或在时间截断时使用相应的 value bootstrap。

### 6.3 节点级 TreeGAE

\[
\boxed{
\hat A^{\mathrm{TreeGAE}}_{x}
=
\sum_{j=1}^{m(x)}
\alpha_{x,j}
\hat A^{\mathrm{TreeGAE}}_{x,j}.
}
\]

对于非分支节点，\(m(x)=1\)，TreeGAE 自动退化为标准 GAE：

\[
\hat A_x
=
\delta_x+\gamma\lambda\hat A_{x'}.
\]

TreeGAE 的含义是：每个子分支先独立计算其 edge advantage，再按照分支聚合权重从叶到根折叠整棵树。

---

## 7. 加权树策略梯度

定义节点访问权重 \(w_x\)。根节点满足

\[
w_{x_0}=1,
\]

子节点权重递归为

\[
w_{x'_j}=w_x\alpha_{x,j}.
\]

树策略梯度估计为

\[
\boxed{
\hat g_{\mathrm{tree}}
=
\sum_{x}
w_x
\sum_{j=1}^{m(x)}
\alpha_{x,j}
\nabla_\theta
\log\pi_\theta(a_{x,j}\mid s_x)
\hat A^{\mathrm{TreeGAE}}_{x,j}.
}
\]

直观上，每个分支的梯度贡献由两部分共同决定：

1. 到达该节点的路径权重 \(w_x\)；
2. 当前节点对该出边分配的局部权重 \(\alpha_{x,j}\)。

当整棵树退化为一条链时，

\[
m(x)=1,\qquad \alpha_{x,1}=1,
\]

上述估计退化为标准链式策略梯度。

---

## 8. 为什么搜索失败的问题可以缓解

某些触发状态可能已经进入几乎不可恢复的低奖励区域，即使展开多个分支也难以恢复回报。

该问题可以通过两种机制缓解。

### 8.1 价值退化向前传播

如果某个动作经常将策略带入不可恢复区域，则随着 critic 学习，

\[
V(s_t)
\]

会下降，并通过 TD/Bellman backup 影响更早的状态和动作：

\[
Q(s_{t-1},a_{t-1})
\approx
r_{t-1}+\gamma V(s_t).
\]

因此，价值退化信号会倾向于逐渐向前传播，使后续训练迭代更早触发分支，在真正进入不可恢复区域之前开始搜索。

该机制依赖于：

- critic 具有足够的拟合精度；
- 坏状态被重复访问；
- TD 信号能够充分向前传播；
- actor 与 critic 的更新速度不过度失衡。

### 8.2 提高触发阈值

令

\[
\beta>0,
\]

则触发条件变为

\[
M_{i,t}<M_T^{(-i)}+\beta.
\]

这样可以在轨迹尚未跌破平均终止表现时提前展开搜索，降低进入不可恢复区域后才开始搜索的概率。

---

## 9. 无偏性与实现约束

分支规则本身只决定在哪里增加采样预算。要使树搜索不额外引入选择偏差，需要满足：

1. **On-policy branching**  
   所有新分支动作和后续轨迹均由当前策略采样。

2. **Prefix-measurable trigger**  
   \(M_{i,t}\) 只依赖当前轨迹截至 \(t\) 的前缀信息；参考值 \(M_T^{(-i)}\) 不包含当前轨迹自身的未来结果。

3. **No gradient through branching decisions**  
   分支位置、阈值和聚合权重均不参与梯度回传。

4. **Retain all branches**  
   原始 continuation 与全部搜索 continuation 均进入 TreeGAE 和策略梯度估计，不能只保留成功分支。

5. **Correct tree weighting**  
   使用路径权重和局部分支权重进行树梯度聚合，避免分支较多的节点被无意中过度计权。

6. **Consistent bootstrap**  
   真正终止状态的 bootstrap 为零；时间截断状态使用 critic value。

在这些条件下，自适应搜索不会额外改变每条被采样分支的 on-policy 性质。若 edge advantage 本身是无偏的，则树策略梯度保持无偏；使用实际 TreeGAE 时，估计器继承与标准 GAE 相同的 critic approximation 和 \(\lambda\)-bias，而不会额外产生由“只选好分支”造成的选择偏差。

---

## 10. 算法流程

```text
Input:
    current policy πθ
    value function Vφ
    batch size N
    branching factor K
    threshold offset β
    discount γ and trace parameter λ

1. Sample N root trajectories from πθ.

2. Compute and cache chain TreeGAE advantages.
   Before branching, TreeGAE is equivalent to standard GAE.

3. For every trajectory i:
       Compute terminal estimate M_{i,T_i}.
       Compute leave-one-out reference M_T^{(-i)}.

4. For every trajectory i, scan t from front to back:
       Compute
           M_{i,t}
           = V(s_{i,0})
             + λ[A^{TreeGAE}_{i,0}
                 - (γλ)^t A^{TreeGAE}_{i,t}].

       If
           M_{i,t} < M_T^{(-i)} + β,
       select the first such t as τ_i and stop scanning.

5. At s_{i,τ_i}, sample K additional on-policy continuations.
   Retain the original continuation and every sampled continuation.

6. If recursive search is enabled, repeat the trigger-and-branch procedure
   on newly generated branches subject to the search budget.

7. After the complete tree is collected:
       Compute TreeGAE from leaves to roots.
       Compute weighted tree policy gradient.
       Update actor and critic.
```

---

## 11. 方法定位

该方法可以定位为：

- **Value-Degradation-Triggered Search**
- **Performance-Triggered Trajectory Branching**
- **Search-Augmented Tree Policy Gradient**

一句话概括：

> 当一条轨迹的前缀价值估计首次跌破其他轨迹的平均终止价值时，从当前状态展开额外的 on-policy continuation，以搜索能够恢复回报的替代后缀，并通过 TreeGAE 与加权树策略梯度联合学习完整树轨迹。
