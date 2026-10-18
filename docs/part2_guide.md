# 第 2 部分：Adam、固定梯度回放与坐标依赖

本说明对应 `adam.py` 和 `adam_diagnostics.py`。完整英文理论、手算反例与结果范围见 [part2_report.tex](part2_report.tex)，可提取正文整合到总报告。

## 1. 各函数返回的是什么

| 函数 | 输入 | 修改的状态 | 返回 | 是否移动参数 |
| --- | --- | --- | --- | --- |
| `adam_direction()` | 当前梯度、已有状态与配置 | 时间步、一阶矩、二阶矩 | 当前一步方向，形状 `(p,)` | 否 |
| `adam_step()` | 当前梯度、已有状态与配置 | 上述状态和参数位置 | `None` | 是 |
| `replay_directions()` | 梯度历史，形状 `(T,p)` | 内部新建并连续使用的 Adam 矩状态 | 方向历史，形状 `(T,p)` | 否 |
| `replay_defect()` | 梯度历史、正交矩阵与配置 | 两次独立回放各自的矩状态 | 每步分子、分母与相对缺陷 | 否 |

`adam_direction()` 每次只计算一个向量，不会返回整段历史。把各步方向组合成矩阵的是 `replay_directions()`。方向尚未乘学习率；`adam_step()` 才执行：

```python
state.position -= config.learning_rate * adam_direction(state, gradient, config)
```

设第 $t$ 次更新使用更新前的梯度 $g_t=\nabla f(w_{t-1})$，零矩初始化时：

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)(g_t\odot g_t),\\
\widehat m_t&=m_t/(1-\beta_1^t),\qquad
\widehat v_t=v_t/(1-\beta_2^t),\\
A_t&=\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon\mathbf1},\qquad
w_t=w_{t-1}-\eta A_t.
\end{aligned}
$$

平方、开方和除法按分量进行，`epsilon` 是根号外统一的正标量。`state.step` 表示已经执行的矩更新次数，不能每个梯度都重置，否则失去历史与正确的 bias correction。

## 2. 行向量存储与旋转记号

数学上将梯度 $g_t$ 视为列向量；用波浪号表示旋转后的量，上标 $\top$ 只表示转置：

$$
\widetilde g_t=Qg_t,\qquad
\widetilde g_t^\top=(Qg_t)^\top=g_t^\top Q^\top.
$$

代码每一行保存一个梯度，故若 $G$ 的第 $t$ 行是 $g_t^\top$，旋转后的历史是 $\widetilde G=GQ^\top$：

```python
base = replay_directions(history, config)
rotated_history = history @ rotation.T
transformed = replay_directions(rotated_history, config)
```

矩阵乘法 `history @ rotation.T` 负责旋转。`replay_directions()` 本身不旋转：它接收给定梯度，输出 Adam **方向**。`rotated_history` 保存 $Qg_t$，`transformed` 保存 $A'_t=A(Qg_1,\ldots,Qg_t)$；二者形状相同，含义不同。

```text
原梯度历史 G ──回放──> 方向历史 A ──乘 Qᵀ──> 旋转后的方向
      │                                           │
    乘 Qᵀ                                      逐行比较
      │                                           │
      └──> 旋转梯度历史 GQᵀ ──回放──> 方向历史 A'
```

`replay_defect()` 返回：

$$
n_t=\|A'_t-QA_t\|_2,\qquad b_t=\|A_t\|_2,\qquad
\delta_t=\frac{n_t}{b_t+10^{-12}}.
$$

这里的 $10^{-12}$ 只是诊断分母保护值，与 Adam 的 `epsilon` 不同。它提供方向误差的尺度，不能当作损失差、收敛率或任意梯度缩放下严格不变的量。

## 3. 为什么先用 GD 生成固定梯度历史

本项目使用正定二次型 $f(w)=\tfrac12w^\top Hw$，梯度为 $Hw$。本诊断选择 canonical 问题 $H=D$ 对角，以固定谱与旋转问题配对；GD 的旋转对应性本身对任意对称 $H$ 成立。

`canonical_gradient_history()` 需要初始点，并在每次 GD 更新前保存梯度：

$$
g_t=Hw_{t-1},\qquad w_t=w_{t-1}-\alpha g_t.
$$

它返回 $T$ 行梯度，其中第一行就是 $g_1=Hw_0$，并非“不需要初始值”。如果 $\widetilde H=QHQ^\top$ 且 $\widetilde w_0=Qw_0$，则归纳假设 $\widetilde w_{t-1}=Qw_{t-1}$ 给出：

$$
\widetilde g_t
=\widetilde H\widetilde w_{t-1}
=QHw_{t-1}=Qg_t,\qquad
\widetilde w_t=Q(w_{t-1}-\alpha g_t)=Qw_t.
$$

因此不用再运行一次旋转后的 GD，直接旋转已有梯度即可得到严格配对的输入历史。对角情况下还能独立手算：

$$
g_t=D(I-\alpha D)^{t-1}w_0.
$$

回放只更新矩与时间步，不根据 Adam 的输出移动参数或生成下一个梯度，所以它不是一次 Adam 优化。它隔离的问题是：**同一段历史只改变坐标表示后，Adam 的矩积累与归一化响应是否对应？**

## 4. 实际 Adam 轨迹回答另一个问题

`adam_position_history()` 从实际初始点运行 Adam，每次在新位置重新计算梯度。它保存初始点及每次更新后的点，共 `T+1` 行：

$$
w_{t-1}\ \longrightarrow\ g_t=Hw_{t-1}
\ \longrightarrow\ (m_t,v_t)
\ \longrightarrow\ w_t.
$$

`.copy()` 保留各时刻独立快照，避免所有记录指向同一个可变数组。

在旋转问题上另运行一次得到 $\widetilde w_t$，逐步比较：

$$
e_t=\frac{\|\widetilde w_t-Qw_t\|_2}{\max(1,\|w_0\|_2)}.
$$

两条 Adam 轨迹一旦分开，之后输入梯度也可能分开；这个位置误差包含完整反馈。它和固定输入的 replay defect 是不同指标，不能混用。

## 5. 正 epsilon 反例与符号排列控制

零矩开始的第一步经 bias correction 后，$\widehat m_1=g$、$\widehat v_1=g\odot g$，所以 $A(g)=g/(|g|+\epsilon\mathbf1)$。

取 $g=(1,0)^\top$，$Q$ 为 $45^\circ$ 旋转，$a=1/\sqrt2$。于是：

$$
A(Qg)=\frac{(a,a)^\top}{a+\epsilon},\qquad
QA(g)=\frac{(a,a)^\top}{1+\epsilon}.
$$

任意有限 $\epsilon>0$ 下，两个方向都不相同。这是手算反例，不需要把实际代码的 epsilon 设成零，也不依赖某次随机运行。

符号排列是特殊情形。若 $Q$ 为带符号排列，$P=|Q|$ 为无符号排列，则对配对历史归纳可得：

$$
m'_t=Qm_t,\qquad v'_t=Pv_t,\qquad A'_t=QA_t.
$$

二阶矩按无符号排列变换；统一标量 epsilon 保持该关系。二维的 $0^\circ$ 和 $90^\circ$ 因而是对应性控制，而非一般旋转。这个结论也不表示 Adam 对重特征空间内的任意换基都对应；第 5 部分另行研究固定共同特征基。

## 6. 运行与结果边界

在项目环境、仓库根目录执行：

```text
python -m coordinate_geometry.adam_diagnostics --config configs/adam_diagnostics.json --output outputs/adam_diagnostics
```

添加 `--plot` 生成图；绘图需要可选依赖 `.[plots]`。配置、问题、逐步 CSV 和汇总均保存到输出目录。重跑会更新同目录输出，需保留旧版本时指定另一个输出目录。

默认配置使用 $D=\operatorname{diag}(0.1,1)$、seed 502002、初始损失 1、GD 学习率 0.5、Adam 学习率 0.1、50 步及 7 个角度，产生 350 行诊断。检查点为第 1、2、10、50 步。一次固定初值、一个谱和一个学习率下的机制诊断，不构成全组调参后的主实验，也不能产生性能排名或收敛定理。英文报告给出实际复现的结果表与独立复核范围。
