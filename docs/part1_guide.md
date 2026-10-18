# 第一部分：GD / heavy-ball 理论、实现与验证

负责人：CHENG Zixuan。模块交付日期：2026-10-16。

本模块交付统一接口下的 GD / heavy-ball、旋转对应性证明、谱收敛分析、二维手算与数值验证、可整合的英文报告。现有 `gd_momentum.py` 已有更新和谱公式，本次补充专用验证与报告，未更改公共接口。以下数值是机制验证，不是全组调参后的主实验，也不代替其他成员的交叉 review。

## 1. 在当前 conda 环境复现

在包含 `pyproject.toml` 的项目根目录运行：

```bash
conda activate aisc5020
python -m unittest discover -s tests -v
python -m coordinate_geometry.part1_experiments --config configs/part1.json --output outputs/part1
python -m coordinate_geometry.verify_part1 --output outputs/part1
```

只需数值输出时，在实验命令末尾添加 `--no-plots`。默认会更新同目录输出；保留历史结果时，给实验和复核命令指定相同的新输出目录。

| 文件 | 用途 |
| --- | --- |
| `src/coordinate_geometry/gd_momentum.py` | 统一接口更新、谱半径、稳定性和理论参数 |
| `src/coordinate_geometry/part1_experiments.py` | 通过公共 runner 采集每一步位置，生成配对验证和图 |
| `src/coordinate_geometry/verify_part1.py` | 不调用 runner / 更新函数 / 谱公式助手，独立重算保存结果 |
| `tests/test_gd_momentum.py` | 手算、目标函数界、伴随矩阵、边界、重根暂态、各向同性和非法输入检查 |
| `configs/part1.json` | seed、谱、角度、预算、误差容差 |
| `docs/part1_report.tex` | 英文理论、证明、实测表格和可整合正文；独立 LaTeX 源码 |
| `docs/part1_review_part5.md` | 对第五部分 P4、基歧义和实现对应性的技术复核记录 |
| `outputs/part1/summary.csv` | 每组方法和角度的参数、状态、阈值和对应误差 |
| `outputs/part1/trajectories.json` | 谱、初始点、旋转、完整位置和损失轨迹、配置及运行器计数 |
| `outputs/part1/summary.json` | 环境版本、UTC 时间、源码 SHA256 和配置哈希 |
| `outputs/part1/artifact_verification.json` | 独立复核结果；`passed` 表示实际通过 |
| `outputs/part1/part1_validation.png` | 收敛参数与边界 / 发散对照图 |

原始输出由 `.gitignore` 排除，提交源码、配置和文档即可；分享结果时可另附本地输出。本模块最初在解压目录中完成，因此验证工件使用源码 SHA256 清单记录版本；迁移到 fork 后，也可以用 Git 提交记录跟踪改动。

## 2. 统一符号和更新公式

令 $f(w)=\frac12w^THw$，$H=QDQ^T$，$Q^TQ=I$，$D=\operatorname{diag}(\lambda_i)\succ0$，$\mu=\min_i\lambda_i$，$L=\max_i\lambda_i$，$\kappa=L/\mu$。定义 $z=Q^Tw$，则 $\phi(z)=\frac12z^TDz$，且 $\nabla f(Qz)=QDz$。

GD：$w_{t+1}=w_t-\eta Hw_t$。

本项目 heavy-ball 的 velocity 是**累计梯度**：

$$
v_{t+1}=\beta v_t+Hw_t,\qquad w_{t+1}=w_t-\eta v_{t+1},\qquad v_0=0.
$$

它没有 $(1-\beta)$ 因子，也没有 Adam 的 bias correction。消去速度得到

$$
w_{t+1}=(1+\beta)w_t-\eta Hw_t-\beta w_{t-1},\qquad w_{-1}=w_0.
$$

最后这个初始条件决定暂态，不能把 $w_{-1}$ 默认为零。`RunConfig.beta` 是 heavy-ball 参数，和 Adam 的 `beta1` / `beta2` 分开。

## 3. 旋转对应性证明

两种坐标使用同一标量学习率、相同动量系数和对应初值 $w_0=Qz_0$；heavy-ball 还要求 $v_0=Qu_0$，项目中两者均为零。

GD：若 $w_t=Qz_t$，则 $w_{t+1}=Qz_t-\eta QDz_t=Qz_{t+1}$。

heavy-ball：若 $w_t=Qz_t$、$v_t=Qu_t$，则

$$
v_{t+1}=Q(\beta u_t+Dz_t)=Qu_{t+1},\quad
w_{t+1}=Q(z_t-\eta u_{t+1})=Qz_{t+1}.
$$

归纳得到每一步都对应，且损失、梯度范数和目标阈值步数在精确算术中相同。结论允许重复特征值，也不依赖收敛；浮点发散对照中误差会随轨迹尺度放大。这个结论针对正交变换和标量参数。

## 4. GD 收敛率

特征坐标独立递推：$z_{t,i}=(1-\eta\lambda_i)^tz_{0,i}$。

$$
\rho_{GD}=\max_i|1-\eta\lambda_i|,\qquad
f(w_t)\le\rho_{GD}^{2t}f(w_0).
$$

对任意初值收敛的充要条件是 $0<\eta<2/L$。在边界 $\eta=2/L$，最高特征方向因子为 $-1$，一般不收敛；特定初值若不含不稳定 / 边界模态则可能仍收敛，不能据单次轨迹反推全局稳定。

固定步长的最坏模态最优参数来自平衡端点：

$$
\eta^*_{GD}=\frac2{L+\mu},\qquad
\rho^*_{GD}=\frac{\kappa-1}{\kappa+1}.
$$

这是状态因子；目标函数界要平方。$\eta=1/L$ 时状态因子为 $1-1/\kappa$。

## 5. heavy-ball 稳定性与最优参数

每个特征方向满足

$$
z_{t+1,i}=(1+\beta-\eta\lambda_i)z_{t,i}-\beta z_{t-1,i},
\quad r^2-(1+\beta-\eta\lambda_i)r+\beta=0.
$$

对实系数二次多项式 $r^2-ar+\beta$，根严格位于单位圆内的条件为 $1-a+\beta>0$、$1+a+\beta>0$、$1-\beta>0$。代入 $a=1+\beta-\eta\lambda$，在项目允许的 $0\le\beta<1$ 内得到

$$
0<\eta<\frac{2(1+\beta)}L.
$$

边界有根 $-1$，因此不属于严格稳定。程序以严格不等式判断，不用浮点算出的谱半径是否恰好小于 1 判断边界。

在 SPD 二次型上最小化最坏模态的渐近根模，得到

$$
\eta^*_{HB}=\frac4{(\sqrt L+\sqrt\mu)^2},\quad
\beta^*_{HB}=\left(\frac{\sqrt L-\sqrt\mu}{\sqrt L+\sqrt\mu}\right)^2,\quad
\rho^*_{HB}=\frac{\sqrt\kappa-1}{\sqrt\kappa+1}.
$$

英文报告给出最优性的下界证明。端点出现重根，解包含 $(A+Bt)r^t$；所以 $\rho_{HB}$ 是渐近状态因子，**不能**声称 $f(w_t)\le\rho_{HB}^{2t}f(w_0)$。例如最高特征方向、零初速度下，最优参数产生 $z_t=[1+(1+\rho)t](-\rho)^tz_0$，初期可明显放大。

## 6. 可手算的二维例子

取 $D=\operatorname{diag}(0.01,1)$，$z_0=(1,1)^T$，$Q$ 为 $45^\circ$ 旋转，$w_0=Qz_0=(0,\sqrt2)^T$。

$$
H=\begin{pmatrix}0.505&-0.495\\-0.495&0.505\end{pmatrix},\qquad f(w_0)=0.505.
$$

取 $\eta=1$。GD 和零初速度 heavy-ball 的第一步均为 $z_1=(0.99,0)^T$，故 $w_1=(0.99/\sqrt2,0.99/\sqrt2)^T$，损失为 $0.0049005$。

heavy-ball 再取 $\beta=0.5$，第二步为 $z_2=(0.9751,-0.5)^T$，$w_2=(1.4751/\sqrt2,0.4751/\sqrt2)^T$，损失为 $0.12975410005$。它仍处稳定区，但损失从第一步回升。对应的单元测试验证了这些具体数值。

## 7. 已运行的验证结果

2026-10-16 在 Python 3.11.17 / NumPy 2.4.6 上运行。配置：seed 5020，谱 $(0.01,1)$，$\kappa=100$，角度 $0,30,45,90$ 度，300 步预算，相对损失目标 $10^{-6}$，爆炸阈值 $10^{12}$。所有角度共用同一个随机初始方向，并归一化为初始损失 1，旋转初值由 $w_0=Qz_0$ 得到。

7 组参数 × 4 个角度 = 28 组配对、56 条轨迹。以下是对角参考结果；旋转运行的状态、首次达标步数相同。

| 参数 | $\eta$ | $\beta$ | 状态 / 梯度调用 | 最终相对损失 | 首次达标 |
| --- | --- | --- | --- | --- | --- |
| GD，$1/L$ | 1 | 0 | finite_cap / 300 | $3.30005\times10^{-5}$ | 未达到 |
| GD，理论最优 | 1.980198 | 0 | finite_cap / 300 | $6.14176\times10^{-6}$ | 未达到 |
| GD，边界 | 2 | 0 | finite_cap / 300 | 0.986279 | 未达到 |
| GD，边界外 | 2.1 | 0 | explosive / 146 | $1.20410\times10^{12}$ | 未达到 |
| HB，理论最优 | 3.305785 | 0.669421 | finite_cap / 300 | $1.51030\times10^{-47}$ | 58 |
| HB，边界 | 3.338843 | 0.669421 | finite_cap / 300 | 25.1526 | 未达到 |
| HB，边界外 | 3.505785 | 0.669421 | explosive / 45 | $1.11170\times10^{12}$ | 未达到 |

稳定参数下，最大配对位置误差 $\max_t\|w_t-Qz_t\|/\max(1,\|z_0\|)$ 为 $6.240\times10^{-15}$。包含发散对照后为 $6.849\times10^{-9}$，此时轨迹已增长至约百万量级。检查使用固定初始尺度后的逐分量 `atol=rtol=1e-9`；并非要求所有绝对误差都小于 $10^{-9}$。损失误差按 $\max(1,|f(z_t)|)$ 缩放，所有配对的最大值为 $3.209\times10^{-13}$。

独立复核从 seed 重建初值，并用特征方向二阶递推核对每步位置、损失、梯度范数、停止状态、达标时刻、最终状态、CSV 与操作次数，最大按当前轨迹尺度归一化的重建误差为 $1.204\times10^{-13}$。源码 / 配置 SHA256 也核对通过。

`finite_cap` 仅表示完成预算且未触发异常；GD 未达标属于本预算截断，不是“不收敛”。爆炸对照保留触发步和大损失，没有当作缺失样本删除。每步一条梯度调用，另有损失评估；理论最优参数使用已知谱端点，不应与未知谱下免费调参混为一谈。

## 8. 交接与后续

- 给模块 3：算法名 `gd` / `heavy_ball`，参数 `learning_rate` / `beta`；`step=0` 为初值，固定预算，阈值只记录不早停。无需修改字段。
- 给模块 2：已在本文明确动量的累计梯度定义、零初速度和旋转归纳证明；仍需另一成员实际 review 更新公式，本文不声称已经完成团队交叉复核。
- 给模块 5：已对 P4 证明、signed-permutation 引理和相关实现作技术复核，见 [复核记录](part1_review_part5.md)；没有重跑第五部分历史全量实验。共同特征基中的 Adam 对应性证明结构与这里一致，但不能由 GD / HB 推断普通 Adam 也旋转等变。重复特征值不影响本模块证明。
- 本模块理论推导与机制验证已完成；仍依赖团队冻结 pilot 的参数预算、维度、谱、seed 和统计协议，之后才能运行和解释全组主实验。英文报告已提供，不把本次固定小样例当作 HB 在所有问题上的性能优势。
