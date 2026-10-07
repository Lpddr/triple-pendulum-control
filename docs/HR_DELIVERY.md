# 三级倒立摆：可复现交付与验收记录

本次实现走题目允许的“从自然下垂出发”分支。MuJoCo 原始装置没有修改：三根串联被动摆杆、仅小车一个水平驱动力、500 N 上限、20 Hz 控制、RK4、0.01 s 物理步长、重力 9.81 m/s²、原导轨限位。旧 SAC/PPO 模型和用户列出的所有快照均保留。

## 直接运行

在 PowerShell 中进入仓库：

```powershell
cd F:\Project\2026-9-AI\triple_pendulum_rl\triple-pendulum-rl
# 实时窗口：35 秒连续运行，第 18 秒自动向顶杆施加外力。
& ..\.venv\Scripts\python.exe tools/demo_hr.py

# 独立批量验收，结果和逐局轨迹写入指定目录。
& ..\.venv\Scripts\python.exe tools/evaluate_gps.py --checkpoint checkpoints/hr/policy.pt --episodes 100 --seed 2000 --width 0.05 --velocity 0.05 --force 5 --out runs/hr_recheck

# 生成完整视频，不需要手动操控。
& ..\.venv\Scripts\python.exe tools/evaluate_gps.py --checkpoint checkpoints/hr/policy.pt --episodes 1 --width 0 --velocity 0 --force 5 --video --out runs/hr_video
```

现成文件：

- 模型：[checkpoints/hr/policy.pt](../checkpoints/hr/policy.pt)
- 35 秒原速演示：[demo.mp4](../artifacts/hr/demo/demo.mp4)
- 曲线：[verification.png](../artifacts/hr/verification.png)
- 最终独立验收：[report.json](../artifacts/hr/validation/report.json)
- 演示原始状态、动作、外力记录：[episode_000.npz](../artifacts/hr/demo/episode_000.npz)
- 强化学习日志：[training.jsonl](../artifacts/hr/training.jsonl)

## 实测结果

最终模型固定后，使用未参与训练的种子 **2000–2099** 测试，100 局全部成功。每局连续运行 35 秒；初始小车位置、三个相对关节角均叠加 U(-0.05, 0.05) 噪声，四个初速度叠加 N(0, 0.05²) 噪声。下垂基准状态为 `[cart_x, q1, q2, q3] = [0, π, 0, 0]`。

| 验收项 | 结果 |
|---|---:|
| 自然下垂附近起始、摆起后连续稳定 ≥10 s | 100/100 |
| 第 18 秒顶杆水平受力 +5 N，持续 0.2 s，随后再连续稳定 ≥10 s | 100/100 |
| 所有局中最短的扰动前连续稳定时长 | 12.01 s |
| 所有局中最短的扰动后连续稳定时长 | 16.22 s |
| 所有局中的最大小车位移绝对值 | 0.627 m 以下 |
| 控制力范围 | ±500 N |
| 每局初始化次数 | 1 |

零初速、精确自然下垂演示中，约 5.98 s 进入持续稳定区间，15.98 s 首次完成连续 10 秒直立；18.0–18.2 s 施加外力；随后约 18.78 s 恢复满足稳定判据。全程无剪辑、无策略切换、无状态修正、无中途重置。

更大的下垂附近扰动（±0.15 rad/m、初速度标准差 0.15，种子 1000–1049）测得 **42/50**，见 [wide_report.json](../artifacts/hr/wide_report.json)。该测试不是最终 100 局验收的同一初始分布，不应混淆。

**能力边界**：这不是任意大角度状态的全局控制器。阶段输入随本局时间前进；若受到足以完全打翻系统的巨大冲击，不能保证重新摆起。已验收的外力是顶杆 5 N × 0.2 s，不能据此宣称任意强度扰动均可恢复。题目没有要求无模型算法，也没有指定必须使用 SAC/PPO；本实现采用模型辅助的直接策略搜索。若另有“必须纯无模型”限制，则此实现不满足那个额外限制。

## 单一策略与强化学习如何实现

最终算法是**轨迹优化初始化 + CEM 直接策略搜索**。不将它表述成已经成功训练的 SAC，也不宣称完整复现了某篇 Guided Policy Search 算法。

1. 训练期间使用真实 MuJoCo RK4 动力学的有限差分，优化完整摆起轨迹。目标同时约束到顶姿态、到顶速度、小车位移和动作代价；从动作层面解决“送到顶但刹不住”。局部终端价值使用离散 Riccati 方程初始化，这些计算仅存在于训练工具。
2. 用局部状态采样、最小二乘学习一个带阶段输入的仿射网络，避免小型 MLP 的近似误差在三级摆中被迅速放大。参考状态和网络权重都是训练产物。
3. 对这个网络进行 CEM 回报优化：30 代，每代 24 个候选策略，每个候选在 32 个随机下垂初始状态上连续闭环运行 20 秒；只依靠候选自己的稳定时长、姿态奖励和越界惩罚选择精英、更新参数分布。这一阶段不使用教师动作标签。训练种子为 7。
4. 部署只读取一个 PyTorch 模型。每个控制周期采用相同计算式：

   `a = clip(w(phase) · (state − reference(phase)) + b(phase), −1, 1)`

   `phase = min(本局控制步数, 最后一个阶段索引)`。

   状态为四个位置和四个速度；动作只有小车水平驱动力。网络共 121 个阶段、1089 个可学习仿射系数。阶段是策略观测的一部分，末阶段继续使用同一网络表达式。没有按高度触发的控制器选择，没有在线 MPC/iLQR/LQR 调用，也没有扰动后的相位重置。

这种结构具有反馈能力：受到扰动后，即使相位不变，动作仍随实时状态误差变化。它不是开环播放动作序列。它也不是通用的无记忆状态策略；相位依赖和适用初始分布应如实说明。

方法参考：[Guided Policy Search via Approximate Mirror Descent](https://papers.neurips.cc/paper_files/paper/2016/hash/a00e5eb0973d24649a4a920fc53d9564-Abstract.html) 介绍了通过轨迹教师训练参数化策略的思路；[Learning Tetris Using the Noisy Cross-Entropy Method](https://direct.mit.edu/neco/article/18/12/2936/7108/Learning-Tetris-Using-the-Noisy-Cross-Entropy) 展示了回报驱动的交叉熵策略搜索。

## 验收口径和反作弊约束

成功不再只看 `tip_y > 1.7`。每个 **0.01 s 物理步**检查：

- 三根杆的**绝对角度**（相对关节角的累加，再周期归一化）均在竖直方向 ±15° 内。
- 三根杆的绝对角速度均不超过 1 rad/s。
- 小车位移绝对值不超过 0.95 m。
- 连续满足上述条件 ≥10 s；任何一个物理步不满足就清零连续计时。

每局只在开始设置一次 `qpos/qvel`。后续只写单个 `ctrl` 和用于扰动的 `xfrc_applied`，调用 `mj_step` 推进时间。外力施加到顶杆刚体，绝不直接改速度。评测文件中的相位也不会在扰动时重置。

报告包含模型 SHA256、MJCF SHA256、初始状态、随机种子、成功时长、作用力和每局原始轨迹。最终模型 SHA256：

```text
7e621269b735615c657506aa4eb6eeffdd494740838795adb5c5605195f36392
```

## 从头复现训练

沿用现有 `.venv` 即可，不需要 CUDA、新增仿真引擎或安装额外训练依赖。运行前将 PyTorch/BLAS 线程数设为 1，MuJoCo rollout 使用独立的少量工作线程。

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& ..\.venv\Scripts\python.exe tools/search_trajectory.py
& ..\.venv\Scripts\python.exe algorithms/gps/train_phase.py --reference runs/gps/search_6.npz --out runs/gps/phase6
& ..\.venv\Scripts\python.exe algorithms/gps/refine_policy.py --checkpoint runs/gps/phase6/policy.pt --out runs/gps/robust6 --episodes 32 --population 24 --width 0.15 --generations 30
& ..\.venv\Scripts\python.exe tools/evaluate_gps.py --checkpoint runs/gps/robust6/policy.pt --episodes 100 --seed 2000 --width 0.05 --velocity 0.05 --force 5 --out runs/hr_retrained_validation
```

轨迹搜索脚本比较 12 个初始化；第 6 号轨迹经过训练用随机样本比较后用于本次交付。浮点库、MuJoCo 或 PyTorch 版本变化可能改变重新训练的数值结果；最终交付模型和验收报告是固定文件，复核行为无需重训。重复上述命令会覆盖相应的新实验目录，不会修改旧 SAC 快照。

## 原有代码中修复的问题

- SAC 工厂未传入 `capture_reward`、`capture_vel_w`、`near_vel_std`；阻尼设置也未应用。相应历史调参不能作为这些变量已被排除的证据。
- 旧评测 CLI 漏传控制频率和时长；传入时长时又可能重复设置同名参数。
- 初速度选项未生效；摆起模式下却按普通平衡任务提前终止。
- “最大角度”“最大位移”和动作饱和比例取的是最后一步，未统计整段轨迹。
- 物理步后补 `mj_forward`，令观测用几何位置对应当前状态。

修复后已增加参数链路、摆起评测、绝对角度判据、RK4 线性化等回归检查；测试结果 **31 passed**。原有 25 个测试仍全部通过。此前“20 Hz 物理上无法捕获”以及“摆起样本必然污染捕获”的说法，不能作为已证实结论继续引用。
