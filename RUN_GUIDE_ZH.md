# 三级倒立摆：模型说明与演示运行指南

更新日期：2026-09-27

## 1. 当前是不是强化学习模型？

**当前使用的是经过回报驱动策略搜索训练的控制策略。准确的算法名称是“轨迹优化初始化 + CEM 直接策略搜索”，属于模型辅助的策略学习方案。**

它不是之前的 SAC 模型，也不是从零开始、完全不利用动力学的纯无模型强化学习。最终交付的策略保存在：

```text
F:\Project\2026-9-AI\triple_pendulum_rl\triple-pendulum-rl\checkpoints\hr\policy.pt
```

训练与运行的区别如下：

| 阶段 | 实际工作 |
|---|---|
| 训练初始化 | 利用 MuJoCo 动力学进行轨迹优化，找到能低速到顶的摆起动作；终端价值初始化使用离散 Riccati 方程 |
| 策略学习 | 从训练轨迹的局部状态样本中学习带阶段输入的仿射反馈策略 |
| 强化学习优化 | 使用 CEM（交叉熵方法）采样候选策略，在仿真中闭环运行，根据连续稳定时长、姿态奖励和越界惩罚优化策略参数 |
| 演示／评测 | 只加载固定的 `policy.pt`，根据实时状态和本局阶段输出小车水平驱动力 |

CEM 优化阶段使用候选策略自己运行得到的回报，不使用教师动作标签。最终训练配置为 30 代、每代 24 个候选策略、每个候选在 32 个随机初始状态上运行 20 秒。

运行时采用同一个策略表达式：

```text
动作 = clip(阶段权重 · (当前状态 − 阶段参考状态) + 阶段偏置, -1, 1)
小车水平力 = 动作 × 500 N
阶段 = min(本局控制步数, 最后阶段索引)
```

状态包括小车位置、三个相对关节角及其速度。权重、偏置和参考状态来自训练；受到扰动时，动作会随实时状态变化，具有闭环反馈能力。策略共有 121 个阶段，最后一个阶段持续使用相同网络计算式。

**运行时没有在线求解 MPC、iLQR 或 LQR，也没有根据高度切换另一个手写控制器。** 训练阶段使用了控制理论与动力学优化，因此介绍项目时应如实说明，不应称为“纯 SAC”或“纯无模型端到端强化学习”。

## 2. 打开实时演示窗口

打开 Windows **PowerShell**，依次执行以下两行：

```powershell
Set-Location "F:\Project\2026-9-AI\triple_pendulum_rl\triple-pendulum-rl"
& "F:\Project\2026-9-AI\triple_pendulum_rl\.venv\Scripts\python.exe" tools/demo_hr.py
```

不需要重新训练，也不需要先激活虚拟环境。该命令会使用已有环境和训练好的模型，弹出 MuJoCo 窗口。

默认演示过程：

1. 从自然下垂附近的随机初态开始。
2. 约 6 秒完成摆起并进入持续稳定区间。
3. 连续保持直立超过 10 秒。
4. 第 18 秒对最上层摆杆施加 5 N 水平外力，持续 0.2 秒。
5. 由同一策略恢复稳定；本局连续运行至 35 秒结束。

全程自动执行，无须键盘控制。提前关闭窗口即可结束演示；默认脚本不会中途重置或循环开始新的一局。

如果要使用**精确自然下垂、零初速度**，运行：

```powershell
Set-Location "F:\Project\2026-9-AI\triple_pendulum_rl\triple-pendulum-rl"
& "F:\Project\2026-9-AI\triple_pendulum_rl\.venv\Scripts\python.exe" tools/demo_hr.py --width 0
```

如果要只演示摆起与保持，不施加外力，运行：

```powershell
Set-Location "F:\Project\2026-9-AI\triple_pendulum_rl\triple-pendulum-rl"
& "F:\Project\2026-9-AI\triple_pendulum_rl\.venv\Scripts\python.exe" tools/demo_hr.py --force 0
```

终端中的 `t` 为仿真时间，`upright` 为当前连续满足直立判据的时长，`cart` 为小车位置，`push` 为外加扰动力。`upright` 在受扰后可能清零，之后重新累计，这是正常的验收计时行为，不是重置物理环境。

## 3. 直接打开已经录好的视频

无需启动仿真，直接执行：

```powershell
Start-Process -FilePath "F:\Project\2026-9-AI\triple_pendulum_rl\triple-pendulum-rl\artifacts\hr\demo\demo.mp4"
```

也可以在资源管理器中进入以下目录，双击 `demo.mp4`：

```text
F:\Project\2026-9-AI\triple_pendulum_rl\triple-pendulum-rl\artifacts\hr\demo
```

该视频为 35 秒、20 FPS 的连续仿真录像，展示了精确下垂起始、摆起、稳定、受扰和恢复的完整过程。

## 4. 自己运行 100 局验收

```powershell
Set-Location "F:\Project\2026-9-AI\triple_pendulum_rl\triple-pendulum-rl"
& "F:\Project\2026-9-AI\triple_pendulum_rl\.venv\Scripts\python.exe" tools/evaluate_gps.py --checkpoint checkpoints/hr/policy.pt --episodes 100 --seed 2000 --width 0.05 --velocity 0.05 --force 5 --out runs/hr_recheck
```

该命令在后台计算仿真，不打开实时窗口。输出文件保存在仓库下的 `runs/hr_recheck/`：

- `report.json`：成功局数、扰动恢复结果、初始状态、模型哈希和完整配置。
- `episode_000.npz` 等：每局时间、位置、速度、动作、外力和连续稳定时长。

重复使用同一输出目录会覆盖上一次该目录内的同名结果；需要保留多次结果时，更换 `--out` 后的目录名。

## 5. 重新生成视频

```powershell
Set-Location "F:\Project\2026-9-AI\triple_pendulum_rl\triple-pendulum-rl"
& "F:\Project\2026-9-AI\triple_pendulum_rl\.venv\Scripts\python.exe" tools/evaluate_gps.py --checkpoint checkpoints/hr/policy.pt --episodes 1 --width 0 --velocity 0 --force 5 --video --out runs/hr_video
```

生成的视频位置：

```text
F:\Project\2026-9-AI\triple_pendulum_rl\triple-pendulum-rl\runs\hr_video\demo.mp4
```

## 6. 已验证结果与成功判据

最终模型固定后，采用未参与训练的随机种子 2000–2099 完成了独立验收。

| 测试内容 | 已保存结果 |
|---|---:|
| 自然下垂附近随机初态，摆起后连续稳定不少于 10 秒 | 100/100 |
| 顶杆受到 +5 N、0.2 秒水平外力，之后再连续稳定不少于 10 秒 | 100/100 |
| 这 100 局中最短的扰动前连续稳定时长 | 12.01 秒 |
| 这 100 局中最短的扰动后连续稳定时长 | 16.22 秒 |
| 最大绝对小车位移 | 小于 0.627 米 |
| 更大下垂初态扰动的额外测试 | 42/50 |
| 最近一次代码回归测试 | 31 passed |

100 局验收的初始噪声范围：小车位置 ±0.05 m，三个相对关节角各 ±0.05 rad；小车初速度标准差 0.05 m/s，三个关节初速度标准差各 0.05 rad/s。额外 50 局测试把上述位置／角度扰动幅度和速度标准差数值扩大至 0.15。

“稳定直立”按每个 0.01 秒物理步检查，要求同时满足：

- 三根杆相对于竖直向上的绝对倾角均不超过 15°。
- 三根杆的绝对角速度均不超过 1 rad/s。
- 小车位置绝对值不超过 0.95 m。
- 上述条件连续成立至少 10 秒。

评测每局只初始化一次，之后仅设置小车控制力及真实外加扰动力，调用 MuJoCo 推进物理状态，不修改摆杆位置或速度，不中途重置。原始模型保留三根被动摆杆、单个小车执行器、±500 N 驱动力上限和 20 Hz 控制频率。

## 7. 能力边界及向 HR 介绍时的表述

当前已验证题目中“从自然下垂出发”的分支，以及下垂附近随机初态的鲁棒性。**没有证明能从任意大角度随机状态出发，也不保证遭到足以完全打翻系统的更强冲击后仍能重新摆起。** 已测试的扰动为顶杆水平外力 5 N、持续 0.2 秒。

策略使用随本局时间前进的阶段输入，阶段不会在受扰后重新开始。这一点应与普通无记忆、只依赖当前物理状态的 SAC/PPO 策略区分。

可以这样介绍：

> 我在 MuJoCo 中搭建了仅由小车水平力驱动的三级串联倒立摆。训练采用轨迹优化初始化，再通过 CEM 回报驱动策略搜索优化一个带阶段输入的反馈策略。运行时只加载这一个策略，完成从下垂状态摆起、连续稳定直立超过 10 秒，以及指定外力扰动后的恢复。最终在约定初始分布上进行了 100 局独立验收，全部通过；任意初态和任意强扰动的全局恢复尚未实现。

如果另有“必须使用 SAC/PPO”或“必须纯无模型强化学习”的要求，当前方法不满足这些额外算法限制，需要另行调整，不能将本方案改称为 SAC。

## 8. 文件索引

以下链接均相对于本文档所在目录：

- [最终策略模型](checkpoints/hr/policy.pt)
- [实时演示脚本](tools/demo_hr.py)
- [批量评测脚本](tools/evaluate_gps.py)
- [完整演示视频](artifacts/hr/demo/demo.mp4)
- [演示数据曲线](artifacts/hr/verification.png)
- [100 局独立验收报告](artifacts/hr/validation/report.json)
- [强化学习优化日志](artifacts/hr/training.jsonl)
- [更详细的技术交付说明和重训命令](docs/HR_DELIVERY.md)

现有旧 SAC/PPO 模型快照保留在原目录，演示默认使用本次交付的 `checkpoints/hr/policy.pt`。
