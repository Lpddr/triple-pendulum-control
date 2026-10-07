# Triple Pendulum Control with Model-Assisted Reinforcement Learning

基于 MuJoCo、Gymnasium 与 PyTorch 的小车三级串联倒立摆控制工程。最终策略采用**轨迹优化初始化 + CEM 回报驱动策略搜索**，以一个带阶段输入的反馈策略完成从自然下垂起始的摆起、稳定直立和指定外力扰动后的恢复。

本工程基于 [Zac Westbrook 的 triple-pendulum-rl](https://github.com/zw22x/triple-pendulum-rl) 扩展，保留原始 Git 历史、作者信息、PPO/SAC 实现及基准模型。下方的原项目文档描述平衡任务；本次摆起交付的详细算法和能力边界见 [技术交付说明](docs/HR_DELIVERY.md)。

## 运行交付策略

推荐 Python 3.11。在仓库根目录安装运行与测试依赖后启动：

```bash
python -m pip install -e ".[hr,dev]"
python tools/demo_hr.py
```

演示使用已提交的 `checkpoints/hr/policy.pt`，无需重新训练。每局连续运行 35 秒，第 18 秒对顶杆施加 5 N、持续 0.2 秒的水平外力。演示时仅小车执行器输出驱动力；没有在线轨迹求解、手写控制器切换或中途重置。

- [完整演示视频](artifacts/hr/demo/demo.mp4)
- [状态与控制曲线](artifacts/hr/verification.png)
- [100 局独立验收报告](artifacts/hr/validation/report.json)
- [中文运行指南](RUN_GUIDE_ZH.md)

```bash
python tools/evaluate_gps.py --checkpoint checkpoints/hr/policy.pt --episodes 100 --seed 2000 --width 0.05 --velocity 0.05 --force 5 --out runs/hr_recheck
python -m pytest -q
```

已保存结果：下垂附近随机初态独立验收 **100/100**，扰动恢复 **100/100**；31 个测试通过。稳定判据同时检查三根杆的绝对倾角不超过 15°、绝对角速度不超过 1 rad/s、小车位移不超过 0.95 m，并要求连续保持至少 10 秒。

这些结果适用于报告中声明的初始分布及扰动力度；任意大角度初态或足以完全打翻系统的强冲击尚未验证。当前方法属于模型辅助策略学习，未宣称纯无模型 SAC/PPO 训练成功。

## 原项目：PPO / SAC 平衡任务

> **HR 摆起任务交付（2026-09-27）：** 新的单一学习策略已从自然下垂附近随机初态完成摆起、连续直立与顶杆外力扰动恢复。独立 100 局验收 100/100 通过；适用范围、视频、完整报告和运行命令见 [交付说明](docs/HR_DELIVERY.md)。实时演示：`python tools/demo_hr.py`。下方是原项目的平衡任务说明，旧模型不应视为已完成摆起任务。

A custom continuous-control benchmark built on Gymnasium and MuJoCo, with from-scratch PyTorch implementations of Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) that solve it.

`InvertedTriplePendulum-v0` extends Gymnasium's inverted double pendulum with a third 0.6 m link. The task is deliberately underactuated: the cart is the only actuator and all three pole hinges are passive, so one bounded horizontal force has to stabilize a coupled, unstable, nonlinear system with three angular degrees of freedom. Both trained policies hold the pendulum upright for the full 1,000-step episode at the reset distribution they were trained on.

The point of the project is the engineering rather than the benchmark score — the MJCF model, the environment wrapper, the reward design, two independent actor-critic training loops, vectorized collection with correct time-limit handling, and a physics/safety test suite.

## Quick Start

```bash
git clone <repository-url>
cd triple-pendulum
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install torch wandb glfw
```

```bash
pytest -q                                   # 25 environment and physics tests
python examples/demo_random.py --episodes 3 # random policy in the viewer
```

Watch a trained policy. With no `--checkpoint` it picks the newest `.pt` under `checkpoints/`:

```bash
python evaluation/viz.py --checkpoint checkpoints/sac/sac_triple_final.pt
```

Train from scratch with the checked-in defaults. Both trainers log to W&B under the `triple-pendulum-RL` project, so configure W&B first or set `WANDB_MODE=offline`:

```bash
python algorithms/ppo/train.py       # 10,000,000 steps, ~90 min on CPU
python algorithms/sac/sac_train.py   # 1,000,000 steps, ~19 min on CPU
```

Hyperparameters live in the `Config` and `cfg` dataclasses inside those two scripts; neither exposes a command-line interface.

## Architecture

```text
                    triple_pendulum/assets/inverted_triple_pendulum.xml
                                        │  rigid-body dynamics, joints, actuator, gravity
                                        ▼
                    triple_pendulum/env.py  (InvertedTriplePendulumEnv)
                                        │  observation, reward, termination, diagnostics
                                        ▼
                          Gymnasium SyncVectorEnv (16 envs)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        algorithms/ppo/train.py                  algorithms/sac/sac_train.py
        rollout buffer → GAE → clipped           replay buffer → twin-Q / actor / α
        policy and value updates                 updates, Polyak-averaged targets
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                        W&B logs  ·  checkpoints  ·  evaluation/viz.py
```

The environment contains no learning logic: it converts simulator state into observations, reward, and termination flags, and each trainer owns its own buffering and update rule.

```text
triple-pendulum/
├── triple_pendulum/
│   ├── __init__.py                         # Gymnasium registration
│   ├── env.py                              # observation, reward, reset, termination
│   └── assets/inverted_triple_pendulum.xml # MuJoCo cart + three-link model
├── algorithms/
│   ├── ppo/
│   │   ├── model.py                        # ActorCritic
│   │   └── train.py                        # collection, GAE, clipped updates
│   └── sac/
│       ├── sac_model.py                    # squashed actor and twin Q critics
│       └── sac_train.py                    # replay and target-network training loop
├── evaluation/
│   └── viz.py                              # deterministic PPO/SAC checkpoint viewer
├── checkpoints/{ppo,sac}/                  # retained checkpoint artifacts
├── examples/
│   ├── demo_random.py                      # render/sanity-check utility
│   └── train_sb3.py                        # optional Stable-Baselines3 PPO baseline
├── tests/
│   ├── test_env.py                         # API, spaces, render, observation smoke tests
│   └── test_physics_and_safety.py          # dynamics, reward, time-limit, safety checks
├── docs/
│   ├── HOW_IT_WORKS.md                     # implementation walkthrough
│   └── SAC_SPS_SPEEDUP.md                  # SAC throughput redesign note
├── pyproject.toml
└── requirements.txt
```

`triple_pendulum` is the only installed package. The other directories are plain folders: each trainer imports its network module as a sibling file, and `viz.py` adds the repository root to `sys.path` so it can reach both. That is why every command above runs a script by path rather than with `python -m`.

## Environment

Importing the package registers the environment:

```python
import gymnasium as gym
import triple_pendulum  # registers InvertedTriplePendulum-v0

env = gym.make("InvertedTriplePendulum-v0")
```

`max_episode_steps=1000` is registered with Gymnasium, so `gym.make` wraps the environment in `TimeLimit`. The raw environment always returns `truncated=False`; the wrapper owns truncation.

### The MuJoCo model

| Component | Implementation |
|---|---|
| Cart | One slider joint along the x-axis, limited to `[-1, 1]` m |
| Links | Three 0.6 m capsule links with hinge joints about the y-axis |
| State coordinates | Cart position plus three hinge angles (`nq=4`, `nv=4`) |
| Actuation | One control-limited motor on the slider; control range `[-1, 1]`, gear `500` |
| Physics | Gravity `(0, 0, -9.81)`, RK4 integration, timestep `0.01` s |
| Control rate | `frame_skip=5`, so a control interval of about `0.05` s |
| Task geometry | Fully upright tip height is 1.8 m; a tip site on link three drives reward and termination |

The scene also carries lighting, a skybox, and non-colliding decorative geometry, which affect rendering only.

### Observation space

`Box(-inf, inf, (12,), float64)`, taken directly from MuJoCo state with no running mean/variance normalization. Trainers cast to `float32` before the forward pass.

| Indices | Quantity | Treatment |
|---:|---|---|
| 0 | Cart position $x$ | Raw position |
| 1–3 | $\sin\theta_1, \sin\theta_2, \sin\theta_3$ | Angle representation |
| 4–6 | $\cos\theta_1, \cos\theta_2, \cos\theta_3$ | Angle representation |
| 7–10 | Cart velocity and the three hinge velocities | Each clipped to $[-10, 10]$ |
| 11 | x component of the cart constraint force | Clipped to $[-10, 10]$ |

The sine/cosine pairs avoid the discontinuity at angular wraparound. Reset adds uniform position noise and Gaussian velocity noise scaled by `reset_noise_scale` — default `0.1`, but both trainers use `0.05`.

### Action space

`Box(-1, 1, (1,), float32)`: a scalar control for the cart motor, matching the MJCF actuator's control range and gear of 500.

SAC emits a `tanh`-squashed Gaussian, so its actions are bounded by construction. PPO samples from an unsquashed Gaussian and relies on the actuator's control range to bound what actually gets applied; the viewer additionally clips PPO's mean action to `[-1, 1]`.

### Reward

With $x_{tip}$ the tip's lateral coordinate, $h_{tip}$ its height (exposed in `info` as `tip_y`), and $\omega_i$ the three hinge angular velocities:

$$
r = r_{alive} - 0.01\,x_{tip}^{2} - (h_{tip} - 1.8)^{2} - 10^{-3}\omega_1^2 - 5\cdot10^{-3}\omega_2^2 - 10^{-2}\omega_3^2
$$

$r_{alive}$ is `healthy_reward` (default `10.0`) while the tip is healthy and zero on a terminating step. The height target is the physically reachable upright height, and the top link carries the largest velocity penalty. Each step's `info` also breaks out `reward_survive`, `distance_penalty`, `velocity_penalty`, `tip_x`, and `tip_y`, so the components are inspectable without reconstructing them from observations.

### Termination vs truncation

- `terminated=True` when `tip_y <= termination_height` (default `1.5`) — a fallen pendulum, with the alive bonus removed on that step.
- `truncated=True` when `TimeLimit` reaches 1,000 steps — an artificial boundary, not a failure.

Both trainers respect the distinction, which is what makes the bootstrapping correct. PPO bootstraps the value of a time-limited final state into its rollout return. SAC records only physical terminations in its replay `done` field, so its TD target bootstraps across truncations. Under same-step vector auto-reset, both read `infos["final_obs"]` rather than mistaking the next episode's reset observation for the successor state.

## Algorithms

Both were trained on the same environment with the same discount (`0.99`), hidden width (256), learning rate (`3e-4`), seed (`1`), and reset noise (`0.05`), on CPU.

### PPO

An `ActorCritic` with separate actor and value MLPs. It collects synchronous rollouts from 16 environments, computes generalized advantage estimates, then runs several epochs of clipped policy and value updates.

| Setting | Value |
|---|---:|
| Total environment steps | 10,000,000 |
| Environments | 16 `SyncVectorEnv` |
| Rollout horizon | 512 steps/env |
| Batch size | $16 \times 512 = 8{,}192$ transitions/update |
| Update epochs / minibatches | 10 / 32 (minibatch 256) |
| Discount $\gamma$ / GAE $\lambda$ | 0.99 / 0.95 |
| Clip coefficient | 0.2 |
| Value / entropy coefficients | 0.5 / 0.0 |
| Adam learning rate | $3\times10^{-4}$, linearly annealed |
| Gradient-norm limit | 0.5 |
| Approximate-KL stop threshold | 0.02 |

The GAE recursion is

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad \hat A_t = \delta_t + \gamma\lambda\hat A_{t+1}
$$

The policy loss uses the standard clipped ratio $\min(r_t\hat A_t, \operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat A_t)$, and the value loss is clipped around the old value estimate. Checkpoints are written every 50 updates as `ppo_InvertedTriplePendulum-v0_<step>.pt`.

### SAC

Written as its own loop rather than a variation on PPO. It maintains an actor, twin critics $Q_1, Q_2$, and matching target critics. After a random warm-up, every vector-environment step inserts a batch of transitions into a uniform replay buffer and performs one gradient update.

| Setting | Value |
|---|---:|
| Total environment steps | 1,000,000 |
| Environments | 16 `SyncVectorEnv` |
| Replay capacity / sample batch | 1,000,000 / 256 |
| Random warm-up | 5,000 steps |
| Discount $\gamma$ | 0.99 |
| Target-update coefficient $\tau$ | 0.005 |
| Adam learning rate | $3\times10^{-4}$ for actor, critics, and temperature |
| Target entropy | $-\dim(a)$ |

The critic target is

$$
y = r + \gamma(1-d)\left[\min_i Q_i'(s', a') - \alpha\log\pi(a' \mid s')\right], \qquad a' \sim \pi(\cdot \mid s')
$$

Critics minimize MSE to $y$; the actor minimizes $\mathbb{E}[\alpha\log\pi(a \mid s) - \min_i Q_i(s,a)]$; and `log_alpha` is tuned toward the target entropy. Target critics are Polyak-averaged after each update. The actor is checkpointed periodically to `sac_triple_latest.pt` and once at the end to `sac_triple_final.pt`.

### Networks

| Component | Architecture | Activation / output |
|---|---|---|
| PPO actor | 12 → 256 → 256 → 256 → 1 | Tanh hidden; Gaussian mean head |
| PPO critic | 12 → 256 → 256 → 256 → 1 | Tanh hidden; scalar $V(s)$ |
| PPO exploration | One learned log-std per action dim, state-independent | Normal policy, no `tanh` squash during training |
| SAC actor trunk | 12 → 256 → 256 | ReLU hidden; separate mean and log-std heads |
| SAC actor action | Reparameterized Normal sample | `tanh` squash with change-of-variables log-prob correction; log-std clamped to `[-5, 2]` |
| SAC critics (×2) | concat(state, action) → 256 → 256 → 1 | ReLU hidden; estimates $Q(s,a)$ |

PPO initializes linear layers orthogonally with a small gain (`0.01`) on the policy head. Parameter counts are 270,339 for PPO and 208,900 across the SAC actor and its two online critics, with target critics as separate copies.

## Results

Local W&B run directories are gitignored, so no reward curves or metric exports are committed. The table transcribes the on-disk `wandb-summary.json` values, keeping finished and interrupted runs distinct.

| Run | Configuration | Steps reached | Episodic return | Episode length | Throughput / runtime |
|---|---|---:|---:|---:|---|
| PPO `dwgtfxla` | 16 envs, 10M requested | 9,994,240 | 9,985.48 | 999 | 1,929 avg SPS; 5,182 s |
| PPO `wkj8m2oa` | same configuration | 9,994,240 | 9,985.48 | 999 | 1,802 avg SPS; 5,545 s |
| SAC `jppyfctk` (interrupted) | earlier single-env loop, 1M requested | 293,911 | 9,991.86 | 1,000 | 79 SPS; 3,732 s |
| SAC `zzierbr6` | 16 envs, one update/vector step, 1M requested | 999,872 | 9,983.85 | 999 | 817 windowed SPS; 1,136 s |

PPO ran 1,220 rollout updates and stopped at 9,994,240 rather than 10,000,000 because the loop takes an integer number of 8,192-transition batches.

Rolling the saved checkpoints out deterministically confirms the training numbers: at the training reset noise of `0.05`, both PPO and SAC balance for the full 1,000 steps on every seed tried, returning ~9,999.

### The one real ablation

SAC's data-collection schedule was redesigned for throughput, documented in [docs/SAC_SPS_SPEEDUP.md](docs/SAC_SPS_SPEEDUP.md):

| Earlier loop | Current loop |
|---|---|
| One environment, one full gradient update per transition | 16 environments, one full gradient update per vector step |
| Update-to-data ratio about 1:1 | One update per 16 collected transitions |
| 79 SPS at the last logged point | 817 windowed SPS at the last logged point |

Amortizing an expensive SAC update over 16 simulator transitions is what buys the wall-clock improvement. This is a scheduling change, not a test of learning quality — it moves environment parallelism and the update-to-data ratio at the same time, and the earlier run was interrupted by hand.

## What These Numbers Do and Don't Show

The runs above are training-time episode statistics from a single seed, not benchmark scores. The registered spec sets `reward_threshold=9100.0`, but neither trainer uses it as a stopping condition or computes a success rate, and neither performs a separate deterministic evaluation sweep.

In particular:

- **This is not a PPO-versus-SAC comparison.** The two share several hyperparameters but differ in budget (10M vs 1M steps), network activation, action parameterization, replay usage, and update-to-data ratio. A fair comparison needs matched evaluation seeds, repeated runs, and a declared budget criterion.
- **This is not a Tanh-versus-ReLU result.** PPO uses Tanh hidden layers and SAC uses ReLU, but that difference is confounded with everything else above. SAC's `tanh` *action squashing* is a bounded-action transform and a separate thing again. No controlled activation experiment exists in the repo, so the README attributes nothing to it.
- **No hyperparameter or architecture sweep was run.** The defaults are exposed as dataclass fields, but that is configurability, not evidence.
- **One seed, no uncertainty estimates.** Nothing here supports confidence intervals, learning-curve shape, or cross-seed variance.

Worth retaining for a stronger result table: a versioned experiment manifest, raw metric exports, hardware and timing, and a fixed evaluation seed set — then repeating runs across seeds and isolating one variable at a time.

## Known Rough Edges

- **Checkpoints land in the working directory.** Both trainers save to bare relative filenames, so a fresh run writes `.pt` files wherever it was launched from, not into `checkpoints/`. That directory holds artifacts moved there from previous runs.
- **The viewer uses a harder reset distribution than training.** `viz.py` builds its environment with the default `reset_noise_scale` of `0.1` while the trainers use `0.05`. The saved policies balance indefinitely at `0.05` and `0.0`, but fall within about 25 steps on roughly three of five seeds at `0.1`.
- **No command-line configuration.** Editing the dataclass in each trainer is the only way to change a run. The optional SB3 path is the only script with an evaluation callback.

## Further Reading

- [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) — implementation walkthrough of the environment, both algorithms, and the supporting scripts.
- [docs/SAC_SPS_SPEEDUP.md](docs/SAC_SPS_SPEEDUP.md) — the SAC throughput redesign, with profiling and the reasoning behind the vectorized loop.

The optional Stable-Baselines3 baseline has its own dependency extra:

```bash
pip install -e ".[train]"
python examples/train_sb3.py --timesteps 200000 --out models/ppo_triple
```
