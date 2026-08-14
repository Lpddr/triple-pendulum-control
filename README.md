# Inverted Triple Pendulum Reinforcement Learning

A custom continuous-control benchmark built with Gymnasium and MuJoCo. The task is to balance three serially connected pendulum links above a cart by applying one bounded horizontal control signal. The repository contains the environment, from-scratch PyTorch implementations of Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC), local experiment logs, checkpoints, visualization tools, and environment/physics tests.

## Overview

`InvertedTriplePendulum-v0` extends the cart-and-pole structure of Gymnasium's inverted double pendulum with a third 0.6 m link. The additional unactuated hinge increases the state dimension and makes the upright equilibrium more difficult to maintain: one cart force must control a coupled, unstable, nonlinear system with three angular degrees of freedom.

The project has two independent actor-critic training paths:

| Path | Implementation | Learning regime |
|---|---|---|
| PPO | `model.py` + `train.py` | On-policy rollouts, GAE, clipped updates |
| SAC | `sac_model.py` + `sac_train.py` | Off-policy replay, twin Q-functions, entropy-temperature tuning |

Both paths train against the same registered MuJoCo environment. `viz.py` can load either checkpoint type and run its deterministic policy in the MuJoCo viewer.

## Motivation

The aim is to study the practical engineering of continuous-control reinforcement learning on a custom physics task rather than only invoke a prebuilt benchmark. The project therefore includes:

- an MJCF model and a Gymnasium-compatible environment wrapper;
- an observation design that represents angles with sine/cosine pairs;
- reward shaping around upright height, lateral displacement, and angular velocity;
- separate on-policy and off-policy actor-critic implementations;
- vectorized data collection, correct handling of time-limit transitions, checkpoints, W&B logging, and physics/safety tests.

The task is deliberately underactuated: the cart is the only actuator, while all three pole hinges are passive.

## System Architecture

```text
MJCF model ──> InvertedTriplePendulumEnv ──> Gymnasium vector environments
                    │                                  │
                    │                                  ├── PPO rollout buffer → GAE → ActorCritic update
                    │                                  │
                    │                                  └── SAC replay buffer → twin-Q / actor / α updates
                    │
                    └── reward, termination, diagnostics → W&B / checkpoints / viewer
```

The XML model supplies rigid-body dynamics, joints, actuator limits, gravity, and rendering. `triple_pendulum/env.py` converts simulator state into observations, reward, termination, and diagnostic information. The training scripts own algorithm-specific buffering and updates; the environment itself does not contain learning logic.

## Environment

The package registers the following Gymnasium environment when `triple_pendulum` is imported:

```python
import gymnasium as gym
import triple_pendulum  # registers InvertedTriplePendulum-v0

env = gym.make("InvertedTriplePendulum-v0")
```

`max_episode_steps=1000` is registered with Gymnasium, so `gym.make` applies a `TimeLimit` wrapper. The raw environment returns `truncated=False`; the wrapper is responsible for time-limit truncation.

### Triple Pendulum

The MuJoCo model in `triple_pendulum/assets/inverted_triple_pendulum.xml` contains:

| Component | Implementation |
|---|---|
| Cart | One slider joint along the x-axis, limited to `[-1, 1]` m |
| Links | Three 0.6 m capsule links with hinge joints about the y-axis |
| State coordinates | Cart position plus three hinge angles (`nq=4`, `nv=4`) |
| Actuation | One control-limited motor on the slider; control range `[-1, 1]`, gear `500` |
| Physics | Gravity `(0, 0, -9.81)`, RK4 integration, MuJoCo timestep `0.01` s |
| Control rate | Python environment `frame_skip=5` by default, hence a control interval of about `0.05` s |
| Task geometry | Fully upright tip height is 1.8 m; a tip site at the end of link three drives reward and termination |

The scene also contains lighting, a skybox, decorative objects, and non-colliding visual geometry. These are rendering elements; the task dynamics come from the cart, three links, joints, and actuator.

### Observation Space

The observation is `Box(-inf, inf, (12,), float64)`. It is constructed directly from MuJoCo state and is not standardized by a running mean/variance normalizer. Training scripts cast observations to `float32` before passing them to PyTorch.

| Indices | Quantity | Treatment |
|---:|---|---|
| 0 | Cart position \(x\) | Raw position |
| 1–3 | \(\sin(\theta_1), \sin(\theta_2), \sin(\theta_3)\) | Angle representation |
| 4–6 | \(\cos(\theta_1), \cos(\theta_2), \cos(\theta_3)\) | Angle representation |
| 7–10 | Cart velocity and the three hinge velocities | Each clipped to \([-10,10]\) |
| 11 | x component of the cart constraint force | Clipped to \([-10,10]\) |

Using sine/cosine features avoids a discontinuity at angular wraparound. Reset adds uniform position noise and Gaussian velocity noise, both scaled by `reset_noise_scale` (default `0.1`; both custom trainers set it to `0.05`).

### Action Space

The action is `Box(-1, 1, (1,), float32)`: a scalar control for the cart motor. The MJCF actuator has the same control range and a gear value of 500.

SAC produces actions with a `tanh`-squashed Gaussian, so they are naturally bounded. PPO samples from an unsquashed Gaussian policy; those samples are passed to the MuJoCo actuator, whose control range bounds the applied control. The PPO viewer path additionally clips its deterministic mean action to `[-1, 1]`.

### Reward Function

Let \(x_{tip}\) be the tip's lateral coordinate, \(h_{tip}\) its vertical coordinate (reported in `info` as `tip_y`), and \(\omega_i\) the three hinge angular velocities. The environment computes

\[
r = r_{alive}
    - 0.01x_{tip}^{2}
    - (h_{tip}-1.8)^{2}
    - 10^{-3}\omega_1^2
    - 5\mathbin{\cdot}10^{-3}\omega_2^2
    - 10^{-2}\omega_3^2.
\]

`r_alive` is `healthy_reward` (default `10.0`) while the tip is healthy and zero on a terminating step. The vertical target is the physically reachable upright height of the three links. The top-link velocity has the largest velocity penalty.

Each step's `info` dictionary exposes `reward_survive`, `distance_penalty`, `velocity_penalty`, `tip_x`, and `tip_y`, which makes the reward components inspectable without reconstructing them from observations.

### Termination Conditions

- `terminated=True` when `tip_y <= termination_height` (default `1.5`). This represents a fallen pendulum, and the alive bonus is removed on that transition.
- `truncated=True` when the Gymnasium `TimeLimit` reaches 1,000 steps. This is an artificial episode boundary, not a physical failure.

This distinction is carried into learning code. PPO bootstraps the value of a time-limited final state into its rollout reward. SAC stores only physical terminations in its replay-buffer `done` field, so its TD target can bootstrap across time-limit truncations. With same-step vector auto-reset, both trainers recover `infos["final_obs"]` rather than accidentally treating the next episode's reset observation as the preceding transition's successor.

## Reinforcement Learning Algorithms

### PPO

The PPO implementation uses an `ActorCritic` with separate actor and value MLPs. It collects synchronous rollouts from 16 environments, then performs generalized advantage estimation (GAE) and multiple epochs of clipped policy/value updates.

Default PPO configuration in `train.py`:

| Setting | Value |
|---|---:|
| Total requested environment steps | 10,000,000 |
| Environments | 16 `SyncVectorEnv` instances |
| Rollout horizon | 512 steps/environment |
| Batch size | \(16 \times 512 = 8,192\) transitions/update |
| Update epochs / minibatches | 10 / 32 (minibatch size 256) |
| Discount \(\gamma\) / GAE \(\lambda\) | 0.99 / 0.95 |
| PPO clip coefficient | 0.2 |
| Value / entropy coefficients | 0.5 / 0.0 |
| Adam learning rate | \(3\times10^{-4}\), linearly annealed |
| Gradient-norm limit | 0.5 |
| Approximate-KL stop threshold | 0.02 |

For a rollout transition, the GAE recursion is

\[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad
\hat A_t = \delta_t + \gamma\lambda\hat A_{t+1},
\]

with episode boundaries and time-limit bootstrapping handled as described above. The policy loss uses the usual clipped ratio \(\min(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)\); the value loss is also clipped around the old value estimate.

Episode return and length, policy/value/entropy losses, approximate KL, clip fraction, learning rate, and steps per second (SPS) are logged to W&B. Checkpoints are saved every 50 PPO updates as `ppo_InvertedTriplePendulum-v0_<step>.pt`.

### SAC

SAC is implemented separately rather than as a modification of PPO. It maintains an actor, two critics \(Q_1,Q_2\), and matching target critics. After random warm-up, each vector-environment iteration inserts a batch of transitions into a uniform replay buffer and performs one gradient update.

Default SAC configuration in `sac_train.py`:

| Setting | Value |
|---|---:|
| Total requested environment steps | 1,000,000 |
| Environments | 16 `SyncVectorEnv` instances |
| Replay capacity / sample batch | 1,000,000 / 256 |
| Random warm-up | 5,000 environment steps |
| Discount \(\gamma\) | 0.99 |
| Target-update coefficient \(\tau\) | 0.005 |
| Adam learning rate | \(3\times10^{-4}\) for actor, critics, and temperature |
| Target entropy | \(-\text{action dimension}\) |

The critic target is

\[
y = r + \gamma(1-d)\left[\min_i Q_i'(s',a') - \alpha\log\pi(a'\mid s')\right],
\qquad a'\sim\pi(\cdot\mid s').
\]

The critics minimize MSE to \(y\). The actor minimizes

\[
\mathbb{E}\left[\alpha\log\pi(a\mid s) - \min_i Q_i(s,a)\right],
\]

and `log_alpha` is optimized to match the target entropy. Target critics are Polyak-averaged after each update. SAC periodically writes actor-only checkpoints to `sac_triple_latest.pt` and writes `sac_triple_final.pt` at the end of the loop.

## Network Architecture

| Component | Architecture | Activation / output behavior |
|---|---|---|
| PPO actor | 12 → 256 → 256 → 256 → 1 | Tanh hidden layers; Gaussian mean head |
| PPO critic | 12 → 256 → 256 → 256 → 1 | Tanh hidden layers; scalar \(V(s)\) |
| PPO exploration | One learned, state-independent log standard deviation per action dimension | Normal policy; no `tanh` squash in training |
| SAC actor trunk | 12 → 256 → 256 | ReLU hidden layers; separate mean and log-std heads |
| SAC actor action | Reparameterized Normal sample | `tanh` squash plus change-of-variables log-probability correction; log std clamped to `[-5, 2]` |
| SAC critics (two) | concat(state, action) → 256 → 256 → 1 | ReLU hidden layers; estimates \(Q(s,a)\) |

PPO initializes linear layers orthogonally, using a small gain (`0.01`) for the policy head. The custom training logs report 270,339 PPO parameters and 208,900 parameters across the SAC actor plus its two online critics; SAC target critics are separate copies.

## Training

The custom trainers choose CUDA when available and otherwise use CPU. They initialize W&B under the `triple-pendulum-RL` project, seed NumPy and PyTorch with `1`, and use a `reset_noise_scale` of `0.05`.

PPO alternates between rollout collection and batched updates. SAC differs in both data source and update schedule: replay transitions are reused, but the current performance-oriented loop executes one gradient update per **16-environment vector step**, not one update per individual transition.

For qualitative evaluation, `viz.py` detects the checkpoint format from state-dict keys, loads either a PPO `ActorCritic` or a SAC `Actor`, and runs the deterministic policy in a human-rendered environment. PPO uses its policy mean; SAC uses `tanh(mean)`. There is no separate custom fixed-seed evaluation harness or explicit success-rate calculation in the from-scratch training scripts. The optional SB3 script does create an `EvalCallback` evaluation environment.

## Experimental Methodology

The experiment code and local W&B data support these measurements:

- episodic return and episodic length from `RecordEpisodeStatistics`;
- PPO policy loss, value loss, entropy, approximate KL, clip fraction, learning rate, and average-from-start SPS;
- SAC critic loss, actor loss, entropy temperature \(\alpha\), and windowed SPS over roughly the last 5,000 environment steps;
- checkpoint artifacts for PPO and SAC actors;
- environment correctness checks covering topology, geometry, gravity, actuation direction, reward/termination consistency, time limits, finite rollouts, clipping, and seeded repeatability.

The registered Gymnasium specification sets `reward_threshold=9100.0`, but the training loops do not use it as a stopping condition and do not calculate a success rate. Nor do the custom trainers perform a separately logged deterministic evaluation sweep. Consequently, reported returns below are training-time episode statistics, not a multi-seed benchmark score.

## Experiments

### PPO vs SAC

Both algorithms were trained on the same environment ID with the same discount (`0.99`), hidden width (256), learning rate (`3e-4`), seed (`1`), and training reset noise (`0.05`). They are not otherwise matched: PPO uses a 10M-step on-policy schedule and Tanh networks, whereas SAC uses a 1M-step off-policy schedule, replay, ReLU networks, and entropy tuning.

The repository therefore demonstrates two implementations on the same task, but it does **not** contain a controlled PPO-versus-SAC sample-efficiency or final-performance study. A fair comparison would require matched evaluation seeds, repeated runs, an identical evaluation protocol, and a declared budget criterion.

### Tanh vs ReLU

The current implementations use different hidden activations: PPO uses Tanh and SAC uses ReLU. SAC additionally uses `tanh` **action squashing**, which is a bounded-action transform and should not be conflated with a hidden-layer activation ablation.

No saved configuration, experiment script, W&B run, result table, or Git history entry in this repository records a controlled Tanh-versus-ReLU experiment in which activation is the only changed variable. For that reason, this README does not attribute any performance difference to the activation function.

### Ablation Studies

One implementation-level throughput ablation is documented in `SAC_SPS_SPEEDUP.md` and represented in the local SAC W&B runs:

| Earlier SAC loop | Current SAC loop |
|---|---|
| One environment, one full gradient update per environment transition | 16 synchronous environments, one full gradient update per vector step |
| Update-to-data ratio about 1:1 | Lower update-to-data ratio: one update per 16 collected transitions |
| Local run summary: 79 SPS at the last logged point | Local run summary: 817 windowed SPS at the last logged point |

This is a throughput/scheduling change, not a controlled test of learning quality. It simultaneously changes environment parallelism and the update-to-data ratio; the earlier run was manually interrupted, and neither run provides multiple seeds or a common evaluation suite.

### Other Experiments

- `examples/train_sb3.py` provides an optional Stable-Baselines3 PPO baseline path with four environments by default, 2,048 rollout steps, a 256-sample batch, and `EvalCallback`. No SB3 training output is stored in the repository.
- The tracked checkpoints show that PPO and SAC training artifacts were retained. The two PPO checkpoint files are byte-identical; the two SAC checkpoint files are distinct actor state dicts.
- No persisted network-width/depth, reward-weight, reset-noise, discount, optimizer, or other hyperparameter sweep was found. The defaults are exposed as dataclass fields in the two custom training scripts, but configuration availability is not evidence of an executed ablation.

## Results

There are no versioned PNG/CSV reward curves or committed W&B exports. Local W&B run directories are ignored by `.gitignore`; the table below transcribes their on-disk `wandb-summary.json` values and run configuration, preserving the distinction between finished and interrupted evidence.

| Run / status | Configuration | Last recorded progress | Last logged episodic return | Last logged episodic length | SPS / runtime |
|---|---|---:|---:|---:|---|
| PPO `dwgtfxla` | 16 envs, requested 10M steps | 9,994,240 | 9,985.48 | 999 | 1,929 average SPS; 5,182.68 s |
| PPO `wkj8m2oa` | Same logged configuration | 9,994,240 | 9,985.48 | 999 | 1,802 average SPS; 5,545.73 s |
| SAC `jppyfctk` — interrupted | Earlier single-environment loop, requested 1M steps | 293,911 | 9,991.86 | 1,000 | 79 SPS; 3,732.77 s |
| SAC `zzierbr6` | 16 envs, one update/vector step, requested 1M steps | 999,872 | 9,983.85 | 999 | 817 windowed SPS; 1,136.25 s |

The PPO terminal logs show 1,220 rollout updates, ending at 9,994,240 rather than exactly 10,000,000 steps because the implementation computes an integer number of 8,192-transition batches. The vectorized SAC summary's last W&B metric is at 999,872 steps; its local output continues to 996,592 for windowed SPS reporting, and `sac_triple_final.pt` is present. These records are sufficient to document logged training behavior, but not to infer confidence intervals, learning-curve shape, or cross-seed variance.

The repository also includes two local, ignored render previews under `examples/`. They illustrate the MuJoCo scene, not policy performance, and no training video or GIF is stored.

## Analysis

**Observed in the local records.** PPO and the vectorized SAC run both logged episodic returns near 9,984–9,985 at the end of their records and episode lengths near the 1,000-step time limit. The current SAC scheduling recorded 817 SPS at its last W&B summary, compared with 79 SPS in the earlier interrupted SAC run. The `SAC_SPS_SPEEDUP.md` note reports a broader post-warm-up range of roughly 700–1,100 SPS on the same CPU setup.

**Interpretation, with limits.** The high end-of-run training returns and near-limit episode lengths are consistent with agents learning to sustain the upright configuration under the logged conditions. The throughput change plausibly improves wall-clock progress because a costly SAC update is amortized over 16 simulator transitions. It does not establish that SAC is intrinsically faster, more sample-efficient, or more stable than PPO: their logging windows differ, their update schedules differ, the budgets differ, and no repeated deterministic evaluation was retained. Similarly, the activation difference is confounded with algorithm, policy parameterization, and replay usage.

## Reproducibility

The commands below reproduce the repository's existing entry points; they do not alter environment, algorithm, or experiment settings. A MuJoCo-capable display is required for `render_mode="human"` on macOS. The custom trainers import PyTorch, W&B, and GLFW in addition to the project package dependencies.

```bash
git clone <repository-url>
cd triple-pendulum
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install torch wandb glfw

# Environment smoke/physics suite
pytest -q

# Random-agent render or one RGB frame
python examples/demo_random.py --episodes 3
python examples/demo_random.py --render-mode rgb_array --save-frame frame.png
```

Run the from-scratch trainers with their checked-in defaults:

```bash
python train.py       # PPO; requests 10,000,000 steps
python sac_train.py   # SAC; requests 1,000,000 steps
```

Both commands initialize W&B logging. Configure W&B before a tracked run, or use its supported offline/disabled mode if only local execution is desired. The run configuration is defined directly in `Config` (`train.py`) and `cfg` (`sac_train.py`); neither script exposes a command-line hyperparameter interface.

Watch a stored policy:

```bash
python viz.py --checkpoint ppo_triple_final.pt
python viz.py --checkpoint sac_triple_final.pt
```

The optional Stable-Baselines3 baseline has a separate dependency extra:

```bash
pip install -e ".[train]"
python examples/train_sb3.py --timesteps 200000 --out models/ppo_triple
```

For a result table comparable to the local evidence above, retain the W&B configuration, raw history, hardware/device, wall-clock timing, checkpoints, and fixed evaluation seeds. The existing summaries alone do not preserve all of that information in version control.

## Repository Structure

```text
triple-pendulum/
├── triple_pendulum/
│   ├── __init__.py                         # Gymnasium registration
│   ├── env.py                              # observation, reward, reset, termination
│   └── assets/inverted_triple_pendulum.xml # MuJoCo cart + three-link model
├── model.py                                # PPO ActorCritic
├── train.py                                # PPO collection, GAE, clipped updates
├── sac_model.py                            # SAC squashed actor and Q critic
├── sac_train.py                            # SAC replay and target-network training loop
├── viz.py                                  # deterministic PPO/SAC checkpoint viewer
├── examples/
│   ├── demo_random.py                      # render/sanity-check utility
│   └── train_sb3.py                        # optional SB3 PPO path
├── tests/
│   ├── test_env.py                         # API, spaces, render, observation smoke tests
│   └── test_physics_and_safety.py          # dynamics, reward, time-limit, safety checks
├── ppo_*.pt / sac_*.pt                     # retained PyTorch checkpoint artifacts
├── HOW_IT_WORKS.md                         # detailed implementation walkthrough
├── SAC_SPS_SPEEDUP.md                      # SAC scheduling/throughput note
├── pyproject.toml                          # package metadata and extras
└── requirements.txt                        # base/dev dependency list
```

## Key Findings

- The project implements a 12-dimensional custom MuJoCo triple-pendulum observation with one bounded cart-control action and a reward aligned to upright height, lateral centering, and low angular velocity.
- PPO and SAC are implemented as separate actor-critic systems with the expected algorithm-specific machinery: GAE and clipped updates for PPO; replay, twin Q targets, reparameterized actions, and learned entropy temperature for SAC.
- The training loops explicitly distinguish physical termination from time-limit truncation and handle vector-environment final observations, which is important for correct value/TD bootstrapping.
- Local W&B summaries show high late-training episodic returns and near-time-limit episode lengths for PPO and the vectorized SAC run.
- The only evidence-backed ablation is SAC's data-collection/update scheduling change. It improves the logged local SPS substantially, but it is not a clean performance or sample-efficiency comparison.

## Limitations

- Local W&B artifacts are ignored rather than versioned, and no static reward curves, evaluation CSVs, or videos are committed.
- The retained records use one logged seed and do not provide uncertainty estimates or independent evaluation rollouts.
- PPO and SAC have different budgets, architectures, action parameterizations, and update-to-data ratios, so their logged returns should not be read as a controlled algorithm ranking.
- There is no persisted Tanh-versus-ReLU-only experiment, hyperparameter sweep, or architecture sweep.
- The custom trainers have no command-line configuration interface and no automatic deterministic evaluation protocol; the optional SB3 path is the only script with an explicit evaluation callback.

## Future Work

Useful next investigations would be to retain a versioned experiment manifest and raw metric exports, evaluate every checkpoint on a fixed set of seeds, repeat runs across random seeds, and isolate one variable at a time for activation, architecture, reward, reset-noise, and update-to-data ablations. Those additions would make it possible to distinguish final control quality, sample efficiency, stability, and wall-clock throughput without conflating them.
