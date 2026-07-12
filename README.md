# Inverted Triple Pendulum (Gymnasium + MuJoCo)

A custom **inverted triple pendulum** environment: Gymnasium’s [`InvertedDoublePendulum-v5`](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/) idea with **one extra pole** on the free end. Same cart-force control and reward family; harder balance problem.

This repo also has from-scratch **PPO** and **SAC** trainers (PyTorch + Weights & Biases), a MuJoCo viewer for checkpoints, demos, and tests.

Visuals: deep-space skybox, neon cyan rail, metallic cart, cyan → violet → magenta poles, gold tip.

![concept](https://gymnasium.farama.org/_images/inverted_double_pendulum.gif)

**Docs**

| File | What it is |
|------|------------|
| [HOW_IT_WORKS.md](HOW_IT_WORKS.md) | End-to-end education: env, physics, PPO, SAC, training loops |
| [SAC_SPS_SPEEDUP.md](SAC_SPS_SPEEDUP.md) | Why SAC was ~80 SPS and what we changed (~700–1100 SPS) |

## Quick start

```bash
cd triple-pendulum
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install torch wandb  # needed for from-scratch PPO / SAC

# Live random agent (opens MuJoCo viewer)
python examples/demo_random.py --episodes 3

# Headless RGB frame
python examples/demo_random.py --render-mode rgb_array --save-frame frame.png

# Tests
pytest -q
```

Optional Stable-Baselines3:

```bash
pip install -e ".[train]"
python examples/train_sb3.py --timesteps 200000 --out models/ppo_triple
```

## Usage

```python
import gymnasium as gym
import triple_pendulum  # registers InvertedTriplePendulum-v0

env = gym.make("InvertedTriplePendulum-v0", render_mode="human")
obs, info = env.reset(seed=0)

for _ in range(1000):
    action = env.action_space.sample()  # force on cart in [-1, 1]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

Or construct the class directly:

```python
from triple_pendulum import InvertedTriplePendulumEnv

env = InvertedTriplePendulumEnv(healthy_reward=10.0, reset_noise_scale=0.05)
```

## Spaces

| | |
|---|---|
| **Action** | `Box(-1, 1, (1,))` — continuous force on the cart |
| **Observation** | `Box(-inf, inf, (12,))` — cart pos, sin/cos of 3 angles, 4 velocities, cart constraint force |
| **Max episode steps** | 1000 (via `TimeLimit`) |

### Observation layout

| Index | Meaning |
|------:|---------|
| 0 | cart \(x\) |
| 1–3 | \(\sin\theta_1, \sin\theta_2, \sin\theta_3\) |
| 4–6 | \(\cos\theta_1, \cos\theta_2, \cos\theta_3\) |
| 7–10 | \(\dot x, \dot\theta_1, \dot\theta_2, \dot\theta_3\) (clipped ±10) |
| 11 | cart constraint force \(x\) (clipped ±10) |

Compared to double pendulum **(9-dim)** this is **(12-dim)** — one extra sin, cos, and angular velocity.

## Reward (same family as double)

\[
r = r_{\text{alive}} - 0.01\,x_{\text{tip}}^2 - (y_{\text{tip}} - y^\*)^2
    - 10^{-3}\omega_1^2 - 5\cdot 10^{-3}\omega_2^2 - 10^{-2}\omega_3^2
\]

- Alive bonus: `healthy_reward` (default **10**) while healthy  
- Target tip height \(y^\*\): default **1.8** (physically reachable upright tip)  
- Terminate when \(y_{\text{tip}} \le\) `termination_height` (default **1.5**)

`info` includes `reward_survive`, `distance_penalty`, `velocity_penalty`, `tip_x`, `tip_y`.

## Configurable kwargs

```python
gym.make(
    "InvertedTriplePendulum-v0",
    healthy_reward=10.0,
    reset_noise_scale=0.1,      # smaller (e.g. 0.05) is easier to learn
    termination_height=1.5,
    target_height=1.8,
    frame_skip=5,
)
```

## Training

Both custom trainers log to the W&B project `triple-pendulum-RL`. Use a smaller `reset_noise_scale` (0.05) while learning.

### PPO (from scratch) — `train.py` + `model.py`

On-policy: collect rollout → GAE → clipped policy/value updates. Default **10M** env steps, **16** vectorized envs. Typical SPS on this machine’s CPU setup: **~1400–2000**.

```bash
python train.py
# checkpoints: ppo_InvertedTriplePendulum-v0_<step>.pt
# opens MuJoCo viewer after training if watch_after_training=True
```

Watch a PPO checkpoint:

```bash
python viz.py                          # newest *.pt in project root
python viz.py --checkpoint ppo_triple_final.pt
```

`viz.py` expects an **ActorCritic** state dict from `train.py` (not SAC).

### SAC (from scratch) — `sac_train.py` + `sac_model.py`

Off-policy soft actor-critic: replay buffer, twin Qs, auto-tuned temperature \(\alpha\). Default **1M** env steps, **16** vectorized envs, **one gradient update per vector step**. Typical SPS after the speedup: **~700–1100** on CPU (was ~80 with 1 env + update every step). Details: [SAC_SPS_SPEEDUP.md](SAC_SPS_SPEEDUP.md).

```bash
python sac_train.py
# checkpoints: sac_triple_latest.pt (every 250k), sac_triple_final.pt
```

SAC saves **actor weights only**. Watching them needs a small loader for `sac_model.Actor` (not `viz.py` as-is).

### Stable-Baselines3 (optional)

```bash
pip install 'stable-baselines3[extra]'
python examples/train_sb3.py --timesteps 200000 --out models/ppo_triple
```

### Tips

1. Start with `reset_noise_scale=0.05`  
2. This is harder than double pendulum — expect longer training  
3. PPO is sample-hungry but simple; SAC reuses data via the replay buffer  
4. Device defaults to CUDA if available, otherwise CPU (no MPS fallback in the scripts)

## Project layout

```
triple-pendulum/
  triple_pendulum/
    __init__.py                 # gym registration → InvertedTriplePendulum-v0
    env.py                      # InvertedTriplePendulumEnv
    assets/
      inverted_triple_pendulum.xml
  model.py                      # PPO ActorCritic
  train.py                      # PPO training loop
  sac_model.py                  # SAC Actor + twin Q critics
  sac_train.py                  # SAC training loop (vectorized)
  viz.py                        # watch PPO checkpoints
  examples/
    demo_random.py              # random agent / render demo
    train_sb3.py                # optional SB3 PPO
  tests/
    test_env.py                 # smoke tests
    test_physics_and_safety.py  # geometry, reward, TimeLimit, finiteness
  HOW_IT_WORKS.md               # education walkthrough
  SAC_SPS_SPEEDUP.md            # SAC throughput change writeup
  README.md
  pyproject.toml
  requirements.txt
```

## Design notes vs double pendulum

| Piece | Double (Gymnasium) | This env |
|-------|--------------------|----------|
| Poles | 2 × 0.6 m | **3 × 0.6 m** |
| Tip upright height | 1.2 m | **1.8 m** |
| Obs dim | 9 | **12** |
| Terminate tip \(y\) | ≤ 1.0 | ≤ **1.5** |
| Distance target \(y\) | 2.0 | **1.8** (upright tip height) |
| Vel penalty | \(\omega_1,\omega_2\) | \(+\omega_3\) weighted higher |
| Actuator | gear 500 on slider | same |

## Dependencies

- Python ≥ 3.10  
- `gymnasium[mujoco]`, `mujoco`, `numpy`  
- Training scripts: `torch`, `wandb` (and `glfw` for live viewers)  
- Optional: `stable-baselines3[extra]`, `pytest`, `imageio`  
- macOS: MuJoCo viewer needs a display for `render_mode="human"`
