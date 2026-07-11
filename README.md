# Inverted Triple Pendulum (Gymnasium + MuJoCo)

A custom **inverted triple pendulum** environment built as a near drop-in extension of Gymnasium’s [`InvertedDoublePendulum-v5`](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/): same cart force control, same reward shape, **one extra pole** stacked on the free end.

Visuals use a deep-space skybox, neon cyan rail, metallic cart, and a cyan → violet → magenta pole gradient with a gold tip jewel.

![concept](https://gymnasium.farama.org/_images/inverted_double_pendulum.gif)

## Quick start

```bash
cd Code/triple-pendulum
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Live random agent (opens MuJoCo viewer)
python examples/demo_random.py --episodes 3

# Headless RGB frame
python examples/demo_random.py --render-mode rgb_array --save-frame frame.png

# Tests
pytest -q
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
- Target tip height \(y^\*\): default **1.8**, the physically reachable upright tip height
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

## Training (optional)

```bash
pip install 'stable-baselines3[extra]'
python examples/train_sb3.py --timesteps 200000 --out models/ppo_triple
```

This is a **harder** control problem than the double pendulum. Tips:

1. Start with `reset_noise_scale=0.05`  
2. Curriculum: train double first, then transfer / fine-tune  
3. Use domain randomization on gravity / pole mass later if you want robustness  

## Project layout

```
triple-pendulum/
  triple_pendulum/
    __init__.py          # gym registration
    env.py               # InvertedTriplePendulumEnv
    assets/
      inverted_triple_pendulum.xml
  examples/
    demo_random.py
    train_sb3.py
  tests/
    test_env.py
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
- macOS: MuJoCo viewer needs a display for `render_mode="human"`
