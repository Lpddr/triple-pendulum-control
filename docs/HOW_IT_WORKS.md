# How this project works

A plain-language tour of every important piece: the physics env, the two RL algorithms (PPO and SAC), and how the training scripts glue them together. Read top-to-bottom if you are new; jump sections if you already know RL and only care about this codebase.

---

## 1. The problem

**Goal:** keep a cart-and-three-poles system balanced upright by applying a horizontal force to the cart.

```
        tip (gold)
          ●
          |
       pole3 (magenta)
          |
       pole2 (violet)
          |
       pole1 (cyan)
          |
       [cart]  ←── force in [-1, 1]
   ======= rail =======
```

Fully upright, the tip sits at **1.8 m** (three 0.6 m poles). The episode **ends early** if the tip drops to **1.5 m** or below (fallen). Episodes also **truncate** at 1000 steps even if still balanced — that is a time limit, not a “failure.”

Why it is hard:

- Three free hinges → highly nonlinear, chaotic when falling  
- One control input (cart force) for a high-DoF system  
- Small early mistakes compound quickly  

Compared to Gymnasium’s inverted **double** pendulum: same control idea, one extra pole, larger observation, tougher balance.

---

## 2. Environment package (`triple_pendulum/`)

### Registration — `__init__.py`

Importing `triple_pendulum` registers the Gymnasium id:

```text
InvertedTriplePendulum-v0  →  InvertedTriplePendulumEnv
max_episode_steps = 1000
```

So this works anywhere in the repo:

```python
import gymnasium as gym
import triple_pendulum
env = gym.make("InvertedTriplePendulum-v0")
```

### Physics asset — `assets/inverted_triple_pendulum.xml`

MuJoCo MJCF model:

| Name | Role |
|------|------|
| `slider` | cart slides on the rail (limited ±1 m) |
| `hinge`, `hinge2`, `hinge3` | three free hinges (one per pole) |
| cart motor | force actuator, gear **500** (same family as double) |
| tip site | world position of the free end (used for reward / termination) |

Also: gravity, RK4 integrator, timestep 0.01 s, decorative sky/stars that do **not** collide (they do not change dynamics).

`frame_skip` (default **5** in the Python env) means each `env.step` advances **5** MuJoCo substeps → control dt ≈ **0.05 s**.

### Gym wrapper — `env.py` (`InvertedTriplePendulumEnv`)

Subclass of Gymnasium’s `MujocoEnv`. Each `step(action)`:

1. Apply cart force and simulate `frame_skip` substeps  
2. Read tip position `(x, y)` from the tip site  
3. Build a **12-D observation**  
4. Compute **reward** and whether the tip is still healthy  
5. Return `(obs, reward, terminated, truncated, info)`  

**Observation (12 floats):**

| Index | Content |
|------:|---------|
| 0 | cart \(x\) |
| 1–3 | \(\sin\) of each pole angle |
| 4–6 | \(\cos\) of each pole angle |
| 7–10 | cart + hinge velocities (clipped ±10) |
| 11 | cart constraint force \(x\) (clipped ±10) |

Sin/cos instead of raw angles avoids wraparound discontinuities and matches the double-pendulum design.

**Reward:**

\[
r = r_{\text{alive}} - 0.01\,x_{\text{tip}}^2 - (y_{\text{tip}} - 1.8)^2
    - 10^{-3}\omega_1^2 - 5\cdot 10^{-3}\omega_2^2 - 10^{-2}\omega_3^2
\]

- Alive bonus (default 10) while tip is above the termination height  
- Distance penalty: tip should be high and not far sideways  
- Velocity penalty: especially on the **top** hinge (easiest to thrash)  

**Termination vs truncation (critical for RL):**

| Signal | Meaning | Bootstrap value / Q? |
|--------|---------|----------------------|
| `terminated=True` | tip fell — real end of episode | **No** — future return is 0 |
| `truncated=True` | hit 1000-step time limit, still up | **Yes** — episode was cut artificially |

PPO handles truncation by bootstrapping \(V(s')\) into the reward. SAC stores `done = terminated` only so the TD target still bootstraps through truncations.

**Reset noise:** on reset, small noise is added to positions/velocities (`reset_noise_scale`). **0.05** is easier than the default **0.1** — less chaotic starts while learning.

---

## 3. Two algorithms, one task

Both learn a policy \(\pi(a \mid s)\): “given observation, push the cart left/right.”

They disagree on **when** to learn and **from what data**.

| | PPO (`algorithms/ppo/train.py`) | SAC (`algorithms/sac/sac_train.py`) |
|--|------------------|----------------------|
| Style | On-policy | Off-policy |
| Data | Fresh rollouts only | Replay buffer (reuse old steps) |
| Critic | \(V(s)\) — value of a state | Twin \(Q(s,a)\) — value of a state-action |
| Exploration | Gaussian with **global** log-std | State-dependent std + entropy bonus |
| Action range | Clip mean/sample to [-1, 1] at env | **Tanh squashing** inside the actor |
| Typical steps | 10M (sample hungry) | 1M default (more sample efficient) |
| Throughput | High SPS: rare big updates | Update more often; SPS depends on schedule |

Neither is “wrong.” PPO is often easier to debug; SAC often needs fewer env steps for continuous control.

---

## 4. PPO stack

### Networks — `algorithms/ppo/model.py` (`ActorCritic`)

Two MLPs sharing the same input dim, **not** a shared trunk:

- **Actor:** obs → 3×256 Tanh → action mean  
- **Critic:** obs → 3×256 Tanh → scalar \(V(s)\)  
- **`actor_logstd`:** one learnable log-std **per action dim**, **not** a function of state  

Actions are sampled from \(\mathcal{N}(\mu(s), \sigma)\). Orthogonal init (policy head small gain) is CleanRL-style.

### Training loop — `algorithms/ppo/train.py`

Four phases, repeated:

```text
┌─────────────┐    ┌─────┐    ┌────────┐    ┌────────┐
│ 1. Collect  │ →  │ 2.  │ →  │ 3.     │ →  │ 4. log │
│  n_envs ×   │    │ GAE │    │ PPO    │    │ + save │
│  n_steps    │    │     │    │ update │    │        │
└─────────────┘    └─────┘    └────────┘    └────────┘
```

Defaults that matter:

- `n_envs=16`, `n_steps=512` → **8192** samples per update  
- 10 epochs over that batch, minibatches of 256  
- clip \(\epsilon=0.2\), GAE \(\lambda=0.95\), \(\gamma=0.99\)  
- early stop epoch if approx KL > 0.02  
- LR anneals linearly to 0 over training  

**Vector envs + SAME_STEP autoreset**  
When an env finishes, Gymnasium can auto-reset inside `step`. With `AutoresetMode.SAME_STEP`, the **true last observation** of the finished episode is in `infos["final_obs"]`, while the array `next_obs` is already the **new** episode’s start. PPO uses `final_obs` only for **truncation bootstrap** (add \(\gamma V(s_{\text{final}})\) into that step’s reward).

**GAE (Generalized Advantage Estimation)**  
After the rollout, work **backward** in time to estimate how much better each action was than the value baseline. Advantages are normalized per minibatch before the policy loss.

**Clipped surrogate**  
Ratio \(r_t = \pi_{\text{new}} / \pi_{\text{old}}\). Don’t trust huge policy changes — clamp the ratio. Value loss can be clipped too.

**Why SPS is high (~1400–2000 here)**  
Most wall time is **MuJoCo stepping**. Gradient updates happen only once per 8192 env steps (then 10 epochs of minibatches). Env steps are “cheap”; updates are “expensive but rare.”

### Watching a policy — `evaluation/viz.py` / `watch()` in `algorithms/ppo/train.py`

Loads a checkpoint, runs the deterministic action (no sampling) in `render_mode="human"`, resets when the pole falls or time limit hits. `viz.py` detects the checkpoint type from its state-dict keys and handles **both** a PPO `ActorCritic` (policy mean) and a SAC actor-only save (`tanh(mean)`); anything else is rejected. It picks the newest `*.pt` under `checkpoints/` unless you pass `--checkpoint`.

Note that `viz.py` builds the env with the **default** `reset_noise_scale` of 0.1, while both trainers use 0.05. The saved policies balance for the full 1000 steps at 0.05 but fall on some seeds at 0.1, so the viewer is a harder test than training was.

---

## 5. SAC stack

### Networks — `algorithms/sac/sac_model.py`

**Actor**

- Backbone → **mean head** and **log_std head** (both state-dependent)  
- Sample with **reparameterization** (`rsample`): \(x = \mu + \sigma \odot \varepsilon\), \(\varepsilon \sim \mathcal{N}(0,I)\)  
- Squash: \(a = \tanh(x)\) so actions live in \((-1,1)\)  
- Log-prob corrected for the tanh (change of variables)  

**Twin Q critics**

- Each Q is MLP on `concat(s, a)` → scalar  
- Two Qs reduce overestimation; training uses \(\min(Q_1, Q_2)\)  
- **Target networks** \(Q_1', Q_2'\) are slow-moving copies (Polyak average with \(\tau\))  

**Temperature \(\alpha\)**  
Auto-tuned via `log_alpha` so policy entropy tracks a target (here \(-\text{act_dim}\)). Higher \(\alpha\) → more exploration; lower → more exploitation.

### Training loop — `algorithms/sac/sac_train.py`

Off-policy cycle (every **vector** step after warmup):

```text
act in n_envs  →  store n_envs transitions  →  sample batch from buffer
        →  update Qs  →  update actor  →  update α  →  soft-update targets
```

**Replay buffer**  
Ring buffer of `(obs, act, rew, next_obs, done)`. `done` is **termination only**. Sampling is uniform random minibatches (size 256).

**Warmup (`start_steps=5000`)**  
Pure random actions so the buffer is not empty/biased before the first gradients.

**Critic target (sketch):**

\[
y = r + \gamma (1-d)\Big(\min_i Q_i'(s', a') - \alpha \log \pi(a'|s')\Big),\quad a'\sim\pi(\cdot|s')
\]

**Actor objective (sketch):** minimize \(\mathbb{E}[\alpha \log \pi(a|s) - \min_i Q_i(s,a)]\)  
i.e. high Q, but keep entropy (via \(\alpha\)).

**Why vectorize + one update per vector step**  
A full SAC update (several forwards/backwards of actor + two Qs + alpha + targets) is ~tens of ms on CPU. Doing that after **every single** env step → ~40–80 SPS. Stepping **16** envs then paying for **one** update → order-of-magnitude more env steps per second. See [SAC_SPS_SPEEDUP.md](SAC_SPS_SPEEDUP.md).

**Autoreset and `final_obs`**  
Same issue as PPO: after an episode ends, vector `next_obs` is the **reset** state. For the transition \((s,a,r,s')\) we must store \(s' =\) `final_obs` when autoreset fired, otherwise the buffer links the last action to the **next episode’s first state**.

---

## 6. Examples and tests

### `examples/demo_random.py`

No learning. Random actions; human window stays open until closed; optional `rgb_array` + PNG. Good sanity check that MuJoCo/rendering works.

### `examples/train_sb3.py`

Optional Stable-Baselines3 PPO. Same env id; different code path than `algorithms/ppo/train.py`. Needs `pip install stable-baselines3[extra]`.

### `tests/test_env.py`

Smoke: spaces, reset/step, sin²+cos²≈1, basic finiteness.

### `tests/test_physics_and_safety.py`

Deeper: joint topology, upright tip ~1.8 m, gear 500, termination/reward consistency, TimeLimit 1000, long-horizon finite values. Guards against “broken physics” or accidental reward exploits.

---

## 7. Config and dependencies (mental map)

| Piece | Role |
|-------|------|
| `pyproject.toml` | package metadata; install env with `pip install -e .` |
| `requirements.txt` | flat list for casual installs |
| `torch` / `wandb` | custom trainers (not in the core package deps) |
| Checkpoints `*.pt` | saved weights (PPO full agent vs SAC actor-only); retained runs live in `checkpoints/ppo` and `checkpoints/sac`, but a fresh run writes to the working directory it was launched from |
| `wandb/` | local W&B run files (gitignored) |

Device line in both trainers:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

On many Macs that means **CPU**. PPO still hits high SPS because it barely updates; SAC is more update-heavy even when vectorized.

---

## 8. Mental model: one env step vs one learning step

**Environment step**  
Observation in → action out → MuJoCo integrates physics → new obs, reward, flags. Cost: sub-millisecond to ~1 ms for this small model.

**Learning step (PPO)**  
Needs a whole rollout, GAE, then many optimizer steps. Expensive, but infrequent relative to env steps.

**Learning step (SAC)**  
Needs a minibatch from the buffer and several network updates. Expensive **and** traditionally run often (once per env step in the original single-env script). Throughput is almost entirely about **how often** you pay that cost relative to how many env steps you collect.

---

## 9. Suggested reading order in the code

1. `triple_pendulum/env.py` — what the agent sees and is paid for  
2. `algorithms/ppo/model.py` then `algorithms/ppo/train.py` — simplest full RL loop in this repo  
3. `algorithms/sac/sac_model.py` then `algorithms/sac/sac_train.py` — off-policy counterpart  
4. `evaluation/viz.py` / `examples/demo_random.py` — interaction without training  
5. `tests/test_physics_and_safety.py` — what “correct env” means here  

For the SAC throughput redesign specifically, read [SAC_SPS_SPEEDUP.md](SAC_SPS_SPEEDUP.md) next.
