# SAC SPS speedup — what changed and why

This note documents the performance redesign of `sac_train.py` (vectorized collection + one gradient update per vector step). It is the detailed writeup behind the short summary in the README.

**Status:** applied to `sac_train.py`  
**Symptom before:** ~**80 SPS** (env steps / second) on the same machine where PPO hit ~**1400–2000 SPS**  
**After smoke test on this machine:** ~**700–1100 SPS** (windowed, post-warmup)

---

## 1. What “SPS” means here

Both trainers log something like:

```text
charts/SPS = env_steps / wall_time
```

So SPS is **how fast we walk the environment**, not “gradient updates per second.”

That distinction matters:

- **PPO** spends most of the time collecting with 16 envs and only then runs a heavy update every 8192 steps → high SPS.  
- **Old SAC** paid for a full critic + actor + alpha + target update after **almost every single** MuJoCo step → low SPS even though the env itself is fast.

---

## 2. Diagnosis (profiled on this machine)

Device selection in the script:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

On this x86 Mac: **CUDA false, MPS unavailable** → training runs on **CPU**. That is expected; the bottleneck is still architectural, not “MuJoCo is slow.”

Rough timings for the **old** single-env loop shape:

| Phase | Time | Implied rate |
|-------|------|----------------|
| Env step only | ~0.4 ms | ~2300 SPS |
| Act (batch-1 network) + env | ~0.8 ms | ~1200 SPS |
| Replay sample | ~0.2 ms | negligible |
| **Full SAC update (batch 256)** | **~24 ms** | **~40 updates/s** |
| Act + env + update (together) | **~23 ms** | **~40–80 SPS** |

So:

- The triple pendulum env is **not** the problem.  
- The **gradient update every env step** is.  
- ~80 SPS was consistent with “update dominates; warmup may slightly inflate the long-run average early on.”

PPO’s 1400–2000 SPS is the same hardware doing **mostly collection**. Comparing raw SPS across algorithms without looking at update frequency is misleading — but it was a fair user expectation that SAC should not be 20× slower just because the file was structured that way.

---

## 3. What was wrong in the old `sac_train.py`

Conceptual loop (before):

```text
for each env step:
    act in 1 env
    store 1 transition
    if past warmup:
        sample batch
        update Q1, Q2
        update actor
        update alpha
        polyak-update targets
```

Problems for throughput:

1. **Update:env ratio = 1:1**  
   One ~24 ms CPU update per ~1 ms of interaction → SPS ≈ 1 / 0.025 ≈ 40–80.

2. **Single environment**  
   No parallelism in simulation. Even with a better update schedule, you leave free env throughput on the table (PPO already used `n_envs=16`).

3. **SPS metric was a run-average from t=0**  
   Random warmup (no updates) pulled the average up early, then the number decayed as updates kicked in — harder to read than a short window.

4. **Secondary (small) costs**  
   Per-step `torch.tensor(...)` for batch-1 act, and `torch.tensor` copies in buffer sample. Real, but dwarfed by the update.

The **learning math** (twin Qs, entropy target, soft targets, `done = terminated` only) was already in good shape. This change was about **how often** and **in what batching structure** we pay for that math relative to env steps.

---

## 4. What we changed

### 4.1 Vectorized environments (`n_envs=16`)

Mirror PPO’s collection pattern:

```python
envs = gym.vector.SyncVectorEnv(
    [make_env() for _ in range(config.n_envs)],
    autoreset_mode=AutoresetMode.SAME_STEP,
)
envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
```

Each loop iteration:

- Steps **16** envs  
- Advances `global_step` by **16**  
- Stores **16** transitions  

`RecordEpisodeStatistics` gives the same episode return/length logging pattern as `train.py`.

### 4.2 One gradient update per **vector** step (not per env)

After warmup:

```text
vector step (16 env transitions)  →  exactly one SAC update
```

Previously:

```text
1 env transition  →  one SAC update
```

So the number of updates per env step dropped by about **`n_envs` (16×)** while env throughput went up.

**Why not “n_envs updates per vector step”?**  
That would keep the same update:data ratio as the old 1:1 loop and would **not** fix SPS — you would still pay ~16 × 24 ms of updates for 16 env steps.

**Learning tradeoff:**  
Update-to-data ratio (UTD) is lower than classic “1 update per transition” SAC. For this low-dim continuous task that is usually fine; you trade a bit of sample efficiency for large wall-clock gains. If learning looks under-updated later, knobs are:

- lower `n_envs`, or  
- run `gradient_steps > 1` occasionally (not added in this change — keep the loop simple).

### 4.3 Batch buffer inserts (`ReplayBuffer.add_batch`)

Vector steps produce shape `(n_envs, …)`. Instead of calling single-transition `add` 16 times, one `add_batch` writes a contiguous slice (and handles ring-buffer wrap by splitting).

Same semantics as 16 singles; less Python overhead.

### 4.4 Correct `next_obs` under autoreset (`final_obs`)

With `AutoresetMode.SAME_STEP`, when an episode ends inside `envs.step`:

- the returned `next_obs[i]` is already the **reset** observation of the **new** episode  
- the true terminal/truncated observation is in `infos["final_obs"][i]`

The buffer must store the **true** \(s'\) for the transition that just happened:

```python
real_next = next_obs.copy()
if "final_obs" in infos:
    for i in range(n_envs):
        if infos["_final_obs"][i]:
            real_next[i] = infos["final_obs"][i]
buffer.add_batch(obs, action, reward, real_next, terminations)
obs = next_obs  # continue from autoreset states
```

Still: **`done` flag = terminated only** (not truncated), so TD still bootstraps through time-limit cuts. That matches the old single-env intent and is the SAC analogue of PPO’s truncation bootstrap.

### 4.5 Windowed SPS logging

Before: `step / (time.time() - start)` from the beginning of the run.  
After: SPS over the **last ~5000 env steps** only.

That matches how you experience speed mid-run and avoids “warmup looked fast, then mysteriously slowed down.”

### 4.6 Unchanged on purpose

- Network architecture (`sac_model.py`)  
- Losses (Q MSE, actor entropy-augmented objective, auto \(\alpha\))  
- Hyperparameters: `batch_size=256`, `tau=0.005`, `gamma=0.99`, `lr=3e-4`, `hidden_dim=256`, `start_steps=5000`, `total_timesteps=1_000_000`  
- Checkpoint names: `sac_triple_latest.pt` / `sac_triple_final.pt` (actor weights only)  
- W&B project name  

Also **not** done in this change (possible later):

- `update_every` / multi gradient steps  
- `hidden_dim=128`  
- MPS device selection  
- `viz.py` support for SAC actors  

---

## 5. Why this gets closer to PPO’s SPS (but not all the way)

Rough cost model after the change (CPU, order of magnitude):

```text
per vector iteration:
    16 × env (+ batched act)  ≈  few ms
    1 × SAC update            ≈  ~20–25 ms
```

Env steps gained per iteration: **16**  
Wall time per iteration: dominated by **one** update  

\[
\text{SPS} \approx \frac{16}{0.02\text{–}0.03\,\mathrm{s}} \approx 500\text{–}800+
\]

Measured smoke run on this machine landed around **~750 SPS** after warmup (higher in mixed warmup windows). PPO still wins on raw SPS because it does **not** update every collection chunk of this size — it updates once per 8192 steps with a different compute pattern.

So:

| Setup | Approx SPS (this CPU box) |
|-------|---------------------------|
| Old SAC: 1 env, 1 update/step | ~40–80 |
| New SAC: 16 envs, 1 update/vector step | ~700–1100 |
| PPO: 16 envs, update every 8192 steps | ~1400–2000 |

---

## 6. Correctness checklist (things that could have broken)

| Risk | How it is handled |
|------|-------------------|
| Storing reset obs as \(s'\) after episode end | Replace with `final_obs` when `_final_obs` is set |
| Treating truncation as terminal | `done = terminations` only |
| Episode stats under vector envs | `RecordEpisodeStatistics` + `infos["episode"]` |
| Buffer wrap at capacity | `add_batch` splits across the ring end |
| Logging before first update | Losses only logged when `q_loss is not None` |
| Checkpoint spam | Save every 250k env steps via `last_save`, not every loop |

---

## 7. Files touched

| File | Change |
|------|--------|
| `algorithms/sac/sac_train.py` | Vector env loop, `add_batch`, one update per vector step, windowed SPS, `final_obs` handling |
| `algorithms/sac/sac_model.py` | **Unchanged** |
| Env / PPO / tests | **Unchanged** |

---

## 8. How to re-verify

Run these from the repository root. The trainer imports `sac_model` by bare name, so its own directory has to be importable:

```bash
WANDB_MODE=disabled PYTHONPATH=algorithms/sac python -c "
import sac_train as s
s.config.total_timesteps = 20000
s.config.start_steps = 2000
s.main()
"
# expect printed windowed SPS in the hundreds after warmup
```

Full training:

```bash
python algorithms/sac/sac_train.py
```

Watch W&B `charts/SPS` and episodic return; compare wall clock to an older single-env run if you still have logs.

---

## 9. One-sentence summary

**SAC was slow because every MuJoCo step paid for a full multi-network CPU update; we now collect 16 steps per update (like amortizing the update the way PPO amortizes its rarer updates), fix vector autoreset next-states, and log SPS in a short window so the number means something.**
