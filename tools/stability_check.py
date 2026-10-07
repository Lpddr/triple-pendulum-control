"""Simulation-stability check for InvertedTriplePendulum-v0.

Rerun after ANY change to the XML asset (masses, timestep, frame_skip, solref),
to the integrator, or to the env's step/reset code.

    python tools/stability_check.py            # full check (~1 min)
    python tools/stability_check.py --quick    # skip the timestep-boundary sweep

Every section prints PASS/FAIL-style raw numbers rather than asserts, so this
is a diagnostic, not a unit test.  The unit-level invariants live in tests/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import triple_pendulum  # noqa: E402  (registers InvertedTriplePendulum-v0)

SEP = "=" * 84
ENV_ID = "InvertedTriplePendulum-v0"
RESULTS: list[tuple[str, str]] = []


def hdr(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL"))
    print(f"  [{('PASS' if ok else 'FAIL')}] {name}: {detail}")


def make(**kw):
    kw.setdefault("reset_noise_scale", 0.0)
    kw.setdefault("termination_height", -999.0)
    return gym.make(ENV_ID, **kw)


# --------------------------------------------------------------------------
def section_config() -> None:
    hdr("1. MODEL / INTEGRATOR CONFIGURATION")
    env = make()
    u = env.unwrapped
    m = u.model
    names = {0: "EULER", 1: "RK4", 2: "IMPLICIT", 3: "IMPLICITFAST"}
    print(f"  nq={m.nq} nv={m.nv} nu={m.nu}   integrator={names.get(m.opt.integrator, m.opt.integrator)}")
    print(f"  physics timestep = {m.opt.timestep} s   frame_skip = {u.frame_skip}   env.dt = {u.dt} s")
    print(f"  solver: iterations={m.opt.iterations} tolerance={m.opt.tolerance} ls_iterations={m.opt.ls_iterations}")
    print(f"  masses = {m.body_mass}   total = {m.body_mass.sum():.3f} kg")
    print(f"  actuator gear = {m.actuator_gear[0, 0]}   ctrlrange = {m.actuator_ctrlrange[0]}")
    for i in range(m.njnt):
        j = m.joint(i)
        dof = j.dofadr[0]
        print(
            f"    joint[{i}] {j.name:8s} damping={m.dof_damping[dof]:.4f} armature={m.dof_armature[dof]:.4f} "
            f"limited={int(m.jnt_limited[i])} range={m.jnt_range[i]}"
        )
    # RK4 + hard limits: MuJoCo docs recommend implicit/implicitfast when the model
    # contains constraints.  The only constraint here is the slider joint limit.
    n_con = int(m.njnt and sum(1 for i in range(m.njnt) if m.jnt_limited[i]))
    print(f"  limited joints (constraints) = {n_con}; contacts/geoms with contype!=0 = "
          f"{int(np.count_nonzero(m.geom_contype))}")
    record("config.dt", u.dt == 0.05, f"control interval {u.dt:.3f} s ({1/u.dt:.0f} Hz)")
    env.close()


# --------------------------------------------------------------------------
def total_energy(model, data) -> float:
    M = np.zeros((model.nv, model.nv))
    if hasattr(data, "qM"):
        mujoco.mj_fullM(model, M, data.qM)
    else:
        mujoco.mj_fullM(model, data, M)
    ke = 0.5 * float(data.qvel @ M @ data.qvel)
    pe = sum(
        model.body_mass[i] * -model.opt.gravity[2] * data.xipos[i][2]
        for i in range(model.nbody)
    )
    return ke + float(pe)


def section_passivity() -> None:
    hdr("2. PASSIVITY — zero action, is the integrator injecting energy?")
    env = make()
    u = env.unwrapped
    u.set_state(np.array([0.0, 0.02, 0.0, 0.0]), np.zeros(4))
    mujoco.mj_forward(u.model, u.data)
    e0 = prev = total_energy(u.model, u.data)
    worst = -np.inf
    worst_t = 0.0
    for step in range(1, 2001):
        u.do_simulation(np.array([0.0]), u.frame_skip)
        e = total_energy(u.model, u.data)
        if e - prev > worst:
            worst, worst_t = e - prev, step * u.dt
        prev = e
    print(f"  E(t=0) = {e0:.6f} J   E(t=100 s) = {prev:.6f} J   drift = {prev - e0:+.6f} J")
    print(f"  largest single-step energy increase = {worst:+.3e} J (at t={worst_t:.2f} s)")
    # Damping=0.05 on every DoF must make dE <= 0 on every step.  A positive value
    # means the integrator is manufacturing energy -> the classic instability tell.
    record("passivity.no_energy_injection", worst <= 0.0, f"max step-wise dE = {worst:+.3e} J")
    env.close()


# --------------------------------------------------------------------------
def rollout_qpos(integrator: int, dt: float, horizon: float, q0, v0):
    env = make()
    u = env.unwrapped
    u.model.opt.integrator = integrator
    u.model.opt.timestep = dt
    u.frame_skip = max(1, int(round(0.05 / dt)))
    u.set_state(np.asarray(q0, float), np.asarray(v0, float))
    mujoco.mj_forward(u.model, u.data)
    for _ in range(int(horizon / dt)):
        u.data.ctrl[:] = 0.0
        mujoco.mj_step(u.model, u.data, nstep=1)
    q = u.data.qpos.copy()
    env.close()
    return q


def section_convergence() -> None:
    hdr("3. TIMESTEP CONVERGENCE — is dt=0.01 inside the asymptotic regime?")
    q0, v0, horizon = [0.0, 0.35, 0.10, -0.05], np.zeros(4), 2.0
    # 2 s is short enough that chaos has not yet amplified discretisation error
    # into pure trajectory decorrelation; see the note in section 6.
    ref = rollout_qpos(1, 0.001, horizon, q0, v0)
    print(f"  reference: RK4 @ dt=0.001 (2000 physics steps), qpos = {ref}")
    errs = {}
    for dt in (0.02, 0.01, 0.005, 0.002):
        q = rollout_qpos(1, dt, horizon, q0, v0)
        errs[dt] = float(np.max(np.abs(q - ref)))
        print(f"  RK4 dt={dt:<6} qpos = {q}   max|err| = {errs[dt]:.3e}")
    # order check: halving dt should shrink the error by roughly 2^4 (RK4)
    orders = [
        np.log2(errs[a] / errs[b]) for a, b in ((0.02, 0.01), (0.01, 0.005), (0.005, 0.002))
    ]
    print(f"  observed convergence order per halving: {[f'{o:.1f}' for o in orders]}  (RK4 nominal: 4-5)")
    record("convergence.dt_0.01", errs[0.01] < 5e-3, f"max|err| at dt=0.01 = {errs[0.01]:.3e} rad")


# --------------------------------------------------------------------------
def section_integrator_order() -> None:
    hdr("4. INTEGRATOR ORDER — is RK4 actually the right choice here?")
    q0, v0, horizon = [0.0, 0.35, 0.10, -0.05], np.zeros(4), 1.0
    # 1 s keeps chaos from decorrelating the trajectories, so the measured
    # difference is genuine truncation error rather than trajectory divergence.
    ref = rollout_qpos(1, 0.0001, horizon, q0, v0)
    print(f"  reference: RK4 @ dt=1e-4, qpos = {ref}")
    print("  integrator      err@0.01     err@0.005    ratio   observed order")
    orders = {}
    for integ, name in ((1, "RK4"), (0, "EULER"), (3, "IMPLICITFAST"), (2, "IMPLICIT")):
        e1 = float(np.max(np.abs(rollout_qpos(integ, 0.01, horizon, q0, v0) - ref)))
        e2 = float(np.max(np.abs(rollout_qpos(integ, 0.005, horizon, q0, v0) - ref)))
        order = float(np.log2(e1 / e2))
        orders[name] = order
        print(f"  {name:14s} {e1:.4e}   {e2:.4e}   {e1/e2:5.2f}   {order:5.2f}")
    print("  (Euler / implicit / implicitfast are 1st order in MuJoCo; only RK4 is 4th)")
    record("integrator.rk4_is_4th_order", orders["RK4"] > 3.5, f"observed order {orders['RK4']:.2f}")
    record(
        "integrator.rk4_beats_1st_order",
        orders["RK4"] > orders["EULER"] + 2.0,
        f"RK4 {orders['RK4']:.2f} vs EULER {orders['EULER']:.2f}",
    )


# --------------------------------------------------------------------------
def peak_qvel(integ: int, dt: float, n_ctrl: int = 1200, seconds_per_ctrl: float = 0.05) -> float:
    rng = np.random.default_rng(11)
    ctrl = rng.uniform(-1, 1, size=n_ctrl)
    env = make()
    u = env.unwrapped
    u.model.opt.integrator = integ
    u.model.opt.timestep = dt
    u.frame_skip = max(1, int(round(seconds_per_ctrl / dt)))
    u.set_state(np.array([0.0, 0.10, 0.0, 0.0]), np.zeros(4))
    mujoco.mj_forward(u.model, u.data)
    pv = 0.0
    for k in range(n_ctrl):
        u.data.ctrl[:] = ctrl[k]
        for _ in range(u.frame_skip):
            mujoco.mj_step(u.model, u.data, nstep=1)
            pv = max(pv, float(np.max(np.abs(u.data.qvel))))
    env.close()
    return pv


def section_timestep_margin(quick: bool) -> None:
    hdr("5. RK4 TIMESTEP MARGIN — where does it actually blow up?")
    grid = (0.02, 0.05) if quick else (0.02, 0.04, 0.05, 0.055, 0.06, 0.07, 0.08, 0.1)
    boundary = None
    for dt in grid:
        pv = peak_qvel(1, dt)
        ok = pv < 500.0
        print(f"  RK4 dt={dt:<6} peak|qvel| = {pv:14.3f}   {'ok' if ok else '*** DIVERGED ***'}")
        if not ok and boundary is None:
            boundary = dt
    shipped = 0.01
    if boundary is None:
        record("timestep.margin", True, f"no divergence up to dt={max(grid)} -> >{max(grid)/shipped:.0f}x margin at dt={shipped}")
    else:
        margin = boundary / shipped
        record("timestep.margin", margin >= 3.0, f"diverges at dt~{boundary} -> {margin:.1f}x margin at shipped dt={shipped}")


# --------------------------------------------------------------------------
def section_equilibrium() -> None:
    hdr("6. EXACT-UPRIGHT EQUILIBRIUM — spurious drift / asymmetry injection?")
    env = make()
    u = env.unwrapped
    u.set_state(np.zeros(4), np.zeros(4))
    mujoco.mj_forward(u.model, u.data)
    print(f"  qacc at the fixed point = {u.data.qacc}  (all-zero => exact equilibrium)")
    z0 = float(u.data.site_xpos[0][2])
    for _ in range(int(100.0 / u.dt)):
        u.data.ctrl[:] = 0.0
        for _ in range(u.frame_skip):
            mujoco.mj_step(u.model, u.data, nstep=1)
    z1 = float(u.data.site_xpos[0][2])
    drift = z1 - z0
    print(f"  tip_z: {z0:.15f} -> {z1:.15f}   drift = {drift:+.3e} m over 100 s")
    print(f"  max|qvel| = {np.max(np.abs(u.data.qvel)):.3e}")
    record("equilibrium.no_drift", abs(drift) < 1e-9, f"drift = {drift:+.2e} m / 100 s")
    env.close()

    hdr("7. SADDLE GROWTH RATE — how fast does the uncontrolled rig fall?")
    print("  initial lean d (rad, per hinge) | time until tip_y < 1.5")
    data = []
    for d in (0.1, 0.02, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        env = make()
        u = env.unwrapped
        q = np.zeros(4)
        q[1:] = d
        u.set_state(q, np.zeros(4))
        mujoco.mj_forward(u.model, u.data)
        t = 0.0
        while u.data.site_xpos[0][2] > 1.5 and t < 60.0:
            u.data.ctrl[:] = 0.0
            for _ in range(u.frame_skip):
                mujoco.mj_step(u.model, u.data, nstep=1)
            t += u.dt
        data.append((d, t))
        print(f"  {d:<31.0e} | {t:6.3f} s")
        env.close()
    arr = np.array(data)
    lam = np.median(np.log(arr[:-1, 0] / arr[1:, 0]) / (arr[1:, 1] - arr[:-1, 1]))
    print(f"  divergence rate lambda ~= {lam:.2f} 1/s  -> time constant tau ~= {1/lam*1000:.0f} ms")
    print(f"  one extra decade of initial error buys only {np.log(10)/lam*1000:.0f} ms of recovery time")
    print(f"  control interval dt={0.05:.2f} s covers {0.05*lam:.2f} time constants")
    record("saddle.rate_physical", 5.0 < lam < 30.0, f"lambda = {lam:.2f} 1/s (tau = {1000/lam:.0f} ms)")

    # Re-measure the fall time with RK4 refined 20x.  RK4 self-converges, so
    # matching fall times proves lambda is physics and not discretisation.
    print("  cross-check, RK4 refined (fall time should match the dt=0.01 column):")
    for i, d in enumerate((1e-2, 1e-4, 1e-6)):
        env = make()
        u = env.unwrapped
        u.model.opt.timestep = 0.0005
        u.frame_skip = 100
        q = np.zeros(4)
        q[1:] = d
        u.set_state(q, np.zeros(4))
        mujoco.mj_forward(u.model, u.data)
        t = 0.0
        while u.data.site_xpos[0][2] > 1.5 and t < 60.0:
            u.data.ctrl[:] = 0.0
            for _ in range(u.frame_skip):
                mujoco.mj_step(u.model, u.data, nstep=1)
            t += 0.05
        j = int(np.argmin(np.abs(arr[:, 0] - d)))
        print(f"    d={d:.0e}: RK4@0.01 -> {arr[j, 1]:.3f} s | RK4@0.0005 -> {t:.3f} s "
              f"| delta = {abs(t - arr[j, 1])*1000:.0f} ms")
        env.close()


# --------------------------------------------------------------------------
def section_long_rollout(steps: int = 200_000) -> None:
    hdr(f"8. LONG DRIVEN ROLLOUT — {steps} physics steps ({steps*0.01:.0f} s sim time)")
    rng = np.random.default_rng(0)
    env = make(reset_noise_scale=0.1)
    u = env.unwrapped
    u.set_state(np.zeros(4), np.zeros(4))
    mujoco.mj_forward(u.model, u.data)
    peak = {"qpos": 0.0, "qvel": 0.0, "qacc": 0.0, "constraint": 0.0}
    nonfinite = None
    for k in range(steps):
        if k % 7 == 0:
            u.data.ctrl[:] = rng.uniform(-1, 1, size=1)
        mujoco.mj_step(u.model, u.data, nstep=1)
        if not np.all(np.isfinite(u.data.qpos)) or not np.all(np.isfinite(u.data.qvel)):
            nonfinite = k
            break
        peak["qpos"] = max(peak["qpos"], float(np.max(np.abs(u.data.qpos))))
        peak["qvel"] = max(peak["qvel"], float(np.max(np.abs(u.data.qvel))))
        peak["qacc"] = max(peak["qacc"], float(np.max(np.abs(u.data.qacc))))
        peak["constraint"] = max(peak["constraint"], float(np.max(np.abs(u.data.qfrc_constraint))))
    print(f"  peaks: max|qpos|={peak['qpos']:.1f} rad  max|qvel|={peak['qvel']:.2f}  "
          f"max|qacc|={peak['qacc']:.1f}  max|qfrc_constraint|={peak['constraint']:.1f}")
    print(f"  slider joint-range violation at the end = "
          f"{max(0.0, -u.data.qpos[0] - 1.0, u.data.qpos[0] - 1.0):.2e} m")
    record("rollout.no_nan", nonfinite is None,
           "finite over the whole run" if nonfinite is None else f"non-finite at step {nonfinite}")
    env.close()


# --------------------------------------------------------------------------
def section_policy() -> None:
    hdr("9. TRAINED POLICY — closed-loop margin vs reset noise")
    try:
        import torch

        from evaluation.viz import load_agent

        ckpt = REPO_ROOT / "sac_triple_final.pt"
        if not ckpt.is_file():
            print(f"  skipped: {ckpt.name} not found")
            return
        env = gym.make(ENV_ID)  # default reset_noise_scale = 0.1
        policy = load_agent(ckpt, env.observation_space.shape[0], env.action_space.shape[0])
        for noise in (0.0, 0.05, 0.1, 0.2):
            lens, tip_min, cart_max, ang_max = [], 1e9, 0.0, 0.0
            for ep in range(20):
                u = env.unwrapped
                env.reset(seed=1000 + ep)  # clear Gymnasium's "must reset" flag
                r = np.random.default_rng(1000 + ep)
                u.set_state(
                    u.init_qpos + r.uniform(-noise, noise, size=u.model.nq),
                    u.init_qvel + r.standard_normal(u.model.nv) * noise,
                )
                mujoco.mj_forward(u.model, u.data)
                obs = u._get_obs()
                n = 0
                while True:
                    with torch.inference_mode():
                        a = policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                    obs, _r, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
                    n += 1
                    tip_min = min(tip_min, info["tip_y"])
                    cart_max = max(cart_max, abs(u.data.qpos[0]))
                    ang_max = max(ang_max, float(np.abs(u.data.qpos[1:]).max()))
                    if term or trunc:
                        break
                lens.append(n)
            lens = np.array(lens)
            print(f"  noise={noise:<5} len mean={lens.mean():7.1f} min={lens.min():4d} "
                  f"full={int((lens >= 1000).sum()):2d}/20 | max|angle|={ang_max:5.3f} rad "
                  f"max|cart x|={cart_max:5.3f} min tip_y={tip_min:.4f}")
        print("  trainers use reset_noise_scale=0.05; the env default is 0.1 "
              "(see README 'Known limitations').")
        env.close()
    except Exception as exc:  # noqa: BLE001
        print(f"  skipped: {type(exc).__name__}: {exc}")


def section_reset_distribution(n: int = 3000) -> None:
    hdr("10. RESET DISTRIBUTION — do the initial states survive the termination test?")
    env = gym.make(ENV_ID)
    u = env.unwrapped
    th = u._termination_height
    tip = np.empty(n)
    for i in range(n):
        env.reset(seed=i)
        tip[i] = float(u.data.site_xpos[0][2])
    dead = float(np.mean(tip <= th))
    print(f"  tip_z at reset: min={tip.min():.3f}  median={np.median(tip):.3f}  max={tip.max():.3f}")
    print(f"  initial states already below termination_height={th}: {100*dead:.1f}%")
    n_imm = 0
    rews = np.empty(n)
    for i in range(n):
        env.reset(seed=10_000 + i)
        _, r, term, _, _ = env.step(np.array([0.0], dtype=np.float32))
        n_imm += int(term)
        rews[i] = r
    imm = n_imm / n
    print(f"  episodes terminated on the first step: {100*imm:.1f}%")
    print(f"  first-step reward: min={rews.min():.3f} mean={rews.mean():.3f} max={rews.max():.3f} "
          f"| share < -5: {100*np.mean(rews < -5):.1f}%")
    # A healthy curriculum may include hard starts, but if most resets are
    # already past the termination test the episode never gets a chance to act.
    record("reset.survives_termination", imm < 0.5, f"{100*imm:.1f}% die on step 1")
    env.close()


def section_reset_noise_param() -> None:
    hdr("11. reset_noise_scale — is the constructor argument actually wired up?")
    stats = {}
    for s in (0.0, 0.05, 0.2, 1.0):
        env = gym.make(ENV_ID, reset_noise_scale=s)
        u = env.unwrapped
        q, v = [], []
        for i in range(200):
            env.reset(seed=i)
            q.append(u.data.qpos.copy())
            v.append(u.data.qvel.copy())
        q, v = np.array(q), np.array(v)
        stats[s] = (float(q[:, 0].std()), float(q[:, 1:].std()), float(v.std()))
        print(f"  reset_noise_scale={s:<5} std(qpos[0])={stats[s][0]:.4f}  "
              f"std(qpos[1:])={stats[s][1]:.4f}  std(qvel)={stats[s][2]:.4f}")
        env.close()
    ref = stats[0.0]
    spread = max(
        abs(t[k] - ref[k]) for t in stats.values() for k in range(len(ref))
    )
    live = spread > 1e-6 * max(1.0, max(abs(x) for x in ref))
    if not live:
        print(f"  -> all four settings agree to {spread:.2e}: reset_noise_scale is DEAD "
              "(reset_model hard-codes its own noise)")
    record("reset.noise_param_live", live,
           f"spread across settings = {spread:.2e} "
           + ("(argument changes the reset distribution)" if live else "(argument has no effect)"))


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="skip the timestep-boundary sweep")
    args = ap.parse_args()

    print(f"Checking {ENV_ID}   mujoco={mujoco.__version__}   gymnasium={gym.__version__}")
    section_config()
    section_passivity()
    section_convergence()
    section_integrator_order()
    section_timestep_margin(args.quick)
    section_equilibrium()
    section_long_rollout()
    section_policy()
    section_reset_distribution()
    section_reset_noise_param()

    hdr("SUMMARY")
    for name, status in RESULTS:
        print(f"  {status}  {name}")
    fails = [n for n, s in RESULTS if s == "FAIL"]
    print(f"\n  {len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
    if fails:
        print("  failing: " + ", ".join(fails))


if __name__ == "__main__":
    main()
