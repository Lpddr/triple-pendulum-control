"""Evaluate a trained policy under an explicit initial-state distribution.

Why this exists
---------------
``triple_pendulum/env.py``'s ``reset_model`` currently samples a curriculum
(30% near-upright / 30% large angle / 40% hanging).  A policy trained on the
ORIGINAL distribution (``init_qpos + U(-0.1, 0.1)``, i.e. near-upright only)
will look broken when evaluated through that reset, because most episodes start
already past the termination height.

This script does NOT touch env.py.  It calls ``env.unwrapped.set_state``
directly, so the initial state is fully controlled and every policy can be
measured on the distribution it was actually trained for.

    python tools/eval_policy.py --start author
    python tools/eval_policy.py --start hanging --episodes 100
    python tools/eval_policy.py --start author --terminate      # apply tip_y<=1.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import triple_pendulum  # noqa: E402
from evaluation.viz import load_agent  # noqa: E402

ENV_ID = "InvertedTriplePendulum-v0"
SEP = "=" * 90


def sample_start(kind: str, u, rng: np.random.Generator,
                 near_prob: float = 0.0, near_width: float = 0.15,
                 angle_limit: float = 3.14,
                 vel_std: float = 0.2) -> None:
    """Set the env state to a specific starting condition."""
    qpos = u.init_qpos.copy()
    qvel = u.init_qvel.copy()

    if kind == "author":
        # exact original reset_model from the upstream repo:
        #   qpos = init_qpos + U(-0.1, 0.1)      (all 4 dofs, incl. cart)
        #   qvel = init_qvel + N(0, 1) * 0.1
        qpos = u.init_qpos + rng.uniform(-0.1, 0.1, size=u.model.nq)
        qvel = u.init_qvel + rng.standard_normal(u.model.nv) * 0.1
    elif kind == "exact":
        pass
    elif kind == "hanging":
        qpos[0] = rng.uniform(-0.1, 0.1)
        qpos[1] = np.pi + rng.uniform(-0.15, 0.15)
        qpos[2] = rng.uniform(-0.15, 0.15)
        qpos[3] = rng.uniform(-0.15, 0.15)
        qvel[:] = rng.normal(0.0, vel_std, size=u.model.nv)
    elif kind == "large":
        qpos[0] = rng.uniform(-0.1, 0.1)
        qpos[1:] = rng.uniform(-angle_limit, angle_limit, size=3)
        qvel[:] = rng.normal(0.0, vel_std, size=u.model.nv)
    elif kind == "swingup":
        # mirrors env.py's swingup reset_model branch (what the trainer sees)
        qpos[0] = rng.uniform(-0.1, 0.1)
        r = rng.random()
        p_hang = 0.5 * (1.0 - near_prob)
        if r < near_prob:
            qpos[1:] = rng.uniform(-near_width, near_width, size=3)
        elif r < near_prob + p_hang:
            qpos[1] = np.pi + rng.uniform(-0.15, 0.15)
            qpos[2] = rng.uniform(-0.15, 0.15)
            qpos[3] = rng.uniform(-0.15, 0.15)
        else:
            qpos[1:] = rng.uniform(-angle_limit, angle_limit, size=3)
        qvel[:] = rng.normal(0.0, vel_std, size=u.model.nv)
    elif kind == "mixed":
        # what env.py currently does
        qpos[0] = rng.uniform(-0.1, 0.1)
        r = rng.random()
        if r < 0.3:
            qpos[1:] = rng.uniform(-0.15, 0.15, size=3)
        elif r < 0.6:
            qpos[1:] = rng.uniform(-1.2, 1.2, size=3)
        else:
            qpos[1] = np.pi + rng.uniform(-0.15, 0.15)
            qpos[2] = rng.uniform(-0.15, 0.15)
            qpos[3] = rng.uniform(-0.15, 0.15)
        qvel[:] = rng.normal(0.0, vel_std, size=u.model.nv)
    else:
        raise ValueError(f"unknown start kind {kind!r}")

    u.set_state(qpos, qvel)
    mujoco.mj_forward(u.model, u.data)


def evaluate(policy, start: str, episodes: int, terminate: bool, seed0: int = 0,
             swingup: bool = False, near_prob=None, near_width=None, angle_limit=None,
             vel_std=None, frame_skip=None, max_episode_steps=None,
             joint_damping=None):
    env_kw = {}
    if swingup:
        env_kw = dict(swingup=True, reset_angle_limit=angle_limit if angle_limit is not None else 3.14)
        if near_prob is not None:
            env_kw["near_prob"] = near_prob
        if near_width is not None:
            env_kw["near_width"] = near_width
    if frame_skip is not None:
        env_kw["frame_skip"] = frame_skip
    damping_patch = joint_damping
    env = gym.make(
        ENV_ID,
        termination_height=1.5 if terminate else -999.0,
        max_episode_steps=max_episode_steps if max_episode_steps is not None else 1000,
        **env_kw,
    )
    if damping_patch is not None:
        env.unwrapped.model.dof_damping[1:] = damping_patch
    u = env.unwrapped
    step_limit = max_episode_steps if max_episode_steps is not None else 1000
    rows = []
    for ep in range(episodes):
        rng = np.random.default_rng(seed0 + ep)
        env.reset(seed=seed0 + ep)          # clear the "must reset" flag
        sample_start(start, u, rng,
                     near_prob=near_prob if near_prob is not None else 0.0,
                     near_width=near_width if near_width is not None else 0.15,
                     angle_limit=angle_limit if angle_limit is not None else 3.14,
                     vel_std=vel_std if vel_std is not None else 0.2)

        tip0 = float(u.data.site_xpos[0][2])
        tip_hist = []
        angle_hist, cart_hist, action_hist = [], [], []
        ever_upright = False
        n = 0
        while True:
            with torch.inference_mode():
                a = policy(
                    torch.as_tensor(u._get_obs(), dtype=torch.float32).unsqueeze(0)
                ).squeeze(0).numpy()
            a = np.clip(a, -1.0, 1.0)
            u.data.ctrl[:] = a
            for _ in range(u.frame_skip):
                mujoco.mj_step(u.model, u.data, nstep=1)
            mujoco.mj_forward(u.model, u.data)
            n += 1
            tip = float(u.data.site_xpos[0][2])
            tip_hist.append(tip)
            angle_hist.append(np.max(np.abs(np.arctan2(
                np.sin(np.cumsum(u.data.qpos[1:])),
                np.cos(np.cumsum(u.data.qpos[1:])),
            ))))
            cart_hist.append(abs(float(u.data.qpos[0])))
            action_hist.append(float(np.max(np.abs(a))))
            # --terminate OFF => never stop early, so swing-up capability is
            # measured over the full 1000-step horizon instead of being cut off
            # at the first moment the tip drops below the balance threshold.
            ever_upright = ever_upright or tip > 1.7
            terminated = bool(terminate and tip <= 1.5 and (not swingup or ever_upright))
            truncated = n >= step_limit
            if terminated or truncated:
                break
        tip_hist = np.array(tip_hist)
        # HR task requirement: "hold upright >= 10 s" -> 10 s / 0.05 s = 200
        # consecutive control steps above the balance threshold.
        above = tip_hist > 1.7
        longest = 0
        cur = 0
        for flag in above:
            cur = cur + 1 if flag else 0
            longest = max(longest, cur)
        rows.append(
            {
                "tip0": tip0,
                "n": n,
                "tip_end": tip_hist[-1],
                "tip_best": float(tip_hist.max()),
                "tip_min": float(tip_hist.min()),
                "tip_mean": float(tip_hist.mean()),
                "ang_max": float(np.max(angle_hist)),
                "cart_max": float(np.max(cart_hist)),
                "sat": float(np.max(action_hist)),
                "sat_fraction": float(np.mean(np.asarray(action_hist) > 0.95)),
                "held": float(np.mean(tip_hist > 1.7)),
                "hold_s": longest * u.dt,
                "limit": step_limit,
            }
        )
    env.close()
    return rows


def report(start: str, rows, terminate: bool) -> None:
    step_limit = rows[0]["limit"]
    n = np.array([r["n"] for r in rows])
    best = np.array([r["tip_best"] for r in rows])
    tip0 = np.array([r["tip0"] for r in rows])
    ang = np.array([r["ang_max"] for r in rows])
    cart = np.array([r["cart_max"] for r in rows])
    sat = np.array([r["sat"] for r in rows])
    held = np.array([r["held"] for r in rows])
    hold_s = np.array([r["hold_s"] for r in rows])

    print(f"  episodes            : {len(rows)}")
    print(f"  early termination   : {'ON  (tip_y <= 1.5 stops the episode)' if terminate else 'OFF (always runs the full 1000 steps)'}")
    print(f"  start tip_y         : min={tip0.min():7.3f}  mean={tip0.mean():7.3f}  max={tip0.max():7.3f}")
    print(f"  episode length      : mean={n.mean():7.1f}  median={np.median(n):6.0f}  "
          f"min={n.min():4d}  max={n.max():4d}")
    print(f"  reached full horizon: {(n >= step_limit).sum()}/{len(rows)}")
    print(f"  best tip_y attained : min={best.min():7.3f}  mean={best.mean():7.3f}  max={best.max():7.3f}")
    print(f"  episodes with best tip_y >= 1.50 : {(best >= 1.5).sum()}/{len(rows)}")
    print(f"  episodes with best tip_y >= 1.70 : {(best >= 1.7).sum()}/{len(rows)}")
    print(f"  fraction of steps with tip_y>1.7 : mean={held.mean()*100:5.1f}%")
    print(f"  LONGEST CONTINUOUS hold above 1.7 (HR needs >= 10.0 s):")
    print(f"      mean={hold_s.mean():6.2f} s   median={np.median(hold_s):6.2f} s   "
          f"max={hold_s.max():6.2f} s   episodes passing 10 s: {(hold_s >= 10.0).sum()}/{len(rows)}")
    print(f"  max |hinge angle|   : {ang.max():.3f} rad      max |cart x| : {cart.max():.3f}")
    print(f"  max |action|        : {sat.max():.3f}         action saturation(>0.95) share: "
          f"{np.mean([r['sat_fraction'] for r in rows])*100:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "sac_triple_final.pt")
    ap.add_argument("--start", default="author",
                    choices=["author", "hanging", "large", "mixed", "exact", "swingup"])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--terminate", action="store_true",
                    help="apply termination_height=1.5 (fidelity to the original task)")
    ap.add_argument("--swingup", action="store_true",
                    help="build the env in swingup mode (conditional termination, "
                         "same distribution the swing-up trainer sees)")
    ap.add_argument("--near-prob", type=float, default=None,
                    help="override env near_prob when --swingup is set")
    ap.add_argument("--near-width", type=float, default=None,
                    help="override env near_width when --swingup is set")
    ap.add_argument("--angle-limit", type=float, default=None,
                    help="override env reset_angle_limit when --swingup is set")
    ap.add_argument("--vel-std", type=float, default=None,
                    help="override reset qvel std when --swingup is set")
    ap.add_argument("--frame-skip", type=int, default=None,
                    help="override env frame_skip (2 = 50 Hz control)")
    ap.add_argument("--joint-damping", type=float, default=None,
                    help="override hinge damping (must match the trained env)")
    ap.add_argument("--max-episode-steps", type=int, default=None)
    ap.add_argument("--all-starts", action="store_true",
                    help="run every start distribution with the same policy")
    args = ap.parse_args()

    ckpt = args.checkpoint
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    probe = gym.make(ENV_ID)
    policy = load_agent(ckpt, probe.observation_space.shape[0], probe.action_space.shape[0])
    probe.close()
    print(f"checkpoint : {ckpt}")
    print(f"md5        : ", end="")
    import hashlib
    print(hashlib.md5(ckpt.read_bytes()).hexdigest())

    starts = ["author", "hanging", "large", "mixed", "exact"] if args.all_starts else [args.start]
    for i, st in enumerate(starts):
        if i:
            print()
        print(SEP)
        print(f"START DISTRIBUTION = {st}")
        print(SEP)
        rows = evaluate(policy, st, args.episodes, args.terminate, swingup=args.swingup,
                        near_prob=args.near_prob, near_width=args.near_width,
                        angle_limit=args.angle_limit, vel_std=args.vel_std,
                        frame_skip=args.frame_skip, max_episode_steps=args.max_episode_steps,
                        joint_damping=args.joint_damping)
        report(st, rows, args.terminate)


if __name__ == "__main__":
    main()
