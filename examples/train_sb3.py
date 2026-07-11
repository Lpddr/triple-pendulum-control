#!/usr/bin/env python3
"""Optional PPO training with Stable-Baselines3 (install extras to use).

    pip install stable-baselines3[extra]
    python examples/train_sb3.py --timesteps 200_000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym

import triple_pendulum  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="models/ppo_triple_pendulum")
    parser.add_argument("--n-envs", type=int, default=4)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is required for this script:\n"
            "  pip install 'stable-baselines3[extra]'"
        ) from exc

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _make():
        return gym.make("InvertedTriplePendulum-v0")

    vec_env = make_vec_env(_make, n_envs=args.n_envs, seed=args.seed)
    eval_env = make_vec_env(_make, n_envs=1, seed=args.seed + 1000)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(out.parent / "tb"),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out.parent / "best"),
        log_path=str(out.parent / "eval"),
        eval_freq=max(10_000 // args.n_envs, 1),
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    model.save(str(out))
    print(f"saved model → {out}.zip")
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
