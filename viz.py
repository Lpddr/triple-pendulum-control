"""Watch the policy from the most recently saved PPO checkpoint.

Run from any directory with ``python path/to/viz.py``.  Pass ``--checkpoint``
to watch a particular ``.pt`` file instead of the newest one.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import glfw
import gymnasium as gym
import numpy as np
import torch

import triple_pendulum  # registers InvertedTriplePendulum-v0
from model import ActorCritic

PROJECT_DIR = Path(__file__).resolve().parent


def newest_checkpoint(directory: Path = PROJECT_DIR) -> Path:
    """Return the most recently modified checkpoint in *directory*."""
    checkpoints = [path for path in directory.glob("*.pt") if path.is_file()]
    if not checkpoints:
        raise FileNotFoundError(
            f"No .pt checkpoints found in {directory}. Train a model first or "
            "pass --checkpoint PATH."
        )
    return max(checkpoints, key=lambda path: path.stat().st_mtime_ns)


def load_agent(checkpoint_path: Path, obs_dim: int, act_dim: int) -> ActorCritic:
    """Load a training state dict and validate that it fits this environment."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"{checkpoint_path} does not contain a PyTorch state dict.")

    try:
        hidden_dim, checkpoint_obs_dim = state_dict["actor.0.weight"].shape
        checkpoint_act_dim = state_dict["actor.6.weight"].shape[0]
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"{checkpoint_path} is not an ActorCritic checkpoint produced by train.py."
        ) from error

    if (checkpoint_obs_dim, checkpoint_act_dim) != (obs_dim, act_dim):
        raise ValueError(
            f"Checkpoint expects observation/action dimensions "
            f"{checkpoint_obs_dim}/{checkpoint_act_dim}, but this environment uses "
            f"{obs_dim}/{act_dim}."
        )

    agent = ActorCritic(obs_dim, act_dim, hidden_dim)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def viewer_is_closed(env: gym.Env) -> bool:
    """Check the Gymnasium GLFW viewer without rendering another frame.

    Gymnasium 1.0 destroys the GLFW window during ``render`` after a red-X
    event, but its renderer then tries to use that destroyed window on the next
    frame.  Checking here lets us leave the loop before that follow-up render.
    """
    renderer = getattr(env.unwrapped, "mujoco_renderer", None)
    viewer = getattr(renderer, "viewer", None)
    window = getattr(viewer, "window", None)
    return window is None or bool(glfw.window_should_close(window))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to watch. Defaults to the most recently modified .pt file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = (args.checkpoint or newest_checkpoint()).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    env = gym.make("InvertedTriplePendulum-v0", render_mode="human")
    try:
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        agent = load_agent(checkpoint_path, obs_dim, act_dim)
        print(f"Watching checkpoint: {checkpoint_path}")

        obs, _ = env.reset()
        env.render()  # create the viewer before reading its close flag
        while not viewer_is_closed(env):
            with torch.inference_mode():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = agent.actor(obs_t).squeeze(0).numpy()

            obs, _, terminated, truncated, _ = env.step(np.clip(action, -1.0, 1.0))
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
