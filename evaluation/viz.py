"""Watch the policy from the most recently saved checkpoint (PPO or SAC).

Run from any directory with ``python path/to/viz.py``.  Pass ``--checkpoint``
to watch a particular ``.pt`` file instead of the newest one under
``checkpoints/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

# ``triple_pendulum`` and the ``algorithms`` trees sit one level up from this
# script, so the repository root has to be importable before they are imported.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import glfw
import gymnasium as gym
import numpy as np
import torch

import triple_pendulum  # registers InvertedTriplePendulum-v0
from algorithms.ppo.model import ActorCritic
from algorithms.sac.sac_model import Actor as SACActor

Policy = Callable[[torch.Tensor], torch.Tensor]


def newest_checkpoint(directory: Path = CHECKPOINT_DIR) -> Path:
    """Return the most recently modified checkpoint under *directory*."""
    checkpoints = [path for path in directory.rglob("*.pt") if path.is_file()]
    if not checkpoints:
        raise FileNotFoundError(
            f"No .pt checkpoints found in {directory}. Train a model first or "
            "pass --checkpoint PATH."
        )
    return max(checkpoints, key=lambda path: path.stat().st_mtime_ns)


def load_agent(checkpoint_path: Path, obs_dim: int, act_dim: int) -> Policy:
    """Load a training state dict and return a deterministic policy for it.

    Detects the checkpoint type from its keys: PPO ``ActorCritic`` state dicts
    saved by train.py contain ``actor.0.weight``; SAC ``Actor`` state dicts
    saved by sac_train.py contain ``mean_head.weight``.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"{checkpoint_path} does not contain a PyTorch state dict.")

    if "actor.0.weight" in state_dict:
        hidden_dim, checkpoint_obs_dim = state_dict["actor.0.weight"].shape
        checkpoint_act_dim = state_dict["actor.6.weight"].shape[0]
        algo = "PPO"
    elif "mean_head.weight" in state_dict:
        hidden_dim, checkpoint_obs_dim = state_dict["net.0.weight"].shape
        checkpoint_act_dim = state_dict["mean_head.weight"].shape[0]
        algo = "SAC"
    else:
        raise ValueError(
            f"{checkpoint_path} is not a checkpoint produced by "
            "algorithms/ppo/train.py or algorithms/sac/sac_train.py."
        )

    if (checkpoint_obs_dim, checkpoint_act_dim) != (obs_dim, act_dim):
        raise ValueError(
            f"Checkpoint expects observation/action dimensions "
            f"{checkpoint_obs_dim}/{checkpoint_act_dim}, but this environment uses "
            f"{obs_dim}/{act_dim}."
        )

    if algo == "PPO":
        agent = ActorCritic(obs_dim, act_dim, hidden_dim)
        agent.load_state_dict(state_dict)
        agent.eval()
        policy = agent.actor
    else:
        actor = SACActor(obs_dim, act_dim, hidden_dim)
        actor.load_state_dict(state_dict)
        actor.eval()
        policy = actor.deterministic

    print(f"Loaded {algo} checkpoint")
    return policy


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
    parser.add_argument(
        "--swingup",
        action="store_true",
        help=(
            "Watch the swing-up task: resets from a hanging chain or large "
            "angles (the distribution the swing-up trainer uses). Without "
            "this flag the env resets near-upright."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = (args.checkpoint or newest_checkpoint()).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.swingup:
        # -999: do not reset when the tip drops — you want to see the whole
        # swing-up attempt, including the failures.
        env = gym.make(
            "InvertedTriplePendulum-v0",
            render_mode="human",
            swingup=True,
            reset_angle_limit=3.14,
            termination_height=-999.0,
        )
        print("mode: swing-up (hanging / large-angle starts)")
    else:
        env = gym.make("InvertedTriplePendulum-v0", render_mode="human")
        print("mode: balance (near-upright starts; add --swingup for swing-up)")
    try:
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        policy = load_agent(checkpoint_path, obs_dim, act_dim)
        print(f"Watching checkpoint: {checkpoint_path}")

        obs, _ = env.reset()
        env.render()  # create the viewer before reading its close flag
        while not viewer_is_closed(env):
            with torch.inference_mode():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = policy(obs_t).squeeze(0).numpy()

            obs, _, terminated, truncated, _ = env.step(np.clip(action, -1.0, 1.0))
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
