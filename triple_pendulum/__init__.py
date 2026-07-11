"""Inverted Triple Pendulum environment package (Gymnasium + MuJoCo)."""

from __future__ import annotations

from gymnasium.envs.registration import register

from triple_pendulum.env import DEFAULT_CAMERA_CONFIG, InvertedTriplePendulumEnv

__all__ = [
    "InvertedTriplePendulumEnv",
    "DEFAULT_CAMERA_CONFIG",
    "register_envs",
]

__version__ = "0.1.0"

_REGISTERED = False


def register_envs() -> None:
    """Idempotently register ``InvertedTriplePendulum-v0`` with Gymnasium."""
    global _REGISTERED
    if _REGISTERED:
        return

    register(
        id="InvertedTriplePendulum-v0",
        entry_point="triple_pendulum.env:InvertedTriplePendulumEnv",
        max_episode_steps=1000,
        reward_threshold=9100.0,
    )
    _REGISTERED = True


# Auto-register on import for `import triple_pendulum; gym.make(...)`
register_envs()
