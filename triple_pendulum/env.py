"""Inverted Triple Pendulum — Gymnasium MuJoCo environment.

Faithful extension of Gymnasium's ``InvertedDoublePendulum-v5`` with a third
pole stacked on top of the second. Observation / reward / termination logic
mirror the double-pendulum design, scaled for three links.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

__all__ = ["InvertedTriplePendulumEnv", "DEFAULT_CAMERA_CONFIG"]

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_DEFAULT_XML = str(_ASSETS_DIR / "inverted_triple_pendulum.xml")

# Taller system than double: three 0.6 m poles → tip at 1.8 m when upright.
DEFAULT_CAMERA_CONFIG: Dict[str, Union[float, int, np.ndarray]] = {
    "trackbodyid": 0,
    "distance": 5.0,
    "lookat": np.array((0.0, 0.0, 0.55)),
    "elevation": -15.0,
}


class InvertedTriplePendulumEnv(MujocoEnv, utils.EzPickle):
    r"""Cart + three hinged poles balanced by continuous cart force (MuJoCo).

    Nearly identical to Gymnasium ``InvertedDoublePendulum-v5``, with one extra
    pole on the free end.

    ## Action Space
    Continuous force on the cart: ``Box(-1, 1, (1,), float32)``.

    ## Observation Space
    ``Box(-inf, inf, (12,), float64)``:

    | Index | Content |
    |-------|---------|
    | 0     | cart x position |
    | 1–3   | sin(pole angles) for hinges 1–3 |
    | 4–6   | cos(pole angles) for hinges 1–3 |
    | 7–10  | cart velocity + angular velocities (clipped ±10) |
    | 11    | cart constraint force x (clipped ±10) |

    ## Reward
    ``reward = alive_bonus - distance_penalty - velocity_penalty``

    - alive_bonus: ``healthy_reward`` (default 10) while tip is healthy
    - distance_penalty: ``0.01 x_tip² + (y_tip - target_height)²``
    - velocity_penalty: weighted squared angular velocities of the three hinges

    ## Termination
    Unhealthy when the tip site's y-coordinate is ≤ ``termination_height``
    (default 1.5; fully upright tip is at 1.8 m).
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }

    def __init__(
        self,
        xml_file: str = _DEFAULT_XML,
        frame_skip: int = 5,
        default_camera_config: Optional[Dict[str, Union[float, int, np.ndarray]]] = None,
        healthy_reward: float = 10.0,
        reset_noise_scale: float = 0.1,
        termination_height: float = 1.5,
        target_height: float = 1.8,
        **kwargs: Any,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            healthy_reward,
            reset_noise_scale,
            termination_height,
            target_height,
            **kwargs,
        )

        self._healthy_reward = healthy_reward
        self._reset_noise_scale = reset_noise_scale
        self._termination_height = termination_height
        self._target_height = target_height

        # 1 cart + 3 sin + 3 cos + 4 vel + 1 constraint = 12
        observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)

        if default_camera_config is None:
            default_camera_config = dict(DEFAULT_CAMERA_CONFIG)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        x, _, y = self.data.site_xpos[0]
        observation = self._get_obs()
        terminated = bool(y <= self._termination_height)
        reward, reward_info = self._get_rew(x, y, terminated)

        if self.render_mode == "human":
            self.render()

        # truncation handled by TimeLimit wrapper from gym.make / registration
        return observation, reward, terminated, False, reward_info

    def _get_rew(self, x: float, y: float, terminated: bool):
        # qvel: [cart, hinge, hinge2, hinge3]
        v1, v2, v3 = self.data.qvel[1:4]
        dist_penalty = 0.01 * x**2 + (y - self._target_height) ** 2
        # Mirror double-pendulum weighting; add stronger weight on the top link
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2 + 1e-2 * v3**2
        alive_bonus = self._healthy_reward * int(not terminated)

        reward = float(alive_bonus - dist_penalty - vel_penalty)
        reward_info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "velocity_penalty": -vel_penalty,
            "tip_x": float(x),
            "tip_y": float(y),
        }
        return reward, reward_info

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.data.qpos[:1],  # cart x
                np.sin(self.data.qpos[1:]),  # 3 link angles
                np.cos(self.data.qpos[1:]),
                np.clip(self.data.qvel, -10, 10),  # 4 velocities
                np.clip(self.data.qfrc_constraint, -10, 10)[:1],
            ]
        ).ravel()

    def reset_model(self) -> np.ndarray:
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self.set_state(
            self.init_qpos
            + self.np_random.uniform(
                low=noise_low, high=noise_high, size=self.model.nq
            ),
            self.init_qvel
            + self.np_random.standard_normal(self.model.nv) * self._reset_noise_scale,
        )
        return self._get_obs()
