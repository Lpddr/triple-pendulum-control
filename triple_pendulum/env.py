"""Inverted Triple Pendulum — Gymnasium MuJoCo environment.

Faithful extension of Gymnasium's ``InvertedDoublePendulum-v5`` with a third
pole stacked on top of the second. Observation / reward / termination logic
mirror the double-pendulum design, scaled for three links.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import mujoco
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
        swingup: bool = False,
        reset_angle_limit: Optional[float] = None,
        energy_weight: float = 1.0,
        dist_weight: float = 0.2,
        near_prob: float = 0.0,
        near_width: float = 0.15,
        capture_reward: bool = False,
        capture_zone: float = 1.6,
        capture_h: float = 4.0,
        capture_vel_w: float = 1.0,
        near_vel_std: float = 0.2,
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
            swingup,
            reset_angle_limit,
            energy_weight,
            dist_weight,
            near_prob,
            near_width,
            capture_reward,
            capture_zone,
            capture_h,
            capture_vel_w,
            near_vel_std,
            **kwargs,
        )

        self._healthy_reward = healthy_reward
        self._reset_noise_scale = reset_noise_scale
        self._termination_height = termination_height
        self._target_height = target_height

        # --- swing-up task extensions (opt-in; default False keeps the
        #     upstream balance-only task byte-for-byte) ---
        # swingup=True changes three things:
        #   1. reset: hanging chain (50%) or uniform angles in
        #      +-reset_angle_limit (50%), instead of near-upright noise
        #   2. termination: only once the tip has been above
        #      _upright_threshold in THIS episode may it fall and terminate
        #   3. reward: alive bonus gated on tip being up, plus an energy
        #      shaping term that pays for pumping toward the upright energy
        self._swingup = bool(swingup)
        self._reset_angle_limit = reset_angle_limit
        self._upright_threshold = 1.7
        self._ever_upright = False
        self._energy_weight = float(energy_weight) if swingup else 0.0
        self._dist_weight = float(dist_weight) if swingup else 1.0
        self._near_prob = float(near_prob)
        self._near_width = float(near_width)
        self._capture_reward = bool(capture_reward)
        self._capture_zone = float(capture_zone)
        self._capture_h = float(capture_h)
        self._capture_vel_w = float(capture_vel_w)
        self._near_vel_std = float(near_vel_std)
        self._energy_target: Optional[float] = None

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

        if self._swingup:
            # data.energy is only populated when this flag is on
            self.model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
            # upright energy (qvel=0) is the pumping target; init_qpos is the
            # fully upright configuration for this model
            self._energy_target = self._system_energy(self.init_qpos, np.zeros(self.model.nv))

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        x, _, y = self.data.site_xpos[0]
        observation = self._get_obs()

        if self._swingup:
            # A swing-up episode may only terminate after the tip has been up
            # in THIS episode; otherwise hanging starts die on step 1.
            if y > self._upright_threshold:
                self._ever_upright = True
            terminated = bool(self._ever_upright and y <= self._termination_height)
        else:
            terminated = bool(y <= self._termination_height)

        reward, reward_info = self._get_rew(x, y, terminated)

        if self.render_mode == "human":
            self.render()

        # truncation handled by TimeLimit wrapper from gym.make / registration
        return observation, reward, terminated, False, reward_info

    def _read_energy(self) -> float:
        """Mechanical energy of the PENDULUM subsystem (cart KE excluded).

        KE/PE come from MuJoCo's energy flag; the cart's translational KE is
        subtracted because cart motion is not usable pump energy and would
        otherwise let the policy fake the energy target by shaking the cart.
        """
        ke = float(self.data.energy[0])
        pe = float(self.data.energy[1])
        m_cart = float(self.model.body_mass[1])  # body 0 is the world
        ke -= 0.5 * m_cart * float(self.data.qvel[0]) ** 2
        return ke + pe

    def _system_energy(self, qpos, qvel) -> float:
        state = (self.data.qpos.copy(), self.data.qvel.copy())
        self.set_state(np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64))
        mujoco.mj_forward(self.model, self.data)
        e = self._read_energy()
        self.set_state(*state)
        mujoco.mj_forward(self.model, self.data)
        return e

    def _get_rew(self, x: float, y: float, terminated: bool):
        # qvel: [cart, hinge, hinge2, hinge3]
        v1, v2, v3 = self.data.qvel[1:4]
        dist_penalty = 0.01 * x**2 + (y - self._target_height) ** 2
        # Mirror double-pendulum weighting; add stronger weight on the top link
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2 + 1e-2 * v3**2

        if self._swingup:
            # No free alive bonus while hanging: "alive" means tip still up.
            alive_bonus = self._healthy_reward * float(y > self._termination_height)
            dist = self._dist_weight * dist_penalty

            if self._capture_reward and y > self._capture_zone:
                # CAPTURE REGION (tip above capture_zone): the energy term is
                # switched OFF here.  Pumping to the separatrix energy means an
                # overshooting arrival carries kinetic energy, and with the
                # energy term active the policy can fly straight through the
                # top at a cost of only ~0.01.  Inside the capture zone the
                # objective must instead say, unambiguously: be high, be slow.
                energy_penalty = 0.0
                posture = self._capture_vel_w * (v1**2 + v2**2 + v3**2)
                capture_bonus = self._capture_h * (y - self._capture_zone)
                reward = float(
                    alive_bonus + capture_bonus - dist - vel_penalty - posture
                )
                reward_info = {
                    "reward_survive": alive_bonus,
                    "distance_penalty": -dist,
                    "velocity_penalty": -vel_penalty - posture,
                    "energy_penalty": 0.0,
                    "energy": self._read_energy(),
                    "tip_x": float(x),
                    "tip_y": float(y),
                }
                return reward, reward_info

            energy_penalty = self._energy_weight * (
                (self._read_energy() - self._energy_target) / self._energy_target
            ) ** 2
            reward = float(
                alive_bonus
                - dist
                - vel_penalty
                - energy_penalty
            )
            reward_info = {
                "reward_survive": alive_bonus,
                "distance_penalty": -dist,
                "velocity_penalty": -vel_penalty,
                "energy_penalty": -energy_penalty,
                "energy": self._read_energy(),
                "tip_x": float(x),
                "tip_y": float(y),
            }
            return reward, reward_info

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
        self._ever_upright = False

        if self._swingup and self._reset_angle_limit is not None:
            # Swing-up curriculum.  near_prob of the episodes start near
            # upright so the "catch and hold" behaviour gets frequent
            # practice; the rest split between the hanging chain and large
            # random angles.  This is the distribution the task asks for
            # ("自然下垂或者大角度随机状态").
            qpos = self.init_qpos.copy()
            qvel = self.init_qvel.copy()
            qpos[0] = self.np_random.uniform(-0.1, 0.1)
            r = self.np_random.random()
            p_hang = 0.5 * (1.0 - self._near_prob)
            if r < self._near_prob:
                # 近直立：捕获/保持的主要练习来源
                qpos[1:] = self.np_random.uniform(-self._near_width, self._near_width, size=3)
            elif r < self._near_prob + p_hang:
                # 自然下垂：第一根转 180°，后两根与它串联向下
                qpos[1] = np.pi + self.np_random.uniform(-0.15, 0.15)
                qpos[2] = self.np_random.uniform(-0.15, 0.15)
                qpos[3] = self.np_random.uniform(-0.15, 0.15)
            else:
                qpos[1:] = self.np_random.uniform(
                    -self._reset_angle_limit, self._reset_angle_limit, size=3
                )
            qvel[:] = self.np_random.normal(0.0, self._near_vel_std, size=self.model.nv)
            self.set_state(qpos, qvel)
            return self._get_obs()

        # Upstream behaviour: near-upright noise, honouring reset_noise_scale.
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
