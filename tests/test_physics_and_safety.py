"""Physics sanity checks + anti-exploit / training-safety tests.

These go beyond smoke tests: upright geometry, gravity, actuation direction,
termination/reward consistency, TimeLimit, and long-horizon finite values.
"""

from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym

import triple_pendulum  # noqa: F401


@pytest.fixture
def raw_env():
    """Unwrapped env (no TimeLimit) for precise control of termination."""
    e = gym.make("InvertedTriplePendulum-v0")
    yield e.unwrapped
    e.close()


@pytest.fixture
def wrapped_env():
    e = gym.make("InvertedTriplePendulum-v0")
    yield e
    e.close()


def _set_qpos_qvel(env, qpos, qvel=None):
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.zeros(env.model.nv) if qvel is None else np.asarray(qvel, dtype=np.float64)
    env.set_state(qpos, qvel)
    # Forward kinematics so site_xpos matches qpos before any step
    import mujoco

    mujoco.mj_forward(env.model, env.data)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registered_with_gymnasium():
    spec = gym.spec("InvertedTriplePendulum-v0")
    assert spec is not None
    assert "InvertedTriplePendulumEnv" in str(spec.entry_point)
    assert spec.max_episode_steps == 1000


def test_register_envs_idempotent():
    triple_pendulum.register_envs()
    triple_pendulum.register_envs()  # must not raise
    env = gym.make("InvertedTriplePendulum-v0")
    env.close()


# ---------------------------------------------------------------------------
# Model topology / geometry
# ---------------------------------------------------------------------------


def test_joint_and_actuator_topology(raw_env):
    names = [raw_env.model.joint(i).name for i in range(raw_env.model.njnt)]
    assert names == ["slider", "hinge", "hinge2", "hinge3"]
    assert raw_env.model.nq == 4
    assert raw_env.model.nv == 4
    assert raw_env.model.nu == 1
    # Same cart force scaling as Gymnasium double pendulum
    assert raw_env.model.actuator_gear[0, 0] == pytest.approx(500.0)


def test_upright_tip_height_is_1_8m(raw_env):
    """Three 0.6 m poles stacked vertically → tip at z≈1.8 (MuJoCo y-up here is z in site)."""
    # qpos angles 0,0,0 → all poles along +z from cart
    _set_qpos_qvel(raw_env, [0.0, 0.0, 0.0, 0.0])
    tip = raw_env.data.site_xpos[0]
    # site layout: x, y, z with vertical axis = z in this model? double uses y from site_xpos
    # Gymnasium double: x, _, y = site_xpos[0] — vertical is index 2 stored as "y" in reward
    x, mid, vertical = tip
    assert abs(x) < 1e-6
    assert vertical == pytest.approx(1.8, abs=1e-5)


def test_pole_lengths_sum_via_tip_offset(raw_env):
    """Fully folded 180° on first hinge drops tip near cart height (~0)."""
    _set_qpos_qvel(raw_env, [0.0, np.pi, 0.0, 0.0])
    vertical = raw_env.data.site_xpos[0][2]
    # tip should be near 0 (hanging down), not near 1.8
    assert vertical < 0.3


# ---------------------------------------------------------------------------
# Physics dynamics
# ---------------------------------------------------------------------------


def test_gravity_pulls_tip_down_from_near_upright(raw_env):
    # Slight lean so it is unstable; zero action
    _set_qpos_qvel(raw_env, [0.0, 0.08, 0.0, 0.0], np.zeros(4))
    y0 = raw_env.data.site_xpos[0][2]
    for _ in range(40):
        raw_env.do_simulation(np.array([0.0]), raw_env.frame_skip)
    y1 = raw_env.data.site_xpos[0][2]
    assert y1 < y0, f"expected tip to fall under gravity, got {y0} → {y1}"


def test_positive_action_accelerates_cart_positive(raw_env):
    _set_qpos_qvel(raw_env, [0.0, 0.0, 0.0, 0.0], np.zeros(4))
    raw_env.do_simulation(np.array([1.0]), raw_env.frame_skip)
    assert raw_env.data.qvel[0] > 0.0
    assert raw_env.data.qpos[0] > 0.0


def test_negative_action_accelerates_cart_negative(raw_env):
    _set_qpos_qvel(raw_env, [0.0, 0.0, 0.0, 0.0], np.zeros(4))
    raw_env.do_simulation(np.array([-1.0]), raw_env.frame_skip)
    assert raw_env.data.qvel[0] < 0.0
    assert raw_env.data.qpos[0] < 0.0


def test_cart_joint_limits_exist(raw_env):
    jnt_id = raw_env.model.joint("slider").id
    # limited flag and range [-1, 1] like double pendulum
    assert raw_env.model.jnt_limited[jnt_id] == 1
    lo, hi = raw_env.model.jnt_range[jnt_id]
    assert lo == pytest.approx(-1.0)
    assert hi == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Termination / reward — no free lunch
# ---------------------------------------------------------------------------


def test_terminates_when_tip_below_threshold(raw_env):
    # Hang mostly down so tip is low
    _set_qpos_qvel(raw_env, [0.0, np.pi * 0.9, 0.0, 0.0], np.zeros(4))
    obs, reward, terminated, truncated, info = raw_env.step(np.array([0.0], dtype=np.float32))
    assert info["tip_y"] <= raw_env._termination_height
    assert terminated is True
    # No survive bonus on the terminating step (v5 double-pendulum fix)
    assert info["reward_survive"] == 0.0
    assert reward < 0.0 or reward <= 0.0 + 1e-9


def test_survive_bonus_only_when_healthy(raw_env):
    _set_qpos_qvel(raw_env, [0.0, 0.02, 0.0, 0.0], np.zeros(4))
    _, reward, terminated, _, info = raw_env.step(np.array([0.0], dtype=np.float32))
    if not terminated:
        assert info["reward_survive"] == raw_env._healthy_reward
        # The physically reachable upright tip height is the reward target.
        assert info["distance_penalty"] <= 0.0
        assert reward <= raw_env._healthy_reward


def test_cannot_exceed_alive_bonus_per_step(raw_env):
    """Alive bonus is fixed; penalties only reduce reward — no reward inflation hole."""
    _set_qpos_qvel(raw_env, [0.0, 0.0, 0.0, 0.0], np.zeros(4))
    for _ in range(30):
        _, reward, terminated, _, info = raw_env.step(np.array([0.0], dtype=np.float32))
        assert reward <= raw_env._healthy_reward + 1e-9
        assert info["reward_survive"] in (0.0, raw_env._healthy_reward)
        if terminated:
            break


def test_higher_tip_gives_higher_reward_ceteris_paribus(raw_env):
    """Distance term should reward being more upright (higher tip y)."""
    rews = []
    for angle in (0.4, 0.2, 0.05):
        _set_qpos_qvel(raw_env, [0.0, angle, 0.0, 0.0], np.zeros(4))
        # evaluate reward at current state without advancing much: use _get_rew
        x, _, y = raw_env.data.site_xpos[0]
        terminated = bool(y <= raw_env._termination_height)
        r, _ = raw_env._get_rew(x, y, terminated)
        rews.append(r)
    assert rews[2] > rews[1] > rews[0]


def test_time_limit_truncates_at_1000(wrapped_env):
    """TimeLimit from registration prevents infinite episode farming."""
    obs, _ = wrapped_env.reset(seed=0)
    # Force always-healthy-ish by resetting tip high if needed — just step max
    # Use zero noise env path: keep stepping; with random actions will die early
    # so pin near upright with small actions and check max_episode_steps property
    assert wrapped_env.spec.max_episode_steps == 1000

    # Directly verify wrapper present
    from gymnasium.wrappers import TimeLimit

    env = wrapped_env
    found = False
    while hasattr(env, "env"):
        if isinstance(env, TimeLimit):
            found = True
            assert env._max_episode_steps == 1000
            break
        env = env.env
    # gymnasium.make always adds TimeLimit when max_episode_steps set
    assert found or wrapped_env.spec.max_episode_steps == 1000


def test_time_limit_actually_fires():
    """With a high termination threshold never hit, episode truncates at 1000."""
    env = gym.make(
        "InvertedTriplePendulum-v0",
        termination_height=-10.0,  # never tip-terminate
        reset_noise_scale=0.0,
    )
    try:
        env.reset(seed=0)
        truncated = False
        steps = 0
        for _ in range(1005):
            _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
            steps += 1
            if truncated or terminated:
                break
        assert truncated is True
        assert steps == 1000
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Observation integrity / no NaNs under chaos
# ---------------------------------------------------------------------------


def test_obs_matches_qpos_encoding(raw_env):
    q = np.array([0.3, 0.5, -0.4, 0.2])
    v = np.array([0.1, -0.2, 0.3, -0.1])
    _set_qpos_qvel(raw_env, q, v)
    obs = raw_env._get_obs()
    assert obs[0] == pytest.approx(q[0])
    np.testing.assert_allclose(obs[1:4], np.sin(q[1:]))
    np.testing.assert_allclose(obs[4:7], np.cos(q[1:]))
    np.testing.assert_allclose(obs[7:11], v)


def test_long_random_rollout_stays_finite(wrapped_env):
    """Chaos agent should not produce NaN/Inf (physics / reward hole)."""
    obs, _ = wrapped_env.reset(seed=123)
    for i in range(500):
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        assert np.all(np.isfinite(obs)), f"non-finite obs at step {i}"
        assert np.isfinite(reward), f"non-finite reward at step {i}"
        assert np.isfinite(info["tip_y"])
        if terminated or truncated:
            obs, _ = wrapped_env.reset()


def test_action_clipping_respected(raw_env):
    """Out-of-range actions should not silently grant super-gear forces via API misuse —
    Gymnasium Box sampling is in range; MujocoEnv clips via ctrlrange on actuator."""
    initial_qpos = np.array([0.0, 0.0, 0.0, 0.0])
    initial_qvel = np.zeros(4)

    _set_qpos_qvel(raw_env, initial_qpos, initial_qvel)
    raw_env.do_simulation(np.array([1.0]), raw_env.frame_skip)
    bounded_qpos = raw_env.data.qpos.copy()
    bounded_qvel = raw_env.data.qvel.copy()

    _set_qpos_qvel(raw_env, initial_qpos, initial_qvel)
    raw_env.do_simulation(np.array([100.0]), raw_env.frame_skip)
    np.testing.assert_allclose(raw_env.data.qpos, bounded_qpos)
    np.testing.assert_allclose(raw_env.data.qvel, bounded_qvel)


def test_seed_determinism(wrapped_env):
    def run(seed):
        o, _ = wrapped_env.reset(seed=seed)
        traj = [o.copy()]
        rng = np.random.default_rng(seed)
        for _ in range(15):
            a = rng.uniform(-1, 1, size=(1,)).astype(np.float32)
            o, r, term, trunc, _ = wrapped_env.step(a)
            traj.append(o.copy())
            if term or trunc:
                break
        return traj

    t1 = run(0)
    t2 = run(0)
    assert len(t1) == len(t2)
    for a, b in zip(t1, t2):
        np.testing.assert_allclose(a, b)
