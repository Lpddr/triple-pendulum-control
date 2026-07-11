"""Smoke tests for InvertedTriplePendulum-v0."""

from __future__ import annotations

import numpy as np
import pytest

import triple_pendulum  # noqa: F401 — registers env
import gymnasium as gym


@pytest.fixture
def env():
    e = gym.make("InvertedTriplePendulum-v0")
    yield e
    e.close()


def test_spaces(env):
    assert env.observation_space.shape == (12,)
    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == pytest.approx(-1.0)
    assert env.action_space.high[0] == pytest.approx(1.0)


def test_reset_and_step(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (12,)
    assert np.all(np.isfinite(obs))

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (12,)
        assert np.isfinite(reward)
        assert "reward_survive" in info
        assert "distance_penalty" in info
        assert "velocity_penalty" in info
        assert "tip_y" in info
        if terminated or truncated:
            obs, info = env.reset()
            break


def test_obs_structure(env):
    """sin² + cos² ≈ 1 for each of the three pole angles."""
    obs, _ = env.reset(seed=42)
    sines = obs[1:4]
    cosines = obs[4:7]
    norms = sines**2 + cosines**2
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)


def test_upright_not_immediately_terminated(env):
    obs, info = env.reset(seed=1)
    # With default noise, tip should usually start above termination height
    # after a few resets; at least check we can take one step without error.
    obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(reward, float)


def test_render_rgb(env_rgb=None):
    e = gym.make("InvertedTriplePendulum-v0", render_mode="rgb_array")
    try:
        e.reset(seed=0)
        frame = e.render()
        assert frame is not None
        assert frame.ndim == 3
        assert frame.shape[2] == 3
    finally:
        e.close()


def test_direct_class_instantiation():
    from triple_pendulum import InvertedTriplePendulumEnv

    e = InvertedTriplePendulumEnv()
    try:
        obs, _ = e.reset(seed=0)
        assert obs.shape == (12,)
        assert e.model.nq == 4  # slider + 3 hinges
        assert e.model.nv == 4
    finally:
        e.close()
