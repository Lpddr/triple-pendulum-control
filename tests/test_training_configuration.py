"""Regression checks for previously ineffective experiment arguments."""
import numpy as np
import torch

from algorithms.sac import sac_train
from tools.eval_policy import evaluate, sample_start
import gymnasium as gym
import triple_pendulum


def test_sac_factory_applies_capture_and_dynamics(monkeypatch):
    for name, value in dict(capture_reward=True, capture_vel_w=.123,
                            near_vel_std=.7, joint_damping=.2,
                            frame_skip=2, max_episode_steps=40).items():
        monkeypatch.setattr(sac_train.config, name, value)
    env = sac_train.make_env()()
    try:
        u = env.unwrapped
        assert u._capture_reward
        assert u._capture_vel_w == .123
        assert u._near_vel_std == .7
        np.testing.assert_allclose(u.model.dof_damping[1:], .2)
        assert u.dt == .02
        assert env.spec.max_episode_steps == 40
    finally:
        env.close()


def test_evaluator_horizon_frequency_and_swingup_termination():
    policy = lambda obs: torch.zeros((len(obs), 1))
    rows = evaluate(policy, 'hanging', 1, True, swingup=True,
                    frame_skip=2, max_episode_steps=5, vel_std=0.)
    assert rows[0]['n'] == 5  # hanging is a valid initial condition
    assert rows[0]['hold_s'] == 0.


def test_initial_velocity_override_is_effective():
    env = gym.make('InvertedTriplePendulum-v0')
    try:
        env.reset(seed=1)
        for kind in ['hanging', 'large', 'mixed', 'swingup']:
            sample_start(kind, env.unwrapped, np.random.default_rng(1), vel_std=0.)
            np.testing.assert_array_equal(env.unwrapped.data.qvel, 0.)
    finally:
        env.close()
