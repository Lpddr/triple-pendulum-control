from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

import triple_pendulum
from evaluation.viz import load_agent

env = gym.make("InvertedTriplePendulum-v0", reset_noise_scale=0.05)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

policy = load_agent(
    Path("sac_triple_final.pt"),
    obs_dim,
    act_dim
)

for ep in range(20):
    obs, _ = env.reset(seed=ep)
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        with torch.inference_mode():
            obs_t = torch.as_tensor(
                obs,
                dtype=torch.float32
            ).unsqueeze(0)

            action = policy(obs_t).squeeze(0).numpy()

        obs, reward, terminated, truncated, info = env.step(
            np.clip(action, -1.0, 1.0)
        )

        steps += 1

    seconds = steps * env.unwrapped.dt
    result = "PASS" if steps >= 1000 else "FAIL"

    print(
        f"Episode {ep + 1:02d}: "
        f"{steps:4d} steps | "
        f"{seconds:5.2f} s | "
        f"{result}"
    )

env.close()