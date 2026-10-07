from stable_baselines3 import PPO
import gymnasium as gym
import triple_pendulum

env = gym.make("InvertedTriplePendulum-v0", render_mode="human")

model = PPO.load("models/ppo_triple_pendulum.zip")

obs, info = env.reset()

while True:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()