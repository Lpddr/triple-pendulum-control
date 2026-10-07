from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

import triple_pendulum
from evaluation.viz import load_agent


# 创建环境
env = gym.make(
    "InvertedTriplePendulum-v0",
    render_mode="human",
    termination_height=-999
)


obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]


# 加载 Stage10 模型
policy = load_agent(
    Path("sac_stage10_final.pt"),
    obs_dim,
    act_dim
)


# 创建初始状态
obs, _ = env.reset(seed=1)

qpos = env.unwrapped.init_qpos.copy()
qvel = env.unwrapped.init_qvel.copy()


# 测试 ±10°
alpha = np.deg2rad(
    [
        10.0,
        -8.0,
        10.0
    ]
)


# 小车位置
qpos[0] = 0.0


# 转换成关节角
qpos[1] = alpha[0]
qpos[2] = alpha[1] - alpha[0]
qpos[3] = alpha[2] - alpha[1]


# 初速度
qvel[:] = 0.0


env.unwrapped.set_state(
    qpos,
    qvel
)


obs = env.unwrapped._get_obs()


env.render()


dt = env.unwrapped.dt

next_time = time.perf_counter()


print("测试：±10°恢复能力")


max_steps = 1000

success_count = 0


for step in range(max_steps):

    with torch.inference_mode():

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32
        ).unsqueeze(0)


        action = policy(
            obs_tensor
        ).squeeze(0).numpy()


    obs, reward, terminated, truncated, info = env.step(
        np.clip(
            action,
            -1.0,
            1.0
        )
    )


    tip_y = info["tip_y"]


    # 每100步打印一次高度
    if step % 100 == 0:
        print(
            f"step={step} "
            f"time={step*dt:.2f}s "
            f"tip_y={tip_y:.3f}"
        )


    # 判断是否保持直立
    if tip_y > 1.6:
        success_count += 1
    else:
        success_count = 0


    # 连续10秒稳定
    if success_count * dt >= 10:

        print(
            "PASS: 稳定保持10秒"
        )

        break


    # 原速播放
    next_time += dt

    sleep_time = (
        next_time
        - time.perf_counter()
    )

    if sleep_time > 0:
        time.sleep(sleep_time)

    else:
        next_time = time.perf_counter()


else:

    print(
        "FAIL: 没有稳定保持10秒"
    )


env.close()