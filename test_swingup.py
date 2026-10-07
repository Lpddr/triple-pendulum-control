from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

import triple_pendulum
from evaluation.viz import load_agent


# 创建环境：测试过程中不因为倒下自动reset
env = gym.make(
    "InvertedTriplePendulum-v0",
    render_mode="human",
    termination_height=-999
)


obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]


# 加载GitHub作者提供的SAC模型
policy = load_agent(
    Path("sac_triple_final.pt"),
    obs_dim,
    act_dim
)


print("模型：sac_triple_final.pt")


# 初始化环境
obs, _ = env.reset(seed=0)


# 设置自然下垂初始状态
qpos = env.unwrapped.init_qpos.copy()
qvel = env.unwrapped.init_qvel.copy()


# 小车位置
qpos[0] = 0.0


# 三根杆自然下垂
qpos[1] = np.pi
qpos[2] = 0.0
qpos[3] = 0.0


# 初速度
qvel[:] = 0.0


env.unwrapped.set_state(
    qpos,
    qvel
)


obs = env.unwrapped._get_obs()


env.render()


dt = env.unwrapped.dt


# 连续稳定10秒
stable_required = int(
    10.0 / dt
)


stable_steps = 0
success = False


next_time = time.perf_counter()


print(
    "开始测试：自然下垂 → 自主摆起 → 稳定10秒"
)


for step in range(1000):

    with torch.inference_mode():

        obs_t = torch.as_tensor(
            obs,
            dtype=torch.float32
        ).unsqueeze(0)


        action = policy(
            obs_t
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


    # 判断是否直立
    if tip_y >= 1.70:
        stable_steps += 1
    else:
        stable_steps = 0


    # 连续稳定10秒
    if stable_steps >= stable_required:

        print(
            f"PASS：已经摆起并连续稳定10秒！"
            f" 当前时间={step*dt:.2f}s"
        )

        success = True
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


if not success:

    print(
        "FAIL：50秒内没有完成摆起并连续稳定10秒。"
    )


env.close()