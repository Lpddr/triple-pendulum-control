#!/usr/bin/env python3
"""Run a random agent on InvertedTriplePendulum-v0 (human window or rgb dump).

In human mode the window stays open until you close it (or press Esc).
Episodes auto-reset when the pendulum falls.
"""

from __future__ import annotations

import argparse
import itertools
import time

import gymnasium as gym
import numpy as np

import triple_pendulum  # noqa: F401


def viewer_closed(env) -> bool:
    """True if the MuJoCo human window was closed or is no longer usable."""
    if getattr(env, "render_mode", None) != "human":
        return False
    try:
        renderer = env.unwrapped.mujoco_renderer
        viewer = getattr(renderer, "viewer", None)
        if viewer is None:
            return False  # window not created until first render
        window = getattr(viewer, "window", None)
        if window is None:
            return True
        import glfw

        # Poll so the OS close event is processed even between steps
        glfw.poll_events()
        return bool(glfw.window_should_close(window))
    except Exception:
        # Destroyed/invalid window after Esc or close → treat as closed
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo inverted triple pendulum")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "Max episodes. Default: unlimited in human mode (until window close), "
            "3 in rgb_array mode."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--render-mode",
        choices=("human", "rgb_array"),
        default="human",
        help="human opens a live window; rgb_array only prints stats",
    )
    parser.add_argument(
        "--save-frame",
        type=str,
        default="",
        help="If set with render_mode=rgb_array, save one PNG path",
    )
    args = parser.parse_args()

    # Human: keep going until the window is closed (was exiting after ~3 short eps)
    if args.episodes is None:
        max_episodes = None if args.render_mode == "human" else 3
    else:
        max_episodes = args.episodes

    env = gym.make("InvertedTriplePendulum-v0", render_mode=args.render_mode)
    print(f"obs space:  {env.observation_space}")
    print(f"act space:  {env.action_space}")
    print(f"dt:         {env.unwrapped.dt:.4f}s  fps≈{env.metadata.get('render_fps')}")
    if args.render_mode == "human":
        print("Stays open until you close the window (or press Esc).")

    stopped_by_window = False
    episode_iter = (
        itertools.count(1) if max_episodes is None else range(1, max_episodes + 1)
    )

    try:
        for ep in episode_iter:
            if viewer_closed(env):
                stopped_by_window = True
                break

            obs, info = env.reset(seed=args.seed + ep - 1)
            total = 0.0
            steps = 0
            while True:
                if viewer_closed(env):
                    stopped_by_window = True
                    break

                action = env.action_space.sample()
                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                except Exception:
                    # Gymnasium may crash mid-render if the window was just closed
                    stopped_by_window = True
                    break

                total += reward
                steps += 1
                if args.render_mode == "human":
                    time.sleep(0.01)
                if terminated or truncated:
                    break

            if stopped_by_window:
                print(f"window closed during episode {ep} (steps={steps})")
                break

            tip_y = info.get("tip_y", float("nan"))
            print(
                f"episode {ep}: steps={steps:4d}  return={total:8.2f}  "
                f"final tip_y={tip_y:.3f}"
            )

            if args.save_frame and args.render_mode == "rgb_array":
                try:
                    import imageio.v2 as imageio

                    frame = env.render()
                    imageio.imwrite(args.save_frame, frame)
                    print(f"saved frame → {args.save_frame}")
                except Exception as exc:  # pragma: no cover
                    print(f"could not save frame: {exc}")
    finally:
        env.close()

    if stopped_by_window:
        print("stopped: visualization window closed.")
    elif max_episodes is not None:
        print(f"finished {max_episodes} episode(s).")


if __name__ == "__main__":
    main()
