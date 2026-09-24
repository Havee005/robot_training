"""Lesson 1, step 3: record a video of the trained policy.

Run:  python lessons/01_first_steps/watch.py
Headless machine:  MUJOCO_GL=osmesa python lessons/01_first_steps/watch.py
Add --random to see what an untrained robot does, for comparison.
"""

import argparse
import os

import gymnasium as gym
import imageio
from stable_baselines3 import PPO

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="InvertedPendulum-v5")
parser.add_argument("--random", action="store_true", help="use random actions")
parser.add_argument("--max-steps", type=int, default=500)
args = parser.parse_args()

# Load the model before creating the renderer: on some headless setups PyTorch
# crashes if it finishes importing after the OpenGL library is loaded.
model = None if args.random else PPO.load(f"models/ppo_{args.env}")
env = gym.make(args.env, render_mode="rgb_array")

obs, _ = env.reset(seed=42)
frames = [env.render()]
for _ in range(args.max_steps):
    if model is None:
        action = env.action_space.sample()
    else:
        action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        break

os.makedirs("videos", exist_ok=True)
tag = "random" if args.random else "trained"
out = f"videos/{args.env}_{tag}.mp4"
imageio.mimsave(out, frames, fps=env.metadata.get("render_fps", 30))
print(f"Wrote {len(frames)} frames to {out}")
