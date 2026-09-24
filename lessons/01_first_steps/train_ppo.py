"""Lesson 1, step 2: train a balancing policy with PPO.

PPO (Proximal Policy Optimization) is the default algorithm for robot
locomotion. Isaac Lab's legged-robot examples use it too.

Run:  python lessons/01_first_steps/train_ppo.py
Watch training live:  tensorboard --logdir runs
"""

import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="InvertedPendulum-v5")
parser.add_argument("--steps", type=int, default=50_000)
parser.add_argument("--n-envs", type=int, default=4, help="parallel simulations")
args = parser.parse_args()

# Several copies of the sim run side by side to collect experience faster.
# Isaac Lab applies the same idea on a GPU with thousands of copies.
vec_env = make_vec_env(args.env, n_envs=args.n_envs, seed=0)

model = PPO(
    "MlpPolicy",        # a small neural network: obs -> action
    vec_env,
    n_steps=1024,       # steps collected per env before each update
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,         # how much future reward matters
    verbose=1,
    tensorboard_log="runs",
    seed=0,
)

eval_env = gym.make(args.env)
before, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"\nBefore training: mean reward {before:.1f}\n")

model.learn(total_timesteps=args.steps, tb_log_name=args.env)

after, std = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"\nAfter training:  mean reward {after:.1f} +/- {std:.1f}")

path = f"models/ppo_{args.env}"
model.save(path)
print(f"Saved to {path}.zip. Record a video with: python lessons/01_first_steps/watch.py")
