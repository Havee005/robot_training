"""Lesson 1, step 1: meet the environment loop.

Every RL setup (MuJoCo, PyBullet, Isaac Lab) boils down to this loop:

    obs = env.reset()
    while not done:
        action = policy(obs)
        obs, reward, done = env.step(action)

Here the "policy" just picks random actions, so the pole falls fast.
Run:  python lessons/01_first_steps/explore_env.py
"""

import gymnasium as gym
import numpy as np

env = gym.make("InvertedPendulum-v5")

print("Observation space:", env.observation_space)
print("  -> [cart position, pole angle, cart velocity, pole angular velocity]")
print("Action space:     ", env.action_space)
print("  -> force pushed on the cart\n")

episode_lengths = []
for episode in range(5):
    obs, info = env.reset(seed=episode)
    total_reward, steps = 0.0, 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()  # random policy
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    # terminated = the task failed (pole fell); truncated = hit the time limit (1000 steps)
    episode_lengths.append(steps)
    print(f"Episode {episode}: {steps:4d} steps, reward {total_reward:6.1f}, "
          f"{'fell' if terminated else 'time limit'}")

print(f"\nRandom policy survives {np.mean(episode_lengths):.1f} steps on average.")
print("The maximum is 1000. Next: train_ppo.py teaches the robot to do better.")
env.close()
