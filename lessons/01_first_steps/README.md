# Lesson 1: First Steps (the RL loop)

**Goal:** train your first robot controller in under a minute and understand
every piece of what happened.

**Robot:** the MuJoCo *inverted pendulum*: a cart on a rail with a pole on
top. Push the cart left and right to keep the pole upright.

## The 5 words you need

You already know supervised learning. RL differs in one way: there are no
labels. The robot tries things and gets a score.

| Term            | Meaning                                                  | Pendulum example                  |
|-----------------|----------------------------------------------------------|-----------------------------------|
| **Observation** | What the robot senses each step                          | cart pos, pole angle, velocities  |
| **Action**      | What the robot does each step                            | a force from -3 to +3             |
| **Reward**      | A number saying "that was good" after each step          | +1 for every step the pole is up  |
| **Episode**     | One attempt from `reset()` until failure or time limit   | ends when the pole tips > 0.2 rad |
| **Policy**      | The neural network mapping observation → action          | a small MLP (64×64)               |

Training searches for the policy weights that **maximize total reward per
episode**. That's all RL is. The rest is making that search work.

## Step 1: Explore the environment

```bash
python lessons/01_first_steps/explore_env.py
```

Random actions keep the pole up for about **5 steps** out of a possible 1000.

Note `terminated` vs `truncated`. *Terminated* means the robot failed.
*Truncated* means the clock ran out. PPO treats them differently: a timeout
isn't the robot's fault. Isaac Lab makes the same distinction (`terminated`
vs `time_outs`).

## Step 2: Train with PPO

```bash
python lessons/01_first_steps/train_ppo.py
```

Takes about 30 seconds on 4 CPU cores. What happens:

1. **Collect.** 4 parallel sims each run the current policy for 1024 steps.
2. **Learn.** PPO nudges the network toward actions that led to more reward
   than expected. It caps how far each update can move ("proximal"), which
   keeps training stable.
3. **Repeat** until 50,000 steps.

Watch the `ep_rew_mean` column climb. Expected result: **mean reward 1000.0**,
a perfect score.

Watch the curves live in a second terminal:
```bash
tensorboard --logdir runs
```

## Step 3: Watch it

```bash
python lessons/01_first_steps/watch.py            # trained
python lessons/01_first_steps/watch.py --random   # untrained, for comparison
```

Videos go to `videos/`. On a machine without a display, prefix the command
with `MUJOCO_GL=osmesa`.

## Exercises (do them in order)

1. **Break it.** Train with `--steps 5000`. What reward do you get? Where in
   `ep_rew_mean` did it stop?
2. **Parallelism.** Train with `--n-envs 1` and then `--n-envs 8`. Compare
   wall-clock time and final reward. (This trade-off is why Isaac Lab runs
   4096 envs on a GPU.)
3. **Harder robot.** Run `--env InvertedDoublePendulum-v5 --steps 200000`.
   Two poles stacked on top of each other. Does 50k steps still work?
4. **Read the code.** In `train_ppo.py`, change `gamma` to `0.5`. The robot
   now barely cares about the future. Predict what happens before you run it.

## Checkpoint questions

Answer these before Lesson 2. Tell me your answers and I'll check them.

1. Why did the random policy score about 5 while the trained one scored 1000?
   What is the reward per step?
2. What's the difference between `terminated` and `truncated`?
3. Why do we run several environments in parallel?
4. In supervised learning you have labels. What plays that role in RL?
