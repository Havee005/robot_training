# Robot Training in Simulation

A hands-on path from "what is reinforcement learning?" to training walking and
mobile robots, ending in NVIDIA Isaac Lab.

## Why we don't start in Isaac Lab

Isaac Lab needs an NVIDIA RTX GPU. We start in **MuJoCo** (runs on any CPU)
because the ideas carry over one-to-one:

| Concept            | MuJoCo + Gymnasium (now)    | Isaac Lab (later)                   |
|--------------------|-----------------------------|-------------------------------------|
| Environment        | `gym.make("Ant-v5")`        | `ManagerBasedRLEnv` / `DirectRLEnv` |
| Observation/action | `obs_space`, `action_space` | same idea, batched on GPU           |
| Reward             | `step()` returns a reward   | reward terms in a config            |
| Algorithm          | PPO (Stable-Baselines3)     | PPO (RSL-RL / skrl)                 |
| Parallelism        | 4–16 CPU envs               | 4096 GPU envs                       |

When you get to Isaac Lab you'll only be learning a new API. The RL concepts
will already be familiar.

## Roadmap

| #  | Lesson                           | Robot                | You learn                                       |
|----|----------------------------------|----------------------|-------------------------------------------------|
| 1  | First steps                      | Inverted pendulum    | The env loop, spaces, training PPO, evaluating  |
| 2  | Reading training curves          | Pendulum / Reacher   | TensorBoard, reward, entropy, hyperparameters   |
| 3  | Reward design                    | Reacher arm          | Shaping rewards, why agents "cheat"             |
| 4  | Build your own environment       | Custom 2D robot      | Writing a Gymnasium env from scratch            |
| 5  | Mobile robot                     | Differential drive   | Navigation, goals, obstacles                    |
| 6  | Legged locomotion                | Ant (quadruped)      | Continuous control at scale, parallel envs      |
| 7  | Walking humanoid                 | Humanoid             | Hard exploration, longer training               |
| 8  | Isaac Lab                        | ANYmal / Go2         | Cloud GPU setup, Isaac Lab tasks, RSL-RL        |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On a headless machine (no display), set `MUJOCO_GL=osmesa` (or `egl` if you
have a GPU) so videos can be rendered.

Start with [`lessons/01_first_steps`](lessons/01_first_steps/README.md).
