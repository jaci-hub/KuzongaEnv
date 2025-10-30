# Divide21Env

A custom Gymnasium-compatible environment for the Divide21 game.

## Installation

```bash
pip install -e .
```

## Usage Example

```python
import gymnasium as gym
import divide21env

env = gym.make("Divide21-v0")
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print(f"Observation: {obs}")
print(f"Reward: {reward}, Terminated: {terminated}")
```

## Cite This Project

If you use **Divide21** in your research, projects, or publications, please cite it as:

Jacinto Jeje Matamba Quimua (2025). Divide21Env: Gym Environment for Reinforcement Learning Experiments. GitHub repository: https://github.com/jaci-hub/divide21Env


### BibTeX

```bibtex
@misc{divide21env2025,
  author       = {Jacinto Jeje Matamba Quimua},
  title        = {Divide21Env: Gym Environment for Reinforcement Learning Experiments},
  year         = 2025,
  howpublished = {\url{https://github.com/jaci-hub/divide21Env}},
}
```

