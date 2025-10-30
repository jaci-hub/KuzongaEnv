# Divide21Env

A custom Gymnasium-compatible environment for the Divide21 game.

## Environment Details
### Action Space

The environment uses a dictionary action space with three components:

| Key       | Type       | Description |
|-----------|------------|-------------|
| division  | 0 or 1     | Whether to attempt division (`1`) or change a digit (`0`). |
| digit     | 0–9        | If `division=1`, the divisor; if `division=0`, the new digit to set at `index`. |
| index     | 0…digits-1 | Index of the digit to overwrite (ignored if `division=1`). |


Example:

```python
action = {"division": 1, "digit": 3, "index": 0}  # attempt division by 3
action = {"division": 0, "digit": 7, "index": 1}  # set the second digit to 7
```

### Observation Space

The environment uses a dictionary observation space with the following keys:

| Key                        | Type                     | Description |
|----------------------------|-------------------------|-------------|
| dynamic_number             | np.int8 array (digits,) | The current number as an array of digits. |
| available_digits_per_index | np.int64 array (digits*10,) | Binary mask of which digits can be set at each position. Flattened from shape (digits, 10). |
| players                    | np.int64 array (num_players*3,) | Each player’s `[id, score, is_current_turn]`. Flattened array of all players. |
| player_turn                | int                      | ID of the player whose turn it is. |


Example:

```python
obs, info = env.reset()
print(obs["dynamic_number"])            # [4, 7]
print(obs["available_digits_per_index"])  # array([1,1,0,...])
print(obs["players"])                   # array([0,0,1,1,0,0])  # two players
print(obs["player_turn"])               # 0
```

### Quick Notes

The *available_digits_per_index* mask ensures illegal moves (e.g., setting a leading zero or creating number 0/1) are prevented.

Rewards and penalties are automatically updated in the environment during *step()*.

The environment fully supports multiple players, and tracks turns via *player_turn* and *is_current_turn*.


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

## Installation

```bash
pip install -e .
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

