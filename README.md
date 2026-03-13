# KuzongaEnv

A custom Gymnasium-compatible environment for the [Kuzonga game](https://www.kuzonga.com).

## Environment Details
### Action Space

The environment uses a dictionary action space with three components:

| Key       | Value       | Description |
|-----------|------------|-------------|
| v  | 1 or 0 (or True and False, respectively)     | Whether to attempt division (`1`) or change a digit (`0`). |
| g     | 0–9        | If `v=1`, the divisor; if `v=0`, the new digit to set at `r`. |
| r     | 0…digits-1 or `None` | Rindex (Right-to-left or reverse index) of the digit to overwrite (if `v=1`, it should be `None`). |


Example:

```python
action = {"v": 1, "g": 3, "r": None}  # attempt division by 3
action = {"v": 0, "g": 7, "r": 1}  # set the second digit (from the right) to 7
```

### Observation Space

The environment uses a dictionary observation space with the following keys:

| Key                        | Type                     | Description |
|----------------------------|-------------------------|-------------|
| s             | np.int8 array (digits,) | The original number as an array of digits. |
| d             | np.int8 array (digits,) | The current number as an array of digits. |
| a | np.int64 array (digits*10,) | Binary mask of which digits can be set at each position. Flattened from shape (digits, 10). |
| p                    | np.int64 array (num_players*3,) | Each player’s `[i, c, m]`, where `i` is the ID, `c` is the score and `m` tells if it is the player's turn (`m=1`) or not (`m=0`). Flattened array of all players. It has one player by default. |
| t                | int                      | ID of the player whose turn it is. |


Example:

```python
obs, info = env.reset()
print(obs["s"])   # [1, 7]
print(obs["d"])   # [4, 7]
print(obs["a"])   # array([1,1,0,...])
print(obs["p"])   # array([0,0,1,1,0,0])  # two players
print(obs["t"])   # 0
```

### Quick Notes

The *a* mask ensures illegal moves (e.g., setting a leading zero or creating number 0/1) are prevented.

Rewards and penalties are automatically updated in the environment during *step()*.

The environment fully supports multiple players, and tracks turns via *t* and *m*.

The *options* parameter in *reset()* allows resetting the environment to a specific given state/obs, by setting it with the format: `options = {'obs': <state/obs dict>}`

The `info` variable value from methods `reset()` and `step()` has the fields:
- `obs_decoded`, which has the plain values of the resulting state/observation. So it can be retrieved as `info['obs_decoded']`
- `all_actions`, which is a list of all the actions in the current state. So it can be retrieved as `info['all_actions']`


## Usage Example

```python
import gymnasium as gym
import kuzongaenv

env = gym.make("Kuzonga-v0")
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

If you use **Kuzonga** in your research, projects, or publications, please cite it as:

Jacinto Jeje Matamba Quimua (2025). KuzongaEnv: Gym Environment for Reinforcement Learning Experiments. GitHub repository: https://github.com/jaci-hub/kuzongaenv


### BibTeX

```bibtex
@misc{kuzongaenv2025,
  author       = {Jacinto Jeje Matamba Quimua},
  title        = {KuzongaEnv: Gym Environment for Reinforcement Learning Experiments},
  year         = 2025,
  howpublished = {\url{https://github.com/jaci-hub/kuzongaenv}},
}
```

## Play Kuzonga Online

[Kuzonga game](https://www.kuzonga.com)
