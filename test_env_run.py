import divide21env 
import gymnasium as gym
from gymnasium.utils.env_checker import check_env


# Test the environment registration
env = gym.make(
    "Divide21-v0",
    render_mode="human",
    auto_render=True
)


if __name__ == "__main__":
    obs, info = env.reset()
    # test five examples
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        print(info)
        if done or trunc:
            break
