import divide21env 
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from divide21env.envs.divide21_env import Divide21Env


# Test the environment registration
env = gym.make(
    "Divide21-v0",
    render_mode="human",
    auto_render=True
)

# env = Divide21Env(
#     render_mode="human",
#     auto_render=True
# )

def manual_reset_test(state):
    options = {
        'obs': state
    }
    obs, info = env.reset(options=options)
    print(info)
    # test five examples
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        print(info)
        if done or trunc:
            break

def reset_test():
    obs, info = env.reset()
    # test five examples
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        print(info)
        if done or trunc:
            break
        

if __name__ == "__main__":
    # # given state
    # state = {
    #     "s": 19,
    #     "d": 59,
    #     "a": {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [2, 3, 4, 6, 7, 8, 9]},
    #     "p": [{"i": 0, "c": -13, "m": 1}],
    #     "t": 0
    # }
    
    # manual_reset_test(state)
    
    reset_test()
