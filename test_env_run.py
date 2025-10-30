import divide21env 
import gymnasium as gym

# Test the environment registration
env = gym.make("Divide21-v0")

# Reset and run a few steps
obs, info = env.reset()

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    print(info)
    if done or trunc:
        print("Game ended.")
        break
