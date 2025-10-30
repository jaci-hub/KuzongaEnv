from gymnasium.envs.registration import register

register(
    id="Divide21-v0",
    entry_point="divide21env.envs.divide21_env:Divide21Env",
)
