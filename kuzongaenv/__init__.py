from gymnasium.envs.registration import register

register(
    id="Kuzonga-v0",
    entry_point="kuzongaenv.envs.kuzonga_env:KuzongaEnv",
)
