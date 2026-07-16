from gymnasium.envs.registration import register

register(
    id="envs/AppleGame-v0",
    entry_point="ai.envs.apple_env:AppleGameEnv"
)