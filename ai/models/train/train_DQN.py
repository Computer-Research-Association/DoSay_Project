import gymnasium as gym
# from ai.envs.apple_env import AppleGameEnv
from stable_baselines3 import DQN
import ai.envs



env = gym.make("envs/AppleGame-v0", render_mode="ansi", rows=9, cols=18)

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="ai\\logs")
model.learn(total_timesteps=100_000)

model.save("ai\\DQN_AppleGame")
print("AI 모델 저장 완료!")