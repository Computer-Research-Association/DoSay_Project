import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import gymnasium as gym
from stable_baselines3 import DQN
import ai.envs

# 1. 환경만 만들고
env = gym.make("envs/AppleGame-v0", render_mode="ansi", rows=9, cols=18)

# 2. 저장해둔 AI 파일만 쏙 불러와서
model = DQN.load("DQN_AppleGame")

# 3. 바로 게임 플레이!
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, _ = env.step(action)
    done = term or trunc
env.close()