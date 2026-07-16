import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import numpy as np
import gymnasium as gym
import ai.envs


def run(n_episodes: int = 300, rows: int = 9, cols: int = 18):
    env = gym.make("envs/AppleGame-v0", rows=rows, cols=cols)
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(n_episodes):
        _, info = env.reset()
        done = False
        while not done:
            valid = np.flatnonzero(info["action_mask"])
            _, _, done, _, info = env.step(int(rng.choice(valid)))
        scores.append(info["score"])
    scores = np.asarray(scores)
    print(f"랜덤 정책 점수: {scores.mean():.1f} ± {scores.std():.1f} "
          f"(min {scores.min()}, max {scores.max()}) / {rows*cols}")


if __name__ == "__main__":
    run()