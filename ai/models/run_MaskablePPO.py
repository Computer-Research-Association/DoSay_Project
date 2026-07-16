import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import time
import numpy as np
import gymnasium as gym
import ai.envs

from sb3_contrib import MaskablePPO

from ai.wrappers.action_mask import wrap_with_mask


# MODEL_PATH = "ai\\MaskablePPO_V2"
MODEL_PATH = "ai\\MaskablePPO_V4.0_1000000"
N_EPISODES = 10000
DELAY_SEC = 0.5  # 한 수 둘 때마다 대기 시간 (보기 편하게)
BASE_SEED = 1234

# apple_grid = np.array([
#     [4, 2, 4, 3, 2, 5, 6, 5, 4, 9, 3, 6, 5, 5, 1, 6, 3, 3],
#     [1, 3, 5, 1, 6, 4, 1, 3, 7, 6, 1, 2, 4, 3, 3, 3, 7, 7],
#     [9, 6, 4, 3, 5, 5, 1, 7, 8, 4, 4, 2, 6, 4, 5, 7, 2, 3],
#     [5, 2, 6, 4, 9, 2, 8, 3, 1, 2, 2, 2, 7, 5, 2, 1, 1, 9],
#     [2, 2, 8, 6, 1, 5, 2, 8, 8, 4, 2, 8, 3, 2, 5, 3, 7, 3],
#     [3, 2, 8, 2, 6, 5, 6, 5, 2, 1, 3, 1, 4, 2, 3, 7, 3, 1],
#     [3, 8, 7, 8, 1, 9, 6, 2, 5, 7, 5, 9, 3, 2, 4, 4, 7, 6],
#     [2, 3, 1, 2, 4, 6, 1, 3, 4, 1, 9, 6, 4, 6, 4, 3, 1, 6],
#     [2, 1, 5, 8, 6, 2, 6, 8, 2, 2, 3, 4, 3, 6, 2, 1, 9, 6]
# ])

step_count_stack = []
score_stack = []

def play_episode(model: MaskablePPO, env, episode_idx: int):
    # obs, info = env.reset(options = {"init_board": apple_grid})
    obs, info = env.reset(seed=BASE_SEED + episode_idx)
    terminated = False
    truncated = False
    step_count = 0

    print(f"\n===== Episode {episode_idx + 1} 시작 =====")
    if env.render_mode != "None":
        env.render()

    while not (terminated or truncated):
        action_masks = env.unwrapped.get_action_mask()

        if not action_masks.any():
            # 유효한 수가 없으면 즉시 종료 (이론상 terminated와 함께 걸림)
            break

        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        step_count += 1

        if env.render_mode != "None":
            print(f"\n[step {step_count}] reward={reward:.1f} score={info['score']}")
            env.render()
            time.sleep(DELAY_SEC)

    step_count_stack.append(step_count)
    score_stack.append(info['score'])
    print(f"===== Episode {episode_idx + 1} 종료 (총 {step_count}수, 최종 점수 {info['score']}) =====")


if __name__ == "__main__":
    env = gym.make("envs/AppleGame-v0", render_mode="None", rows=9, cols=18)
    env = wrap_with_mask(env)

    model = MaskablePPO.load(MODEL_PATH, env=env)

    for ep in range(N_EPISODES):
        play_episode(model, env, ep)
    print(f"===== 총 에피소드 {N_EPISODES}회 종료 (평균 {(sum(step_count_stack)/N_EPISODES):.2f}수, 평균 점수 {(sum(score_stack)/N_EPISODES):.2f}점) =====")