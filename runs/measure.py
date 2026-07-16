from typing import cast

import gymnasium as gym

import ai.envs
from ai.envs.apple_env import AppleGameEnv
from agents.ai.agent import AIAgent
from ai.wrappers.action_mask import wrap_with_mask

MODEL_PATH = "ai/MaskablePPO_V5.1_10000000.zip"
N_EPISODES = 10000
BASE_SEED  = 1234
ROWS, COLS = 9, 18

def main():
    env = gym.make("envs/AppleGame-v0", render_mode="None", rows=ROWS, cols=COLS)
    env = cast(AppleGameEnv, wrap_with_mask(env))

    agent = AIAgent(env, MODEL_PATH)
    print(f"\nRunning {N_EPISODES} episodes ...\n")

    steps_list, score_list = [], []
    for i in range(N_EPISODES):
        seed = BASE_SEED + i
        agent.set_board()
        steps, score = agent.run_episode(False)  # type: ignore
        steps_list.append(steps)
        score_list.append(score)

    env.close()

    n = len(score_list)
    best_i = max(range(n), key=lambda k: score_list[k])
    worst_i = min(range(n), key=lambda k: score_list[k])
    print("\n── Summary ─────────────────────────────────")
    print(f" episodes    : {n}")
    print(f" avg moves   : {sum(steps_list) / n:.2f}")
    print(f" avg score   : {sum(score_list) / n:.2f}")
    print(f" best score  : {score_list[best_i]}  (seed={BASE_SEED + best_i})")
    print(f" worst score : {score_list[worst_i]}  (seed={BASE_SEED + worst_i})")
    print("────────────────────────────────────────────")

def select_agent_type():
    tag_list = ['ai', 'algorithm']
    for i in range(len(tag_list)):
        tag = tag_list[i]



if __name__ == "__main__":
    main()