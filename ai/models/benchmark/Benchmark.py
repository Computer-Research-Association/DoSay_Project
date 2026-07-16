import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(ROOT)
if ROOT not in sys.path:
    sys.path.append(ROOT)

import gymnasium as gym
from Common import load_model, run_episode, format_model_info, quiet_logs

import ai.envs
from ai.wrappers.action_mask import wrap_with_mask

MODEL_PATH = "ai/MaskablePPO_V5.1_10000000.zip"
N_EPISODES = 100
BASE_SEED  = 1234
ROWS, COLS = 9, 18


def main():
    quiet_logs()

    env = gym.make("envs/AppleGame-v0", render_mode="None", rows=ROWS, cols=COLS)
    env = wrap_with_mask(env)

    model, meta = load_model(MODEL_PATH, env=env)
    print(format_model_info(model, meta))
    print(f"\nRunning {N_EPISODES} episodes ...\n")

    steps_list, score_list = [], []
    for i in range(N_EPISODES):
        seed = BASE_SEED + i
        steps, score = run_episode(model, env, meta, seed=seed)
        steps_list.append(steps)
        score_list.append(score)
        print(f"[ep {i + 1}]  seed= {seed}, moves= {steps:>2}, score= {score:>3}")

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


if __name__ == "__main__":
    main()