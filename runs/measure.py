import os
import time
from pathlib import Path
import numpy as np
import game
from agents.selector import select_agent
from system_info import format_system_info

ROOT_DIR = Path(game.__file__).resolve().parent.parent
N_EPISODES = 100
BASE_SEED  = 1234
ROWS, COLS = 9, 18

os.chdir(ROOT_DIR)


def main():
    agent_cls, model_path = select_agent(agents_dir=ROOT_DIR)
    agent = agent_cls((ROWS, COLS), model_path)
    print(agent.get_info_formatted())
    print(format_system_info(agent))
    print(f"\nRunning {N_EPISODES} episodes ...\n")
    start_time = time.time()

    steps_list, score_list = [], []
    for i in range(N_EPISODES):
        seed = BASE_SEED + i
        steps, score = agent.run_episode(board_source=seed, render=False, delay=0.5)
        steps_list.append(steps)
        score_list.append(score)
        # print(f"[ep {i + 1}]  seed= {seed}, moves= {steps:>2}, score= {score:>3}")

    time_elapsed = time.time() - start_time
    n = len(score_list)
    best_i = max(range(n), key=lambda k: score_list[k])
    worst_i = min(range(n), key=lambda k: score_list[k])
    print("\n── Summary ─────────────────────────────────")
    print(f" episodes     : {n}")
    print(f" avg moves    : {sum(steps_list) / n:.2f}")
    print(f" avg score    : {sum(score_list) / n:.2f}")
    print(f" std score    : {np.std(score_list):.2f}")
    print(f" best score   : {score_list[best_i]}  (seed={BASE_SEED + best_i})")
    print(f" worst score  : {score_list[worst_i]}  (seed={BASE_SEED + worst_i})")
    print(f" time elapsed : {int(time_elapsed)}s (avg {time_elapsed / n:.2f})")
    print("────────────────────────────────────────────")


if __name__ == "__main__":
    main()