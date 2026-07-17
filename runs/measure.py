import time
from agents.ai.agent import AIAgent

MODEL_PATH = "ai/MaskablePPO_V5.1_10000000.zip"
N_EPISODES = 10000
BASE_SEED  = 1234
ROWS, COLS = 9, 18

def main():
    agent = AIAgent((ROWS, COLS), MODEL_PATH)
    print(agent.get_info_formatted())
    print(f"\nRunning {N_EPISODES} episodes ...\n")
    start_time = time.time()

    steps_list, score_list = [], []
    for i in range(N_EPISODES):
        seed = BASE_SEED + i
        steps, score = agent.run_episode(board_source=seed, render=False, delay=0.5)
        steps_list.append(steps)
        score_list.append(score)
        print(f"[ep {i + 1}]  seed= {seed}, moves= {steps:>2}, score= {score:>3}")

    time_elapsed = time.time() - start_time
    n = len(score_list)
    best_i = max(range(n), key=lambda k: score_list[k])
    worst_i = min(range(n), key=lambda k: score_list[k])
    print("\n── Summary ─────────────────────────────────")
    print(f" episodes     : {n}")
    print(f" avg moves    : {sum(steps_list) / n:.2f}")
    print(f" avg score    : {sum(score_list) / n:.2f}")
    print(f" best score   : {score_list[best_i]}  (seed={BASE_SEED + best_i})")
    print(f" worst score  : {score_list[worst_i]}  (seed={BASE_SEED + worst_i})")
    print(f" time elapsed : {int(time_elapsed)}s (avg {time_elapsed / n:.2f})")
    print("────────────────────────────────────────────")

def select_agent_type():
    tag_list = ['ai', 'algorithm']
    for i in range(len(tag_list)):
        tag = tag_list[i]

if __name__ == "__main__":
    main()