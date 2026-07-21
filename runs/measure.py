from agents.ai.agent import AIAgent
from agents.algorithm.agent import AlgoAgent

MODEL_PATH = "ai/MaskablePPO_V5.1_10000000.zip"
N_EPISODES = 1000
BASE_SEED  = 1234
ROWS, COLS = 9, 18

WEIGHTS    = {"remove_nine": 1.0, "remove_eight": 1.0, "grouping": 1.0}
AGENT_VERSION = 1.0


def main():
    agent_type = select_agent_type()
    if agent_type == "ai":
        agent = AIAgent((ROWS, COLS), MODEL_PATH)
    else: 
        agent = AlgoAgent((ROWS, COLS), weights=WEIGHTS, version=AGENT_VERSION)

    
    print(f"\nRunning {N_EPISODES} episodes ...\n")

    steps_list, score_list = [], []
    for i in range(N_EPISODES):
        seed = BASE_SEED + i
        steps, score = agent.run_episode(board_source=seed, render=False, delay=0.5)
        steps_list.append(steps)
        score_list.append(score)

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
    for i, tag in enumerate(tag_list):
        print(f"{i}: {tag}")

    while True:
        raw = input("번호 입력: ").strip()
        if raw.isdigit() and int(raw) < len(tag_list):
            return tag_list[int(raw)]
        print("잘못된 입력입니다. 다시 입력하세요.")

if __name__ == "__main__":
    main()