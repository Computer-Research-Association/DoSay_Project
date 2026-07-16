# ai/models/eval_baselines.py
# V5의 105점이 절대적으로 어느 수준인지 좌표를 잡기 위한 휴리스틱 베이스라인.
#   - random     : 유효 행동 중 균등 샘플            (측정 결과: 97.9 ± 11.7)
#   - greedy_max : 매 수 '가장 많이 지우는' 사각형 선택
#   - greedy_min : 매 수 '가장 적게 지우는' 사각형 선택 (조합 보존 성향)
# 학습된 정책이 greedy_min조차 못 넘는다면 아직 단순 휴리스틱 수준이라는 뜻이고,
# 넘는다면 실제로 비자명한 전략을 배운 것.
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT)

import numpy as np
import gymnasium as gym
import ai.envs

ROWS, COLS = 9, 18


def _removed_count(board, action) -> int:
    (r1, c1), (r2, c2) = action.top_left, action.bottom_right
    return int(np.count_nonzero(board.grid[r1:r2 + 1, c1:c2 + 1]))


def run_policy(env, pick_fn, n_episodes: int, rng: np.random.Generator) -> np.ndarray:
    scores = []
    for _ in range(n_episodes):
        _, info = env.reset()
        done = False
        while not done:
            u = env.unwrapped
            action = pick_fn(u, info, rng)
            _, _, done, _, info = env.step(u.action_to_index[action])
        scores.append(info["score"])
    return np.asarray(scores)


def pick_random(u, info, rng):
    acts = u.board.get_valid_actions()
    return acts[rng.integers(len(acts))]


def pick_greedy_max(u, info, rng):
    return max(u.board.get_valid_actions(), key=lambda a: _removed_count(u.board, a))


def pick_greedy_min(u, info, rng):
    return min(u.board.get_valid_actions(), key=lambda a: _removed_count(u.board, a))


if __name__ == "__main__":
    env = gym.make("envs/AppleGame-v0", rows=ROWS, cols=COLS)
    rng = np.random.default_rng(0)
    n = 300

    for name, fn in [("random", pick_random),
                     ("greedy_max", pick_greedy_max),
                     ("greedy_min", pick_greedy_min)]:
        s = run_policy(env, fn, n, rng)
        print(f"{name:>10}: {s.mean():6.1f} ± {s.std():4.1f}  "
              f"(min {s.min()}, max {s.max()}) / {ROWS * COLS}")