"""
Optuna로 nine/eight/action_count 의 weight 비율을 최적화한다.

핵심:
- weight는 파일에 안 박고, executor 로드 후 registry.entries[i].weight 를 덮어써서 시도.
  → Optuna가 파일 생성 없이 자유롭게 weight를 탐색.
- score = sum(w_i * f_i) 는 모든 w를 상수배해도 argmax가 안 바뀜(scale invariant).
  → action_count weight를 1.0으로 고정하고 nine/eight만 튜닝(2D). 중복 차원 제거.
- 모든 trial이 같은 고정 시드로 평가 → paired 비교라 공정.

사용법:
    python algorithm/experiment/tune.py
    python algorithm/experiment/tune.py --trials 60 --games 60
"""
import os
import sys
import argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import optuna

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))
os.chdir(ROOT)

from algorithm.models.model_executor import GreedyExecutor   # noqa: E402

GRID = (9, 18)
TOTAL = GRID[0] * GRID[1]
VERSION_DIR = ROOT / "algorithm" / "models" / "version"
MODEL = "Greedy_V08c"          # nine + eight + action_count 3개 피처
BASE_SEED = 1234

_EX = None

def _init(model_path_str):
    global _EX
    _EX = GreedyExecutor(GRID, Path(model_path_str))

def _play(arg):
    seed, weights = arg
    for e in _EX.cls.registry.entries:     # 이 trial의 weight로 덮어쓰기
        e.weight = weights.get(e.name, e.weight)
    _EX.reset(seed)
    score = 0
    while True:
        over, _ = _EX.board.is_done()
        if over:
            break
        score += _EX.do_step()
    return score

_POOL = None
_N_GAMES = 50

def _evaluate(weights):
    args = [(BASE_SEED + i, weights) for i in range(_N_GAMES)]
    scores = _POOL.map(_play, args)
    return float(np.mean(scores))

def objective(trial):
    weights = {
        "feature_remove_nine":   trial.suggest_float("nine", 0.0, 8.0),
        "feature_remove_eight":  trial.suggest_float("eight", 0.0, 8.0),
        "feature_action_count":  1.0,      # 앵커 고정 (scale invariance)
    }
    return _evaluate(weights)


def main():
    global _POOL, _N_GAMES
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--games", type=int, default=50)
    args = p.parse_args()
    _N_GAMES = args.games

    model_path = (VERSION_DIR / f"{MODEL}.py").resolve()
    _POOL = Pool(processes=min(os.cpu_count() or 1, _N_GAMES),
                 initializer=_init, initargs=(str(model_path),))

    # 기준선: 전부 weight 1.0 (지금 V8c)
    base = _evaluate({"feature_remove_nine": 1.0, "feature_remove_eight": 1.0,
                      "feature_action_count": 1.0})
    print(f"[기준선] 전부 1.0 → avg {base:.2f}  ({_N_GAMES}판)\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))

    def cb(study, trial):
        best = study.best_value
        print(f"  trial {trial.number:>2}: {trial.value:.2f}  "
              f"(nine={trial.params['nine']:.2f} eight={trial.params['eight']:.2f}) "
              f"| best {best:.2f}")

    study.optimize(objective, n_trials=args.trials, callbacks=[cb])

    print(f"\n=== 최적 결과 ({_N_GAMES}판 기준) ===")
    print(f" 기준선(1,1,1) : {base:.2f}")
    print(f" 최적 avg      : {study.best_value:.2f}  (기준 대비 {study.best_value-base:+.2f})")
    bp = study.best_params
    print(f" 최적 weight   : nine={bp['nine']:.2f}  eight={bp['eight']:.2f}  action_count=1.00")
    _POOL.close()


if __name__ == "__main__":
    main()
