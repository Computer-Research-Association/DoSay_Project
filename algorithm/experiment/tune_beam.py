"""
빔서치 weight 튜닝 (Optuna) — 데드락 방지 버전.

각 trial이 검증된 beam.py를 서브프로세스로 호출한다 (beam.py가 내부에서 병렬).
→ Optuna 쪽엔 Pool을 두지 않아 중첩 Pool 데드락이 원천적으로 없음.

action_count weight=1.0 고정(scale invariance), 나머지를 [0,5] 탐색.
모든 trial 같은 고정 시드 → paired 공정 비교.

사용법:
    python algorithm/experiment/tune_beam.py --version Beam_V06d --trials 30 --games 60
"""
import os
import sys
import json
import argparse
import importlib
import subprocess
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))

BEAM = str(ROOT / "algorithm" / "experiment" / "beam.py")
PY = sys.executable

_TUNABLE = ["feature_remove_nine", "feature_remove_eight",
            "feature_action_diversity", "feature_diversity_sig",
            "feature_big_diversity"]


def _run(version, width, depth, games, weights):
    """beam.py를 서브프로세스로 실행해 avg score 반환."""
    cmd = [PY, BEAM, "--version", version, "--width", str(width),
           "--depth", str(depth), "--games", str(games), "--quiet"]
    if weights:
        cmd += ["--weights", json.dumps(weights)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    try:
        return float(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise RuntimeError(f"beam.py 실행 실패:\n{out.stderr[-500:]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="Beam_V06d")
    p.add_argument("--width", type=int, default=5)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--games", type=int, default=60)
    args = p.parse_args()

    reg = importlib.import_module(f"algorithm.models.version.{args.version}").registry
    names = [e.name for e in reg.entries]
    present = [n for n in _TUNABLE if n in names]

    base = _run(args.version, args.width, args.depth, args.games,
                {n: 1.0 for n in names})
    print(f"[{args.version}] beam({args.width},{args.depth}), {args.games}판")
    print(f"[기준선] 전부 weight 1.0 → {base:.2f}")
    print(f"튜닝 대상: {[n.replace('feature_','') for n in present]} (action_count=1.0 고정)\n")

    def objective(trial):
        w = {"feature_action_count": 1.0}
        for n in present:
            w[n] = trial.suggest_float(n.replace("feature_", ""), 0.0, 5.0)
        return _run(args.version, args.width, args.depth, args.games, w)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))

    def cb(st, tr):
        ps = " ".join(f"{k}={v:.2f}" for k, v in tr.params.items())
        print(f"  trial {tr.number:>2}: {tr.value:.2f}  ({ps}) | best {st.best_value:.2f}", flush=True)

    study.optimize(objective, n_trials=args.trials, callbacks=[cb])

    print(f"\n=== 최적 ({args.games}판) ===")
    print(f" 기준선(all 1.0): {base:.2f}")
    print(f" 최적 avg       : {study.best_value:.2f}  ({study.best_value-base:+.2f})")
    print(f" 최적 weight    : {study.best_params}  (+ action_count=1.0)")


if __name__ == "__main__":
    main()
