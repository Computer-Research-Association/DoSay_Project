"""
가치 데이터 추출 — 한 보드를 어닐링(worst)해서 best 수순을 찾고,
그 수순의 각 상태에 대해 '남은 도달가능 칸수'(= best_total - 지금까지 제거)를 라벨로 emit.

출력: JSONL, 한 줄 = {"s": [162 flat state], "v": 남은칸수}
사용법: python dl/anneal_seq.py --seed 1234 --iters 1200
"""
import os, sys, json, math, random, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

import algorithm.experiment.anneal as A   # noqa: E402
from algorithm.experiment.anneal import (  # noqa: E402
    _make_grid, _valid_actions, _apply, _rollout, _build_prefix, _pick_t, _load_policy,
)


def anneal_best_seq(seed, iters, rng, T0=3.0, greedy_prob=0.8):
    grid = _make_grid(seed)
    acts = _valid_actions(grid)
    if not acts:
        return [], 0
    first = A._POLICY(grid, acts, rng)
    cur_seq, cur_score = _rollout(grid.copy(), first, rng, 1.0)
    best_seq, best_score = cur_seq, cur_score
    pg, pc = _build_prefix(seed, cur_seq)
    for it in range(iters):
        T = T0 * (1 - it / iters) + 1e-6
        if len(cur_seq) < 2:
            break
        t = _pick_t(len(cur_seq), "worst", rng, pc)
        base_g = pg[t]
        acts = _valid_actions(base_g)
        if len(acts) < 2:
            continue
        cur_a = cur_seq[t]
        alt = [a for a in acts if a != cur_a]
        if not alt:
            continue
        nf = alt[rng.randrange(len(alt))]
        tail_seq, tail_total = _rollout(base_g.copy(), nf, rng, greedy_prob)
        new_score = pc[t] + tail_total
        delta = new_score - cur_score
        if delta >= 0 or rng.random() < math.exp(delta / T):
            cur_seq = cur_seq[:t] + tail_seq
            cur_score = new_score
            pg, pc = _build_prefix(seed, cur_seq)
            if cur_score > best_score:
                best_seq, best_score = cur_seq, cur_score
    return best_seq, best_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="Anneal_V01b")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--iters", type=int, default=1200)
    args = p.parse_args()
    _load_policy(args.version)
    rng = random.Random(args.seed)
    seq, total = anneal_best_seq(args.seed, args.iters, rng)

    grid = _make_grid(args.seed)
    cleared = 0
    out = []
    out.append({"s": grid.flatten().tolist(), "v": total - cleared})   # 시작 상태
    for a in seq:
        cleared += _apply(grid, a)
        out.append({"s": grid.flatten().tolist(), "v": total - cleared})
    sys.stdout.write("\n".join(json.dumps(o) for o in out) + "\n")


if __name__ == "__main__":
    main()
