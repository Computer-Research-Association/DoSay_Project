"""
임의 상태에서 어닐링 — 주어진 grid에서 도달 가능한 '최대 추가 제거 칸수'를 근사.
가치 라벨러의 핵심 부품 (seed 기반 anneal.py를 grid 기반으로 일반화).
"""
import os, sys, math, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

import algorithm.experiment.anneal as A   # noqa: E402
from algorithm.experiment.anneal import (  # noqa: E402
    _valid_actions, _apply, _rollout, _pick_t,
)


def _build_prefix_grid(grid0, actions):
    grid = grid0.copy()
    pgrid = [grid.copy()]; pclear = [0]; acc = 0
    for a in actions:
        acc += _apply(grid, a)
        pgrid.append(grid.copy()); pclear.append(acc)
    return pgrid, pclear


def anneal_from_grid(grid0, iters, rng, T0=3.0, greedy_prob=0.8):
    """grid0에서 worst-이웃 어닐링. best 추가 제거 칸수 반환."""
    acts = _valid_actions(grid0)
    if not acts:
        return 0
    first = A._POLICY(grid0, acts, rng)
    cur_seq, cur_score = _rollout(grid0.copy(), first, rng, 1.0)
    best = cur_score
    pg, pc = _build_prefix_grid(grid0, cur_seq)
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
            pg, pc = _build_prefix_grid(grid0, cur_seq)
            if cur_score > best:
                best = cur_score
    return best
