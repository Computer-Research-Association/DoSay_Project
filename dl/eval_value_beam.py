"""
가치기반 빔서치 — 빔의 leaf 평가를 손수제 휴리스틱 대신 학습된 V(s)로.
  각 단계 상위 W개(누적제거 + V(leaf) 기준) 유지, depth까지, 최고의 첫 수 선택.
비교: fewest 112 / actmax 118 / value-greedy 105 / 어닐링 135.

사용법: python dl/eval_value_beam.py --width 5 --depth 3 --games 20
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from algorithm.experiment.anneal import _make_grid, _valid_actions, BASE_SEED   # noqa: E402
from dl.train_value import ValueNet   # noqa: E402
H, W = 9, 18

_dev = "mps" if torch.backends.mps.is_available() else "cpu"
_net = ValueNet().to(_dev)
_net.load_state_dict(torch.load(ROOT / "dl" / "value_model.pt", map_location=_dev))
_net.eval()


def V(grids):
    X = np.stack(grids).astype(np.float32).reshape(-1, 1, H, W) / 9.0
    with torch.no_grad():
        return _net(torch.from_numpy(X).to(_dev)).cpu().numpy()


def apply(g, a):
    r1, c1, r2, c2 = a
    child = g.copy()
    cl = int(np.count_nonzero(child[r1:r2+1, c1:c2+1]))
    child[r1:r2+1, c1:c2+1] = 0
    return child, cl


def beam_choose(grid, width, depth):
    root = _valid_actions(grid)
    if not root:
        return None
    beam = []   # (grid, first_action, cum_cleared)
    for a in root:
        child, cl = apply(grid, a)
        beam.append((child, a, cl))
    # 평가: cum + V(leaf)
    vals = V([b[0] for b in beam]) + np.array([b[2] for b in beam])
    order = np.argsort(-vals)[:width]
    beam = [beam[i] for i in order]
    for _ in range(depth - 1):
        cand = []
        for g, fa, cum in beam:
            acts = _valid_actions(g)
            if not acts:
                cand.append((g, fa, cum)); continue
            for a in acts:
                child, cl = apply(g, a)
                cand.append((child, fa, cum + cl))
        vals = V([c[0] for c in cand]) + np.array([c[2] for c in cand])
        order = np.argsort(-vals)[:width]
        beam = [cand[i] for i in order]
    vals = V([b[0] for b in beam]) + np.array([b[2] for b in beam])
    return beam[int(vals.argmax())][1]


def play(seed, width, depth):
    g = _make_grid(seed); score = 0
    while True:
        if not _valid_actions(g):
            return score
        a = beam_choose(g, width, depth)
        if a is None:
            return score
        _, cl = apply(g, a)
        r1, c1, r2, c2 = a
        score += int(np.count_nonzero(g[r1:r2+1, c1:c2+1]))
        g[r1:r2+1, c1:c2+1] = 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=5)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--games", type=int, default=20)
    args = p.parse_args()
    play(9999, args.width, args.depth)   # 워밍업
    scores = [play(BASE_SEED + i, args.width, args.depth) for i in range(args.games)]
    arr = np.array(scores)
    print(f"value-beam w{args.width} d{args.depth} 평균({args.games}판): {arr.mean():.2f}  (min {arr.min()} max {arr.max()})")


if __name__ == "__main__":
    main()
