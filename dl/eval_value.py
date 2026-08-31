"""
가치기반 탐색 평가 — V(s)로 수를 고름.
  한 수 결정: 각 후보 a에 대해 (즉시 제거칸 + V(child)) 최대인 a 선택. (1-ply value greedy)
비교: fewest-cells 112 / action-max 118 / 어닐링 135 기준.

사용법: python dl/eval_value.py --games 20
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from algorithm.experiment.anneal import _make_grid, _valid_actions, _apply, BASE_SEED   # noqa: E402
from dl.train_value import ValueNet   # noqa: E402
H, W = 9, 18

_dev = "mps" if torch.backends.mps.is_available() else "cpu"
_net = ValueNet().to(_dev)
_net.load_state_dict(torch.load(ROOT / "dl" / "value_model.pt", map_location=_dev))
_net.eval()


def value_batch(grids):
    """grids: list of (9,18) → V 예측 배열."""
    X = np.stack(grids).astype(np.float32).reshape(-1, 1, H, W) / 9.0
    with torch.no_grad():
        return _net(torch.from_numpy(X).to(_dev)).cpu().numpy()


def play(seed):
    g = _make_grid(seed); score = 0
    while True:
        acts = _valid_actions(g)
        if not acts:
            return score
        children, imm = [], []
        for a in acts:
            r1, c1, r2, c2 = a
            child = g.copy()
            cleared = int(np.count_nonzero(child[r1:r2+1, c1:c2+1]))
            child[r1:r2+1, c1:c2+1] = 0
            children.append(child); imm.append(cleared)
        v = value_batch(children)
        val = np.array(imm) + v            # 즉시 + 남은 가치
        a = acts[int(val.argmax())]
        score += _apply(g, a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=20)
    args = p.parse_args()
    play(9999)   # 워밍업
    scores = [play(BASE_SEED + i) for i in range(args.games)]
    arr = np.array(scores)
    for i, s in enumerate(scores):
        print(f"  seed {BASE_SEED+i} : {s}")
    print(f"가치기반 greedy 평균({args.games}판): {arr.mean():.2f}  (min {arr.min()} max {arr.max()})")


if __name__ == "__main__":
    main()
