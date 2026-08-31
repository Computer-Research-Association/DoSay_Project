"""
DAgger 상태 수집 — 현재 value-beam(있으면) + ε-탐험으로 게임을 두며 방문 상태 수집.
→ 이 상태들을 label_states.py가 어닐링으로 라벨링 (on-policy 분포 커버).

넷(value_model.pt) 없으면 fewest-cells로 폴백.
출력: states.jsonl (한 줄 = {"s":[162]})

사용법: python dl/gen_states.py --games 400 --base 20000 --eps 0.25 --sample 0.5
"""
import os, sys, json, random, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from algorithm.experiment.anneal import _make_grid, _valid_actions   # noqa: E402
H, W = 9, 18

_net = None
_dev = None


def _load_net():
    global _net, _dev
    import torch
    from dl.train_value import ValueNet
    mp = ROOT / "dl" / "value_model.pt"
    if not mp.exists():
        return False
    _dev = "mps" if torch.backends.mps.is_available() else "cpu"
    _net = ValueNet().to(_dev)
    _net.load_state_dict(torch.load(mp, map_location=_dev))
    _net.eval()
    return True


def _V(grids):
    import torch
    X = np.stack(grids).astype(np.float32).reshape(-1, 1, H, W) / 9.0
    with torch.no_grad():
        return _net(torch.from_numpy(X).to(_dev)).cpu().numpy()


def _choose(grid, acts, eps, rng):
    """ε 확률 랜덤, 아니면 value 1-ply(넷 있으면) 또는 fewest-cells."""
    if rng.random() < eps:
        return acts[rng.randrange(len(acts))]
    if _net is None:
        return min(acts, key=lambda a: int(np.count_nonzero(grid[a[0]:a[2]+1, a[1]:a[3]+1])))
    children, imm = [], []
    for a in acts:
        c = grid.copy(); c[a[0]:a[2]+1, a[1]:a[3]+1] = 0
        children.append(c); imm.append(int(np.count_nonzero(grid[a[0]:a[2]+1, a[1]:a[3]+1])))
    val = np.array(imm) + _V(children)
    return acts[int(val.argmax())]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--base", type=int, default=20000)   # 상태수집 시드 시작(라벨링/평가와 겹치지 않게)
    p.add_argument("--eps", type=float, default=0.25)
    p.add_argument("--sample", type=float, default=0.5)  # 방문 상태 중 저장 비율
    args = p.parse_args()
    _load_net()
    rng = random.Random(args.base)
    out = open(ROOT / "dl" / "states.jsonl", "w")
    cnt = 0
    for gi in range(args.games):
        g = _make_grid(args.base + gi)
        while True:
            acts = _valid_actions(g)
            if not acts:
                break
            if rng.random() < args.sample:
                out.write(json.dumps({"s": g.flatten().tolist()}) + "\n"); cnt += 1
            a = _choose(g, acts, args.eps, rng)
            g[a[0]:a[2]+1, a[1]:a[3]+1] = 0
    out.close()
    print(f"수집 상태 {cnt:,}개  ({args.games}판, net={'O' if _net else 'X(fewest)'})  → dl/states.jsonl")


if __name__ == "__main__":
    main()
