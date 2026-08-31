"""
value head warm-start — 어닐링 (상태→가치) 데이터로 AZ 망의 value를 먼저 학습.
목적: 냉시작 신호 0 문제 해결 → value가 상태 구분 → MCTS가 방향 잡음 → self-play 부트스트랩 시작.
policy head는 건드리지 않음 (self-play가 학습).

사용: python dl/az/pretrain.py --epochs 25
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from dl.az.net import Net, encode   # noqa: E402
from dl.az.game import TOTAL         # noqa: E402
H, W = 9, 18


def load():
    S, V = [], []
    with open(ROOT / "dl" / "value_data.jsonl") as f:
        for line in f:
            d = json.loads(line)
            S.append(np.array(d["s"], dtype=np.int8).reshape(H, W))
            V.append(d["v"] / TOTAL)      # AZ value 스케일 [0,1]
    return S, np.array(V, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--bs", type=int, default=256)
    args = p.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    S, V = load()
    X = torch.from_numpy(np.stack([encode(s) for s in S]))
    y = torch.from_numpy(V)
    n = len(y); nval = n // 10
    idx = torch.randperm(n); vi, ti = idx[:nval], idx[nval:]
    Xt, yt, Xv, yv = X[ti], y[ti], X[vi].to(dev), y[vi].to(dev)

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4)   # 안정화
    best_mae, best_state = 1e9, None
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(len(yt))
        for b in range(0, len(yt), args.bs):
            j = perm[b:b + args.bs]
            xb, yb = Xt[j].to(dev), yt[j].to(dev)
            val = net.value(net.features(xb))
            loss = F.mse_loss(val, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            mae = (net.value(net.features(Xv)) - yv).abs().mean().item() * TOTAL
        if mae < best_mae:                        # best 저장 (발산 방지)
            best_mae = mae
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if ep % 3 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:2d}  val MAE {mae:.1f}칸  (best {best_mae:.1f})")
    net.load_state_dict(best_state)
    torch.save(net.state_dict(), ROOT / "dl" / "az" / "model.pt")
    print(f"→ dl/az/model.pt (value warm-start, best MAE {best_mae:.1f}칸, n={n})")


if __name__ == "__main__":
    main()
