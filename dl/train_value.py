"""
가치망 학습 — 입력: 보드상태(9×18), 출력: 남은 도달가능 칸수 V(s).
value_data.jsonl (어닐링 best 수순의 상태별 라벨)로 학습.

사용법: python dl/train_value.py --epochs 40
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
H, W = 9, 18


class ValueNet(nn.Module):
    def __init__(self, ch=48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(ch * H * W, 256), nn.ReLU(), nn.Linear(256, 1),
        )

    def forward(self, x):
        z = self.conv(x)
        return self.head(z.flatten(1)).squeeze(-1)


def load(path):
    S, V = [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            S.append(d["s"]); V.append(d["v"])
    S = np.array(S, dtype=np.float32).reshape(-1, 1, H, W) / 9.0
    V = np.array(V, dtype=np.float32)
    return torch.from_numpy(S), torch.from_numpy(V)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=512)
    args = p.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    X, y = load(ROOT / "dl" / "value_data.jsonl")
    n = len(y); idx = torch.randperm(n); nval = n // 10
    vi, ti = idx[:nval], idx[nval:]
    Xt, yt, Xv, yv = X[ti], y[ti], X[vi].to(dev), y[vi].to(dev)

    net = ValueNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(len(yt))
        for b in range(0, len(yt), args.bs):
            j = perm[b:b+args.bs]
            xb, yb = Xt[j].to(dev), yt[j].to(dev)
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            mae = (net(Xv) - yv).abs().mean().item()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:2d}  val MAE {mae:.2f} (칸)")
    torch.save(net.state_dict(), ROOT / "dl" / "value_model.pt")
    print(f"→ dl/value_model.pt 저장  (device={dev}, n={n})")


if __name__ == "__main__":
    main()
