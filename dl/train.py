"""
DL 정책 파일럿 — 학습.

입력: 2채널 9×18  (ch0 = 보드값/9, ch1 = 후보 사각형 마스크)
출력: 스칼라 = 그 수를 둔 뒤 남는 유효액션 수 (회귀)
→ 학습된 넷을 정책으로: 모든 후보를 배치 추론해 argmax = action-max의 빠른 근사

사용법: python dl/train.py --epochs 30
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
H, W = 9, 18


class Net(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(ch * H * W, 128), nn.ReLU(), nn.Linear(128, 1),
        )

    def forward(self, x):
        z = self.conv(x)
        return self.head(z.flatten(1)).squeeze(-1)


def build_inputs(boards, rects):
    """(N,9,18) int8 보드 + (N,4) 사각형 → (N,2,9,18) float 텐서."""
    n = len(boards)
    x = np.zeros((n, 2, H, W), dtype=np.float32)
    x[:, 0] = boards.astype(np.float32) / 9.0
    for i, (r1, c1, r2, c2) in enumerate(rects):
        x[i, 1, r1:r2+1, c1:c2+1] = 1.0
    return torch.from_numpy(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=512)
    args = p.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    d = np.load(ROOT / "dl" / "data.npz")
    X = build_inputs(d["boards"], d["rects"])
    y = torch.from_numpy(d["targets"].astype(np.float32))

    n = len(y); idx = torch.randperm(n)
    nval = n // 10
    vi, ti = idx[:nval], idx[nval:]
    Xt, yt, Xv, yv = X[ti], y[ti], X[vi], y[vi]

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()

    Xv_d, yv_d = Xv.to(dev), yv.to(dev)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(yt))
        for b in range(0, len(yt), args.bs):
            j = perm[b:b+args.bs]
            xb, yb = Xt[j].to(dev), yt[j].to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xv_d)
            mae = (pv - yv_d).abs().mean().item()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:2d}  val MAE {mae:.2f} (남은액션수 단위)")

    torch.save(net.state_dict(), ROOT / "dl" / "model.pt")
    print(f"→ dl/model.pt 저장  (device={dev})")


if __name__ == "__main__":
    main()
