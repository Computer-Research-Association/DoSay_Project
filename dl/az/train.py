"""
학습 — self-play 데이터(dl/az/data/*.pkl)로 policy+value 망 갱신.
policy: CE(π, softmax(move_logits)),  value: MSE(z, value(s)).
"""
import os, sys, glob, pickle, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from dl.az.net import Net, encode   # noqa: E402


def load_data(max_samples=200000):
    samples = []
    for f in sorted(glob.glob(str(ROOT / "dl" / "az" / "data" / "*.pkl"))):
        with open(f, "rb") as fh:
            samples += pickle.load(fh)
    if len(samples) > max_samples:            # 최근 위주 리플레이 버퍼
        samples = samples[-max_samples:]
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--bs", type=int, default=128)
    args = p.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    data = load_data()
    if not data:
        print("데이터 없음"); return

    net = Net().to(dev)
    mp = ROOT / "dl" / "az" / "model.pt"
    if mp.exists():
        net.load_state_dict(torch.load(mp, map_location=dev))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    n = len(data)
    for ep in range(args.epochs):
        net.train()
        order = np.random.permutation(n)
        ploss_sum = vloss_sum = 0.0
        nb = 0
        for b in range(0, n, args.bs):
            idx = order[b:b + args.bs]
            batch = [data[j] for j in idx]
            X = torch.from_numpy(np.stack([encode(s[0]) for s in batch])).to(dev)
            z = torch.tensor([s[3] for s in batch], dtype=torch.float32, device=dev)
            feat = net.features(X)
            val = net.value(feat)
            vloss = F.mse_loss(val, z)
            # policy: 샘플별 (가변 수) — 배치 feature 재사용
            ploss = 0.0
            for k, s in enumerate(batch):
                moves, pi = s[1], s[2]
                if len(moves) < 2:
                    continue
                logits = net.move_logits(feat[k], moves)
                logp = F.log_softmax(logits, dim=0)
                tgt = torch.tensor(pi, dtype=torch.float32, device=dev)
                ploss = ploss - (tgt * logp).sum()
            ploss = ploss / len(batch)
            loss = vloss + ploss
            opt.zero_grad(); loss.backward(); opt.step()
            ploss_sum += float(ploss.detach()); vloss_sum += float(vloss.detach()); nb += 1
        print(f"epoch {ep}: policy {ploss_sum/nb:.3f}  value {vloss_sum/nb:.4f}")

    torch.save(net.state_dict(), mp)
    print(f"→ 저장 dl/az/model.pt  (샘플 {n})")


if __name__ == "__main__":
    main()
