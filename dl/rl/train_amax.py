"""
③ 학습 — CNN 정책망이 action-max의 선택을 재현하도록 지도학습 (classification).
eval: net-greedy 점수 (목표 ≈ action-max 118, fewest 112 넘기).
저장: dl/rl/amax_model.pt
"""
import os, sys, pickle, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from dl.az.game import make_board, legal_moves, apply_move   # noqa: E402
from dl.az.net import Net, encode                            # noqa: E402


@torch.no_grad()
def net_greedy(net, seed, dev):
    net.eval(); g = make_board(seed); sc = 0
    while True:
        mv = legal_moves(g)
        if not mv:
            return sc
        logits = net.move_logits(net.features(torch.from_numpy(encode(g)).unsqueeze(0).to(dev))[0], mv)
        sc += apply_move(g, mv[int(logits.argmax())])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=128)
    args = p.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    data = pickle.load(open(ROOT / "dl" / "rl" / "amax_data.pkl", "rb"))
    n = len(data); nval = n // 10
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    for ep in range(args.epochs):
        net.train(); order = np.random.permutation(n - nval) + nval
        tot = 0.0; nb = 0
        for b in range(0, len(order), args.bs):
            mb = [data[j] for j in order[b:b + args.bs]]
            X = torch.from_numpy(np.stack([encode(s[0]) for s in mb])).to(dev)
            feat = net.features(X)
            loss = 0.0
            for k, s in enumerate(mb):
                logits = net.move_logits(feat[k], s[1])
                loss = loss + F.cross_entropy(logits.unsqueeze(0),
                                              torch.tensor([s[2]], device=dev))
            loss = loss / len(mb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        # val top-1 정확도
        net.eval(); correct = 0
        with torch.no_grad():
            for s in data[:nval]:
                logits = net.move_logits(net.features(torch.from_numpy(encode(s[0])).unsqueeze(0).to(dev))[0], s[1])
                correct += int(int(logits.argmax()) == s[2])
        if ep % 5 == 0 or ep == args.epochs - 1:
            ev = np.mean([net_greedy(net, 1234 + i, dev) for i in range(10)])
            print(f"epoch {ep:2d}  loss {tot/nb:.3f}  amax일치 {correct/nval:.2%}  net-greedy {ev:.1f}")
    torch.save(net.state_dict(), ROOT / "dl" / "rl" / "amax_model.pt")
    print("→ dl/rl/amax_model.pt")


if __name__ == "__main__":
    main()
