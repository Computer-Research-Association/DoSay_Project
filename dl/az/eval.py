"""
평가 — 고정 보드에서 MCTS argmax(무noise, 무temperature)로 플레이, 평균 점수.
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from dl.az.game import make_board, legal_moves, apply_move   # noqa: E402
from dl.az.net import Net                                     # noqa: E402
from dl.az.mcts import run_mcts                               # noqa: E402


def play(seed, net, dev, sims):
    g = make_board(seed); score = 0
    while True:
        mv = legal_moves(g)
        if not mv:
            return score
        _, pi = run_mcts(g, net, dev, sims, add_noise=False)
        score += apply_move(g, mv[int(pi.argmax())])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--base", type=int, default=1234)   # 고정 평가 시드 (self-play와 분리)
    p.add_argument("--sims", type=int, default=80)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(2)
    net = Net().to(args.device); net.eval()
    mp = ROOT / "dl" / "az" / "model.pt"
    if mp.exists():
        net.load_state_dict(torch.load(mp, map_location=args.device))
    scores = [play(args.base + i, net, args.device, args.sims) for i in range(args.games)]
    print(f"EVAL avg {np.mean(scores):.2f} (min{int(min(scores))} max{int(max(scores))}) sims={args.sims}")


if __name__ == "__main__":
    main()
