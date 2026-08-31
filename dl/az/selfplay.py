"""
Self-play — MCTS로 게임을 두며 (상태, 방문분포 π, 최종가치 z) 수집.
z(s) = s에서 실제로 추가 제거된 칸수 / 162.
저장: dl/az/data/shard_{tag}.pkl  (append 아님, 샤드별 파일)
"""
import os, sys, time, pickle, argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from dl.az.game import make_board, legal_moves, apply_move, TOTAL   # noqa: E402
from dl.az.net import Net                                            # noqa: E402
from dl.az.mcts import run_mcts                                      # noqa: E402


def play_game(seed, net, device, sims, temp_moves=12):
    g = make_board(seed)
    hist = []           # (grid, moves, pi, cleared_before)
    cleared, move_num = 0, 0
    while True:
        mv = legal_moves(g)
        if not mv:
            break
        _, pi = run_mcts(g, net, device, sims, add_noise=True)
        hist.append((g.copy(), mv, pi, cleared))
        if move_num < temp_moves:
            i = int(np.random.choice(len(mv), p=pi))      # τ=1 탐험
        else:
            i = int(pi.argmax())                          # τ→0
        cleared += apply_move(g, mv[i]); move_num += 1
    samples = [(grid, moves, pi, (cleared - cb) / TOTAL) for (grid, moves, pi, cb) in hist]
    return samples, cleared


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--base", type=int, default=30000)
    p.add_argument("--sims", type=int, default=60)
    p.add_argument("--tag", default="0")
    p.add_argument("--device", default="cpu")   # 병렬 self-play는 CPU (MPS 프로세스 경합 회피)
    args = p.parse_args()
    torch.set_num_threads(1)                      # 프로세스당 1스레드 (병렬 효율)
    dev = args.device
    net = Net().to(dev); net.eval()
    mp = ROOT / "dl" / "az" / "model.pt"
    if mp.exists():
        net.load_state_dict(torch.load(mp, map_location=dev))

    data, scores = [], []
    t0 = time.time()
    for gi in range(args.games):
        s, final = play_game(args.base + gi, net, dev, args.sims)
        data += s; scores.append(final)
    outdir = ROOT / "dl" / "az" / "data"; outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"shard_{args.tag}.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"[{args.tag}] {args.games}판 avg {np.mean(scores):.1f} (min{int(min(scores))} max{int(max(scores))})  "
          f"샘플 {len(data)}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
