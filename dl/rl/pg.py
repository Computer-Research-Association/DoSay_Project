"""
① 정책경사 RL (REINFORCE + value baseline = A2C).
- 정책망이 게임을 끝까지 샘플링 플레이 → 최종 return으로 학습 (MCTS 없음).
- return G_t = t 이후 제거 칸수 / 162.  advantage A_t = G_t - v(s_t).
- policy loss = -Σ logπ(a_t)·A_t,  value loss = MSE(v, G),  + entropy 보너스.
- AZ 망(dl/az/net.py) 재사용. value warm-start(dl/az/model.pt) 이어받음.

사용: python dl/rl/pg.py --iters 400 --games 48
"""
import os, sys, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from dl.az.game import make_board, legal_moves, apply_move, TOTAL   # noqa: E402
from dl.az.net import Net, encode                                   # noqa: E402


@torch.no_grad()
def play(net, seed, dev, greedy=False):
    net.eval()
    g = make_board(seed)
    traj, cleared = [], 0
    while True:
        mv = legal_moves(g)
        if not mv:
            break
        x = torch.from_numpy(encode(g)).unsqueeze(0).to(dev)
        logits = net.move_logits(net.features(x)[0], mv)
        if greedy:
            i = int(logits.argmax())
        else:
            i = int(np.random.choice(len(mv), p=torch.softmax(logits, 0).cpu().numpy()))
        before = g.copy()
        r = apply_move(g, mv[i])
        traj.append((before, mv, i, r))
        cleared += r
    return traj, cleared


def samples_with_returns(traj):
    """궤적 → [(grid, moves, chosen, G_norm)]. G = 이후 제거칸 누적."""
    G, acc = [], 0
    for (_, _, _, r) in reversed(traj):
        acc += r; G.append(acc)
    G = G[::-1]
    return [(traj[t][0], traj[t][1], traj[t][2], G[t] / TOTAL) for t in range(len(traj))]


def train_batch(net, opt, samples, dev, bs=256, c_v=0.5, c_e=0.002):
    net.train()
    idx = np.random.permutation(len(samples))
    pl = vl = el = 0.0; nb = 0
    for b in range(0, len(samples), bs):
        mb = [samples[j] for j in idx[b:b + bs]]
        X = torch.from_numpy(np.stack([encode(s[0]) for s in mb])).to(dev)
        Gt = torch.tensor([s[3] for s in mb], dtype=torch.float32, device=dev)
        feat = net.features(X)
        v = net.value(feat)
        vloss = F.mse_loss(v, Gt)
        adv = (Gt - v).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)   # advantage 정규화 (작은 신호도 방향 살림)
        ploss = eloss = 0.0
        for k, s in enumerate(mb):
            logits = net.move_logits(feat[k], s[1])
            logp = F.log_softmax(logits, 0)
            ploss = ploss - logp[s[2]] * adv[k]
            eloss = eloss - (logp.exp() * logp).sum()
        ploss = ploss / len(mb); eloss = eloss / len(mb)
        loss = ploss + c_v * vloss - c_e * eloss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        pl += float(ploss.detach()); vl += float(vloss.detach()); el += float(eloss.detach()); nb += 1
    return pl / nb, vl / nb, el / nb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--games", type=int, default=48)
    p.add_argument("--base", type=int, default=50000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--device", default="mps")
    p.add_argument("--arch", default="cnn", choices=["cnn", "attn"])
    args = p.parse_args()
    dev = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"

    if args.arch == "attn":
        from dl.rl.attn_net import AttnNet
        net = AttnNet().to(dev); print("어텐션 정책망 (scratch)")
        ckpt = ROOT / "dl" / "rl" / "pg_attn_model.pt"
        warm = None
    else:
        net = Net().to(dev)
        warm = ROOT / "dl" / "az" / "model.pt"
        ckpt = ROOT / "dl" / "rl" / "pg_model.pt"
    if ckpt.exists():
        net.load_state_dict(torch.load(ckpt, map_location=dev))
    elif warm and warm.exists():
        net.load_state_dict(torch.load(warm, map_location=dev)); print("value warm-start 이어받음")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    seed = args.base
    for it in range(1, args.iters + 1):
        t0 = time.time()
        scores, samples = [], []
        for _ in range(args.games):
            traj, sc = play(net, seed, dev); seed += 1
            scores.append(sc); samples += samples_with_returns(traj)
        pl, vl, el = train_batch(net, opt, samples, dev)
        torch.save(net.state_dict(), ckpt)
        if it % 5 == 0 or it == 1:
            ev = np.mean([play(net, 1234 + i, dev, greedy=True)[1] for i in range(10)])
            print(f"iter {it:3d}  train_avg {np.mean(scores):5.1f}  EVAL(greedy) {ev:5.1f}  "
                  f"ploss {pl:+.3f} vloss {vl:.4f} ent {el:.2f}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
