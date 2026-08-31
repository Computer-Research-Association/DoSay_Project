"""
모든 모델 best 버전을 m10big 1000판에 돌려 (점수 + 수순) 기록.
샤드: idx % nshards == shard 인 보드만 처리 (import/JIT 1회 상각).
모델: random / greedy / beam / mcts / ga / anneal(numba) / dl(AZ net policy-greedy)
출력: shard_{k}.jsonl — 한 줄 = {"idx":i, "results":{model:{"score":s,"seq":[[r1,c1,r2,c2],...]}}}
사용: python eval_shard.py --shard 0 --nshards 10
"""
import os, sys, json, math, random, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

import models.board as B
from models.board_numba import valid_actions_numba as VA
# 모든 모델이 numba valid_actions 쓰게 패치
import models.greedy, models.beam, models.mcts, models.ga
B.valid_actions = VA
for m in (models.greedy, models.beam, models.mcts, models.ga):
    m.valid_actions = VA
from models.board import make_board, apply_move, cells_of, heuristic, TOTAL
from models.beam import beam_choose
from models.ga import ga as ga_run
from models.anneal import deploy as anneal_deploy


# ---------- 각 모델 play → (score, seq) ----------
def play_random(board, rng):
    g = board.copy(); seq = []; sc = 0
    while True:
        a = VA(g)
        if not a: break
        mv = a[rng.randrange(len(a))]; sc += apply_move(g, mv); seq.append(mv)
    return sc, seq


def play_greedy(board):
    g = board.copy(); seq = []; sc = 0
    while True:
        acts = VA(g)
        if not acts: break
        best, bh = acts[0], -1e9
        for a in acts:
            r1, c1, r2, c2 = a
            ch = g.copy(); ch[r1:r2 + 1, c1:c2 + 1] = 0
            h = heuristic(ch)
            if h > bh: bh, best = h, a
        sc += apply_move(g, best); seq.append(best)
    return sc, seq


def play_beam(board, w=5, d=3):
    g = board.copy(); seq = []; sc = 0
    while True:
        if not VA(g): break
        a = beam_choose(g, w, d)
        if a is None: break
        sc += apply_move(g, a); seq.append(a)
    return sc, seq


# --- MCTS 수 선택 (move-by-move) ---
class _N:
    __slots__ = ("grid", "score", "mv", "untried", "children", "N", "Q")
    def __init__(self, grid, score, mv=None):
        self.grid = grid; self.score = score; self.mv = mv
        self.untried = VA(grid); self.children = []; self.N = 0; self.Q = score

def _rollout(grid, rng):
    t = 0
    while True:
        a = VA(grid)
        if not a: return t
        mv = min(a, key=lambda m: cells_of(grid, m)) if rng.random() < 0.8 else a[rng.randrange(len(a))]
        t += apply_move(grid, mv)

def _mcts_pick(grid, sims, rng, c=0.7):
    root = _N(grid.copy(), 0)
    if not root.untried and not root.children:
        return None
    for _ in range(sims):
        node, path = root, [root]
        while not node.untried and node.children:
            lg = math.log(node.N + 1)
            node = max(node.children, key=lambda x: x.Q / TOTAL + c * math.sqrt(lg / (x.N + 1e-9)))
            path.append(node)
        if node.untried:
            a = node.untried.pop(rng.randrange(len(node.untried)))
            cg = node.grid.copy(); cl = apply_move(cg, a)
            ch = _N(cg, node.score + cl, a); node.children.append(ch); path.append(ch); node = ch
        leaf = node.score + _rollout(node.grid.copy(), rng)
        for nd in path:
            nd.N += 1
            if leaf > nd.Q: nd.Q = leaf
    return max(root.children, key=lambda x: x.N).mv if root.children else None

def play_mcts(board, sims, rng):
    g = board.copy(); seq = []; sc = 0
    while True:
        if not VA(g): break
        a = _mcts_pick(g, sims, rng)
        if a is None: break
        sc += apply_move(g, a); seq.append(a)
    return sc, seq


# --- DL: AZ net policy-greedy ---
_dl = {}
def _load_dl():
    import torch
    from dl.az.net import Net, encode
    dev = "cpu"
    net = Net().to(dev); net.load_state_dict(torch.load(ROOT / "dl" / "az" / "model.pt", map_location=dev)); net.eval()
    _dl.update(torch=torch, net=net, encode=encode, dev=dev)

def play_dl(board):
    if not _dl: _load_dl()
    torch = _dl["torch"]; net = _dl["net"]; encode = _dl["encode"]; dev = _dl["dev"]
    g = board.copy(); seq = []; sc = 0
    while True:
        acts = VA(g)
        if not acts: break
        with torch.no_grad():
            feat = net.features(torch.from_numpy(encode(g)).unsqueeze(0).to(dev))
            logits = net.move_logits(feat[0], acts)
            a = acts[int(logits.argmax())]
        sc += apply_move(g, a); seq.append(a)
    return sc, seq


def _tup(seq):
    return [[int(x) for x in mv] for mv in seq]


def eval_board(board, rng):
    out = {}
    s, q = play_random(board, rng);          out["random"] = {"score": s, "seq": _tup(q)}
    s, q = play_greedy(board);               out["greedy"] = {"score": s, "seq": _tup(q)}
    s, q = play_beam(board, 5, 3);           out["beam"]   = {"score": s, "seq": _tup(q)}
    s, q = play_mcts(board, 150, rng);       out["mcts"]   = {"score": s, "seq": _tup(q)}
    s, q = ga_run(board, pop=30, gens=40, rng=random.Random(rng.randrange(1 << 30)))
    out["ga"] = {"score": int(s), "seq": _tup(q)}
    s, q = anneal_deploy(board, iters=5000, instances=8)
    out["anneal"] = {"score": int(s), "seq": _tup(q)}
    try:
        s, q = play_dl(board);               out["dl_az"] = {"score": s, "seq": _tup(q)}
    except Exception as e:
        out["dl_az"] = {"score": -1, "seq": [], "err": str(e)[:80]}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--nshards", type=int, default=10)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()
    VA(make_board(1))  # JIT 워밍업
    boards = np.load(ROOT / "testcases" / "m10big" / "boards.npy")
    D = ROOT / "testcases" / "allmodels"; D.mkdir(exist_ok=True)

    if args.test:
        r = eval_board(boards[0], random.Random(0))
        for k, v in r.items():
            print(f"  {k:8s}: {v['score']}  ({len(v['seq'])}수)")
        return

    out = open(D / f"shard_{args.shard}.jsonl", "w")
    for i in range(len(boards)):
        if i % args.nshards != args.shard:
            continue
        res = eval_board(boards[i], random.Random(1000 + i))
        out.write(json.dumps({"idx": i, "results": res}) + "\n"); out.flush()
    out.close()


if __name__ == "__main__":
    main()
