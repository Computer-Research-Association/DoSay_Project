"""
블랙박스 탐색 총출동 vs 어닐 — 이웃연산자 동일(worst-point), 골격만 다름. 동일 시간.
  TA(Threshold Accepting) / GD(Great Deluge) / Tabu / VNS
사용: python search_zoo.py [budget초] [보드수]
"""
import sys, time, random
from collections import deque
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.board import make_board, apply_move
from models.board_numba import valid_actions_numba as VA
from seq_compare import anneal as anneal_worst, _fewest, _rollout_tail, _pick_worst

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 12


def neighbor(board, cur_S, rng, gp=0.8, forbid=None):
    gg = board.copy(); pcum = [0]
    for mv in cur_S: pcum.append(pcum[-1] + apply_move(gg, mv))
    j = _pick_worst(pcum, rng)
    if forbid:
        for _ in range(4):
            if j not in forbid: break
            j = _pick_worst(pcum, rng)
    g = board.copy(); cleared = 0
    for t in range(j): cleared += apply_move(g, cur_S[t])
    a = VA(g)
    if not a: return None
    choices = [m for m in a if m != cur_S[j]] or a
    mv = choices[rng.randrange(len(choices))]
    cleared += apply_move(g, mv)
    tc, ts = _rollout_tail(g, rng, gp)
    return cleared + tc, cur_S[:j] + [mv] + ts, j


def init_sol(board, rng, gp=0.8):
    acts = VA(board)
    if not acts: return 0, []
    g = board.copy(); c0 = apply_move(g, _fewest(board, acts))
    tc, ts = _rollout_tail(g, rng, gp)
    return c0 + tc, [_fewest(board, acts)] + ts


def local_search(board, S, s, rng, patience):
    noimp = 0
    while noimp < patience:
        nb = neighbor(board, S, rng)
        if nb is None: break
        ns, nS, _ = nb
        if ns > s: s, S, noimp = ns, nS, 0
        else: noimp += 1
    return s, S


def ta(board, seed, budget, T0=3.0):
    rng = random.Random(seed); cur_s, cur_S = init_sol(board, rng); best = cur_s
    t0 = time.time()
    while time.time() - t0 < budget:
        nb = neighbor(board, cur_S, rng)
        if nb is None: continue
        ns, nS, _ = nb
        T = T0 * (1 - (time.time() - t0) / budget)
        if ns >= cur_s - T: cur_s, cur_S = ns, nS
        if ns > best: best = ns
    return best


def gd(board, seed, budget, rain=0.05):
    rng = random.Random(seed); cur_s, cur_S = init_sol(board, rng); best = cur_s
    level = float(cur_s); t0 = time.time()
    while time.time() - t0 < budget:
        nb = neighbor(board, cur_S, rng)
        if nb is None: continue
        ns, nS, _ = nb
        if ns >= level:
            cur_s, cur_S = ns, nS
            level += rain
        if ns > best: best = ns
    return best


def tabu(board, seed, budget, tenure=7, cand=4):
    rng = random.Random(seed); cur_s, cur_S = init_sol(board, rng); best = cur_s
    tl = deque(maxlen=tenure); t0 = time.time()
    while time.time() - t0 < budget:
        bnb = None
        for _ in range(cand):
            nb = neighbor(board, cur_S, rng, forbid=set(tl))
            if nb is None: continue
            if bnb is None or nb[0] > bnb[0]: bnb = nb
        if bnb is None: continue
        ns, nS, j = bnb
        cur_s, cur_S = ns, nS; tl.append(j)
        if ns > best: best = ns
    return best


def vns(board, seed, budget, kmax=5, patience=12):
    rng = random.Random(seed); cur_s, cur_S = init_sol(board, rng)
    cur_s, cur_S = local_search(board, cur_S, cur_s, rng, patience); best = cur_s
    k = 1; t0 = time.time()
    while time.time() - t0 < budget:
        s, S = cur_s, cur_S
        for _ in range(k):
            nb = neighbor(board, S, rng)
            if nb is None: break
            s, S = nb[0], nb[1]
        s, S = local_search(board, S, s, rng, patience)
        if s > cur_s: cur_s, cur_S, k = s, S, 1
        else: k = min(kmax, k + 1)
        if s > best: best = s
    return best


def main():
    VA(make_board(1))
    print(f"=== 9×18 {NB}판 · 탐색 총출동 vs 어닐 동일시간 (budget={BUDGET}s) ===")
    boards = [(400000 + k, make_board(400000 + k)) for k in range(NB)]
    an = np.array([anneal_worst(b, s*7+2, BUDGET, 'worst') for s, b in boards])
    algos = [('어닐 (SA)', None),
             ('Threshold Accept', ta), ('Great Deluge', gd),
             ('Tabu Search', tabu), ('VNS', vns)]
    print(f"{'알고리즘':<20}{'평균':>8}{'중앙':>6}{'min':>5}{'max':>5}{'vs어닐':>8}")
    print(f"{'어닐 (SA)':<20}{an.mean():>8.2f}{int(np.median(an)):>6}{int(an.min()):>5}{int(an.max()):>5}{'—':>8}")
    for name, fn in algos[1:]:
        sc = np.array([fn(b, s*7+3, BUDGET) for s, b in boards])
        d = sc - an
        print(f"{name:<20}{sc.mean():>8.2f}{int(np.median(sc)):>6}{int(sc.min()):>5}{int(sc.max()):>5}{d.mean():>+8.2f}")


if __name__ == "__main__":
    main()
