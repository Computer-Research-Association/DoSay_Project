"""
수순 어닐링 (Simulated Annealing) — grid 레벨 경량 버전.

- 해 = 한 판의 액션 시퀀스 (액션 = (r1,c1,r2,c2) 튜플).
- 이웃 = 지점 t에서 '다른' 유효 수로 갈아타 → 그 뒤 rollout 재플레이.
- rollout 정책(경량, child board 안 만듦): 9 지우는 수 우선 → 적게 지우는(딱 맞는 짝) 수 → 랜덤.
  (Board 객체/모델 heuristic 미사용 → valid_actions 재계산을 최소화해 빠름)
- 적합도 = 총 제거 칸 수 (정확 계산).
- head 상태 캐싱 (채택 시에만 갱신).

사용법:
    python algorithm/experiment/anneal.py --iters 2000 --games 20
"""
import os
import sys
import json
import time
import math
import random
import argparse
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from game.board import compute_prefix_sum   # noqa: E402  (모듈 함수, Board 클래스 안 씀)

GRID = (9, 18)
TOTAL = GRID[0] * GRID[1]
BASE_SEED = 1234
RESULTS_DIR = ROOT / "results"


def _make_grid(seed):
    return np.random.default_rng(seed).integers(1, 10, size=GRID, dtype=np.int8)

def _valid_actions(grid):
    """grid에서 합=10인 '최소 사각형' 유효 수 목록 (r1,c1,r2,c2 튜플). board.py와 동일 규칙."""
    H, W = grid.shape
    P = compute_prefix_sum(grid)
    def area(r1, c1, r2, c2):
        return int(P[r2+1, c2+1] - P[r1, c2+1] - P[r2+1, c1] + P[r1, c1])
    res = []
    for r1 in range(H):
        for r2 in range(r1, H):
            for c1 in range(W):
                for c2 in range(c1, W):
                    s = area(r1, c1, r2, c2)
                    if s == 10:
                        if area(r1, c1, r1, c2) == 0: continue
                        if area(r1, c2, r2, c2) == 0: continue
                        if area(r2, c1, r2, c2) == 0: continue
                        if area(r1, c1, r2, c1) == 0: continue
                        res.append((r1, c1, r2, c2))
                    elif s > 10:
                        break
    return res

def _apply(grid, a):
    r1, c1, r2, c2 = a
    region = grid[r1:r2+1, c1:c2+1]
    cleared = int(np.count_nonzero(region))
    region[:] = 0
    return cleared

_POLICY = None   # version 파일의 rollout_policy(grid, actions, rng) 로드됨

def _load_policy(version):
    global _POLICY
    import importlib
    _POLICY = importlib.import_module(f"algorithm.models.version.{version}").rollout_policy

def _cheap_pick(grid, actions, rng, greedy_prob):
    """확률적으로 랜덤, 아니면 로드된 정책(rollout_policy)으로 선택."""
    if rng.random() >= greedy_prob:
        return actions[rng.randrange(len(actions))]
    return _POLICY(grid, actions, rng)

def _rollout(grid, first_action, rng, greedy_prob):
    """first_action 두고 끝까지 플레이. (액션 시퀀스, 총 제거) 반환. grid는 소모됨."""
    seq, total = [], 0
    a = first_action
    while a is not None:
        total += _apply(grid, a)
        seq.append(a)
        acts = _valid_actions(grid)
        if not acts:
            break
        a = _cheap_pick(grid, acts, rng, greedy_prob)
    return seq, total

def _build_prefix(seed, actions):
    """각 시점 (grid 스냅샷, 누적 제거) 캐싱."""
    grid = _make_grid(seed)
    pgrid = [grid.copy()]
    pclear = [0]
    acc = 0
    for a in actions:
        acc += _apply(grid, a)
        pgrid.append(grid.copy())
        pclear.append(acc)
    return pgrid, pclear


def _pick_t(n, mode, rng, pc=None):
    """이웃 지점 t 선택."""
    if mode == "front":
        return int(n * (rng.random() ** 2))
    if mode == "back":
        return n - 1 - int(n * (rng.random() ** 2))
    if mode == "worst" and pc is not None:
        # 각 수의 제거 칸 수 → 2칸 초과(작은 수 낭비)일수록 우선 수정
        w = [(pc[i + 1] - pc[i] - 2) ** 2 + 0.1 for i in range(n)]
        tot = sum(w); r = rng.random() * tot; acc = 0.0
        for i, wi in enumerate(w):
            acc += wi
            if acc >= r:
                return i
        return n - 1
    return rng.randrange(n)

def _anneal_loop(seed, iters, greedy_prob, T0, neighbor, rng, cur_seq, cur_score, patience=0):
    """단일 어닐링 루프. (best_score, best_turns, iters_done) 반환.
    patience>0 이면 그만큼 개선 없을 때 조기 종료(수렴 감지)."""
    best_score, best_turns = cur_score, len(cur_seq)
    pg, pc = _build_prefix(seed, cur_seq)
    no_improve = 0
    it = 0
    for it in range(iters):
        T = T0 * (1 - it / iters) + 1e-6
        if len(cur_seq) < 2:
            break
        t = _pick_t(len(cur_seq), neighbor, rng, pc)
        base_g = pg[t]
        acts = _valid_actions(base_g)
        if len(acts) < 2:
            continue
        cur_a = cur_seq[t]
        alt = [a for a in acts if a != cur_a]
        if not alt:
            continue

        if neighbor == "multi":                       # 여러 tail 시도 후 최고
            tail_seq, tail_total = None, -1
            for _ in range(3):
                nf = alt[rng.randrange(len(alt))]
                ts, tt = _rollout(base_g.copy(), nf, rng, greedy_prob)
                if tt > tail_total:
                    tail_seq, tail_total = ts, tt
        else:
            gp = 0.4 if neighbor == "ruin" else greedy_prob   # ruin=더 랜덤 재구성
            nf = alt[rng.randrange(len(alt))]
            tail_seq, tail_total = _rollout(base_g.copy(), nf, rng, gp)

        new_score = pc[t] + tail_total
        delta = new_score - cur_score
        if delta >= 0 or rng.random() < math.exp(delta / T):
            cur_seq = cur_seq[:t] + tail_seq
            cur_score = new_score
            pg, pc = _build_prefix(seed, cur_seq)
            if cur_score > best_score:
                best_score, best_turns = cur_score, len(cur_seq)
                no_improve = 0
                continue
        no_improve += 1
        if patience and no_improve >= patience:   # 수렴 → 조기 종료
            break
    return best_score, best_turns, it + 1

def _anneal_one(arg):
    seed, iters, greedy_prob, T0, neighbor, n_restarts, restart_gp, rng_seed = arg
    rng = random.Random(rng_seed)   # 보드는 seed, 어닐링 랜덤성은 rng_seed (병렬 다양성용)

    def fresh(gp_init):
        grid = _make_grid(seed)
        acts = _valid_actions(grid)
        first = _cheap_pick(grid, acts, rng, gp_init) if acts else None
        return _rollout(grid, first, rng, gp_init) if first else ([], 0)

    if neighbor == "restart":
        per = max(1, iters // n_restarts)
        init_score, best_score, best_turns = None, 0, 0
        for _ in range(n_restarts):
            cur_seq, cur_score = fresh(restart_gp)   # 다양성 위해 랜덤 섞은 초기해
            if init_score is None:
                init_score = cur_score
            bs, bt, _ = _anneal_loop(seed, per, greedy_prob, T0, "single", rng, cur_seq, cur_score)
            if bs > best_score:
                best_score, best_turns = bs, bt
        return init_score, best_score, best_turns

    if neighbor == "adaptive":
        # 예산(iters) 소진까지: fresh 시작 → worst 이웃 + 수렴감지 → 수렴하면 재시작.
        # 어려운/쉬운 판(빨리 수렴)=재시작 많이(표본↑), 중간 판(개선 지속)=깊게.
        patience = max(200, iters // 6)
        init_score, best_score, best_turns = None, 0, 0
        used = 0
        while used < iters:
            cur_seq, cur_score = fresh(1.0)
            if init_score is None:
                init_score = cur_score
            bs, bt, done = _anneal_loop(seed, iters - used, greedy_prob, T0,
                                        "worst", rng, cur_seq, cur_score, patience=patience)
            used += done
            if bs > best_score:
                best_score, best_turns = bs, bt
        return init_score, best_score, best_turns

    # single / front / back / worst / ruin / multi — t 분포·재구성만 다름
    cur_seq, cur_score = fresh(1.0)
    best_score, best_turns, _ = _anneal_loop(seed, iters, greedy_prob, T0, neighbor, rng, cur_seq, cur_score)
    return cur_score, best_score, best_turns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="Anneal_V01a")  # rollout 정책 버전
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--greedy-prob", type=float, default=0.8)
    p.add_argument("--T0", type=float, default=3.0)
    p.add_argument("--neighbor", default="single",
                   choices=["single", "front", "back", "restart", "worst", "ruin", "multi", "adaptive"])
    p.add_argument("--restarts", type=int, default=4)      # restart 이웃: 재시작 횟수
    p.add_argument("--restart-gp", type=float, default=0.7)  # restart 초기해 greedy_prob
    p.add_argument("--seed", type=int, default=None)   # 지정 시 그 시드 1판만 (병렬 드라이버용)
    p.add_argument("--instance", type=int, default=0)  # 같은 보드 병렬 실행 시 rng 다양성 인덱스
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    _load_policy(args.version)         # rollout 정책 로드
    cfg = (args.iters, args.greedy_prob, args.T0, args.neighbor, args.restarts, args.restart_gp)

    # 단판 모드: multiprocessing 없이 한 판만. 점수 하나 출력 (xargs 병렬용)
    # rng_seed = seed*1000 + instance → 보드는 seed 고정, 어닐링 랜덤성만 instance별로 다름
    if args.seed is not None:
        _, best, _turns = _anneal_one((args.seed, *cfg, args.seed * 1000 + args.instance))
        print(best)
        return

    # 다판 모드: 순차 실행 (Pool 미사용 → macOS fork+numpy 데드락 원천 차단)
    t0 = time.perf_counter()
    results = [_anneal_one((BASE_SEED + i, *cfg, BASE_SEED + i)) for i in range(args.games)]
    dt = time.perf_counter() - t0

    init = np.array([r[0] for r in results])
    best = np.array([r[1] for r in results])
    turns = np.array([r[2] for r in results])
    if args.quiet:
        print(f"{best.mean():.4f}")
        return
    print(f"어닐링 iters={args.iters} greedy_prob={args.greedy_prob} T0={args.T0}, {args.games}판")
    print(f" 초기(경량greedy) 평균 : {init.mean():.2f}")
    print(f" 어닐링 후 평균        : {best.mean():.2f}  ({best.mean()-init.mean():+.2f})")
    print(f" ratio                : {best.mean()/TOTAL:.3f}")
    print(f" time                 : {dt:.0f}s  ({dt/args.games:.1f}s/판)")

    _save(args, init, best, turns, dt)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None

def _save(args, init, best, turns, dt):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.version}_i{args.iters}"
    out = RESULTS_DIR / f"{tag}.json"
    out.write_text(json.dumps({
        "version": tag,
        "policy": args.version,
        "method": "annealing",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        "n_games": len(best),
        "base_seed": BASE_SEED,
        "config": {
            "iters": args.iters, "greedy_prob": args.greedy_prob, "T0": args.T0,
            "policy": args.version, "neighbor": "single-point-swap+replay",
        },
        "metrics": {
            "avg_score": round(float(best.mean()), 2),
            "std_score": round(float(best.std()), 2),
            "min_score": int(best.min()),
            "max_score": int(best.max()),
            "avg_ratio": round(float(best.mean()) / TOTAL, 3),
            "avg_turns": round(float(turns.mean()), 2),
            "clear_rate": round(float((best == TOTAL).mean()), 3),
            "init_greedy_avg": round(float(init.mean()), 2),
            "sec_per_game": round(dt / len(best), 1),
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f" → 저장: results/{out.name}")


if __name__ == "__main__":
    main()
