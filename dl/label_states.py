"""
DAgger 라벨러 — states.jsonl 의 한 샤드를 어닐링으로 라벨링 (병렬 실행용).
각 상태에서 anneal_from_grid → 도달 가능 추가 칸수 = V 라벨.

사용법(샤드 k / 총 N):
    python dl/label_states.py --shard 0 --nshards 10 --iters 500
출력: dl/_labelparts/{shard}.jsonl  (한 줄 = {"s":[162], "v":값})
"""
import os, sys, json, random, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from algorithm.experiment.anneal import _load_policy   # noqa: E402
from dl.anneal_core import anneal_from_grid            # noqa: E402
H, W = 9, 18


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--nshards", type=int, required=True)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--restarts", type=int, default=3)   # 상태당 어닐링 반복 후 max (라벨 품질↑)
    p.add_argument("--version", default="Anneal_V01b")
    args = p.parse_args()
    _load_policy(args.version)

    states = ROOT / "dl" / "states.jsonl"
    outdir = ROOT / "dl" / "_labelparts"; outdir.mkdir(exist_ok=True)
    out = open(outdir / f"{args.shard}.jsonl", "w")
    rng = random.Random(1000 + args.shard)
    n = 0
    with open(states) as f:
        for i, line in enumerate(f):
            if i % args.nshards != args.shard:
                continue
            s = json.loads(line)["s"]
            grid = np.array(s, dtype=np.int8).reshape(H, W)
            # 보드 채움 정도에 비례해 iter 조절(꽉 찰수록 더 필요)
            filled = int(np.count_nonzero(grid))
            it = max(150, int(args.iters * filled / (H * W)))
            v = max(anneal_from_grid(grid, it, rng) for _ in range(args.restarts))  # max-of-restarts
            out.write(json.dumps({"s": s, "v": v}) + "\n"); n += 1
    out.close()
    print(f"shard {args.shard}: {n} 라벨")


if __name__ == "__main__":
    main()
