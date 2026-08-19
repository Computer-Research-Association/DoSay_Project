"""`search_bench.py --shard i/n` 으로 나눠 돌린 결과를 하나로 합친다.

    python runs/merge_shards.py results/search_cut-all_V15_DQN_50000_seed1234_shard*.json

──────────────────────────────────────────────────────────────────────────
왜 나눠 도는가
──────────────────────────────────────────────────────────────────────────
복구 빔은 폭 64 라 자식이 512개뿐이고, 그 크기로는 어느 가속기든 논다. 판끼리는
완전히 독립이므로 프로세스를 여러 개 띄우면 처리량이 거의 선형으로 붙는다.
**판당 시간은 그대로**이고 실험 한 번의 벽시계 시간만 준다 (100판 x 56초 = 93분).

VRAM 만 보면 된다. 첫 빔(W=1024/top-8)이 봉우리라 프로세스 수만큼 곱해지므로,
8GB 에서는 `--eval-chunk` 를 낮추거나 프로세스를 2~4개로 두는 편이 안전하다.

──────────────────────────────────────────────────────────────────────────
합치는 규칙
──────────────────────────────────────────────────────────────────────────
조각들이 **같은 실험**인지 먼저 확인한다 (체크포인트·프리셋·예산·손잡이·base_seed).
하나라도 다르면 죽는다 — 서로 다른 설정의 판을 섞어 평균 내면 그 숫자는
아무 뜻이 없고, 그런 사고는 조용히 일어난다.

판이 겹치거나 빠져도 죽는다. 100판 실험이라고 적힌 표에 97판짜리가 섞이면
같은 seed 집합으로 비교한다는 이 트랙의 전제가 무너진다.
"""

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"

import numpy as np

BEAM_BASELINE = 131.77

# 이 값들이 조각마다 같아야 한다
IDENTITY = ("source_path", "preset", "engine", "base_seed", "uses_neural",
            "budget_limit", "knobs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+",
                        help="조각 JSON 경로 또는 글롭 (셸이 안 풀어 주면 따옴표로)")
    parser.add_argument("--expect", type=int, default=0, metavar="N",
                        help="합친 뒤 판 수가 N 이어야 한다 (다르면 실패). 0 이면 확인 안 함")
    args = parser.parse_args()

    paths = sorted({Path(p) for pattern in args.patterns for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"조각을 찾지 못했습니다: {args.patterns}")

    shards = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "seeds" not in data:
            raise SystemExit(f"{path.name} 에 seeds 가 없습니다 "
                             f"(--shard 로 만든 파일이 아닙니다)")
        shards.append((path, data))

    # 1. 같은 실험인가
    head = shards[0][1]
    for path, data in shards[1:]:
        for key in IDENTITY:
            if data.get(key) != head.get(key):
                raise SystemExit(f"조각이 서로 다른 실험입니다.\n"
                                 f"  {shards[0][0].name}: {key} = {head.get(key)!r}\n"
                                 f"  {path.name}: {key} = {data.get(key)!r}")
        if data["config"] != head["config"]:
            raise SystemExit(f"{path.name} 의 탐색 설정이 다릅니다.")

    # 2. 판이 겹치거나 빠지지 않았는가
    rows: dict[int, tuple[int, int]] = {}
    for path, data in shards:
        for seed, score, init in zip(data["seeds"], data["scores"], data["init_scores"]):
            if seed in rows:
                raise SystemExit(f"seed {seed} 가 두 조각에 들어 있습니다 ({path.name}).")
            rows[seed] = (score, init)

    seeds = sorted(rows)
    gaps = [s for s in range(seeds[0], seeds[-1] + 1) if s not in rows]
    if gaps:
        raise SystemExit(f"seed 가 {len(gaps)}개 비어 있습니다: "
                         f"{gaps[:10]}{' ...' if len(gaps) > 10 else ''}")
    if args.expect and len(seeds) != args.expect:
        raise SystemExit(f"판이 {len(seeds)}개입니다 (--expect {args.expect}).")

    s = np.array([rows[k][0] for k in seeds], dtype=float)
    b = np.array([rows[k][1] for k in seeds], dtype=float)
    d = s - b
    # 조각들이 동시에 돌았으므로 벽시계는 가장 오래 걸린 조각이 곧 실험 시간이다
    wall = max(x["elapsed_sec"] for _, x in shards)
    total_cpu = sum(x["elapsed_sec"] for _, x in shards)

    print(f"조각 {len(shards)}개 -> {len(seeds)}판 "
          f"(seed {seeds[0]}~{seeds[-1]})   {head['preset']} / {Path(head['source_path']).stem}")
    print(f"손잡이: {head.get('knobs')}\n")
    print(f"{'':<12}{'점수':>9}{'std':>8}")
    print(f"{'빔만':<12}{b.mean():>9.2f}{b.std(ddof=1):>8.2f}")
    print(f"{head['label']:<12}{s.mean():>9.2f}{s.std(ddof=1):>8.2f}")
    print(f"\n국소탐색의 이득  {d.mean():+.2f} ± {d.std(ddof=1)/np.sqrt(len(d)):.2f} (SE)"
          f"   개선 {int((d>0).sum())}판 / 악화 {int((d<0).sum())}판")
    print(f"기준선 {BEAM_BASELINE} 대비  {s.mean()-BEAM_BASELINE:+.2f}")
    print(f"판당 {total_cpu/len(seeds):.1f}s (프로세스 합)   "
          f"벽시계 {wall/60:.1f}분 ({len(shards)}개 동시)")

    merged = {**head,
              "agent": head["agent"] + f" [{len(shards)}조각 합침]",
              "episodes": len(seeds), "shard": None,
              "seeds": seeds,
              "avg_score": round(float(s.mean()), 2),
              "std_score": round(float(s.std(ddof=1)), 2),
              "avg_init_score": round(float(b.mean()), 2),
              "avg_gain": round(float(d.mean()), 2),
              "best_score": int(s.max()), "worst_score": int(s.min()),
              "elapsed_sec": round(total_cpu, 1),
              "wall_sec": round(wall, 1),
              "sec_per_episode": round(total_cpu / len(seeds), 3),
              "merged_from": [p.name for p, _ in shards],
              "measured_at": datetime.now().isoformat(timespec="seconds"),
              "scores": [int(rows[k][0]) for k in seeds],
              "init_scores": [int(rows[k][1]) for k in seeds]}
    for key in ("avg_iterations", "avg_init_sec", "avg_destroy_sec", "avg_repair_sec"):
        merged[key] = round(float(np.mean([x[key] for _, x in shards])), 2)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / (f"search_{head['preset']}_{Path(head['source_path']).stem}_"
                        f"seed{head['base_seed']}_merged_"
                        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
