"""
여러 feature weight 설정(버전)에 대해 시뮬레이션을 N번씩 돌리고,
결과(EvaluateSummary)를 하나의 JSON 파일로 저장한다.

run 하나가 끝날 때마다 (전체 루프가 끝나길 기다리지 않고) 그 시점까지의 누적 결과를
JSON + 대시보드 HTML로 즉시 로컬에 저장한다 -> 중간에 실패해도 이미 끝난 run들은 안전하게 남음.
--output 경로를 구글 드라이브 동기화 폴더 안으로 잡으면, 저장하는 순간 자동으로 클라우드에도 올라감.

이 JSON은 apple-game-dashboard 스킬(generate_dashboard.py)의 입력으로 쓰인다.

사용법:
    python .claude/scripts/comparison_test.py
    python .claude/scripts/comparison_test.py --n-games 200
    python .claude/scripts/comparison_test.py --config my_runs.json
    python .claude/scripts/comparison_test.py --seed -1   # 시드 고정 없이 실행
    python .claude/scripts/comparison_test.py --output "G:\\내 드라이브\\DoSay_results\\comparison.json"
"""
import argparse
import json
import sys
import time
from pathlib import Path
import hashlib
import inspect
import subprocess


def find_project_root(start: Path) -> Path: #simulator.simulator import Simulator의 경로를 올바르게 잡아줌
    for p in [start, *start.parents]:
        if (p / "game").is_dir():
            return p
    raise RuntimeError("프로젝트 루트를 찾을 수 없음 (game/ 폴더 기준)")


# 프로젝트 루트 및 이 스크립트가 있는 폴더를 sys.path에 추가
# (반드시 아래 import들보다 먼저 실행되어야 함 -> 순서 바뀌면 ModuleNotFoundError 남)
PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # generate_dashboard.py/diff_features.py sibling import용

# ---------------------------------------------------------------------------
# 결과를 저장할 기본 폴더. 여기 한 곳만 바꾸면 --output/--output-dir 없이도 항상 이 위치에 저장됨.
# 구글 드라이브 동기화 폴더 경로를 넣으면, 저장할 때마다 자동으로 클라우드에도 올라감.
# 예: DEFAULT_OUTPUT_DIR = Path(r"G:\내 드라이브\DoSay_results")
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = Path(r"G:\내 드라이브\DoSay_results")
from generate_dashboard import build_html  # noqa: E402
import diff_features  # noqa: E402
from algorithm.simulator.simulator import run_single, DEFAULT_WEIGHTS  # noqa: E402
from algorithm.feature_assistance.feature_spec import FEATURES  # noqa: E402


def _git_commit_info() -> dict:
    """현재 git 커밋 해시와 '커밋 안 된 변경사항 있는지'를 확인한다.
    git이 없거나 레포가 아니어도 죽지 않고 None으로 채움."""
    def run(cmd):
        try:
            return subprocess.check_output(
                cmd, cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:  # noqa: BLE001
            return None

    commit = run(["git", "rev-parse", "HEAD"])
    status = run(["git", "status", "--porcelain", "feature/features.py"])
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,  # features.py에 커밋 안 된 수정 있는지
    }


def capture_feature_snapshot() -> dict:
    """지금 이 순간 실제로 쓰이고 있는 feature 함수들의 소스코드를 그대로 캡처한다."""
    snapshot = {}
    for spec in FEATURES:
        try:
            source = inspect.getsource(spec.func)
        except (OSError, TypeError):
            source = None  # 소스를 못 읽는 경우(예: C 확장 함수)는 건너뜀
        snapshot[spec.name] = {
            "default_weight": spec.weight,
            "source": source,
            "source_hash": hashlib.sha256(source.encode("utf-8")).hexdigest()[:12] if source else None,
        }
    return snapshot


# ---------------------------------------------------------------------------
# simulator.py에 있는 DEFAULT_WEIGHTS를 그대로 가져다 씀 -> 거기서 값 하나 바꾸면
# simulator.py 단독 실행이든 이 비교 스크립트든 둘 다 자동으로 반영됨 (별도 설정 파일 없음).
# 다른 weight 조합이랑도 비교하고 싶으면 리스트에 항목을 더 추가하면 됨.
# ---------------------------------------------------------------------------
DEFAULT_RUNS = [
    {"name": "from_simulator_default", "weights": DEFAULT_WEIGHTS},
]


def next_numbered_path(output_dir: Path, prefix: str = "comparison", suffix: str = ".json") -> Path:
    """output_dir 안을 훑어서 prefix_1, prefix_2, ... 중 가장 큰 번호 다음 걸 반환한다.
    (comparison_1.json, comparison_2.json ... 식으로 실행할 때마다 번호가 늘어남)
    """
    existing_numbers = []
    if output_dir.exists():
        for p in output_dir.glob(f"{prefix}_*{suffix}"):
            num_part = p.stem[len(prefix) + 1:]
            if num_part.isdigit():
                existing_numbers.append(int(num_part))
    next_n = max(existing_numbers, default=0) + 1
    return output_dir / f"{prefix}_{next_n}{suffix}"


def save_snapshot(output_data: dict, output_dir: Path, live_dashboard: bool = True) -> None:
    """지금까지 끝난 run들의 누적 결과를 즉시 로컬에 저장한다 (JSON + 대시보드 HTML).

    매 run 직후 호출되므로, 루프 중간에 어떤 run이 실패해도 그 전까지 끝난 run들은
    이미 디스크에 안전하게 남아있다. output_dir이 구글 드라이브 동기화 폴더 안이면
    이 write 자체가 곧 업로드 트리거가 됨 (별도 API 호출 필요 없음).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "comparison_latest.json"
    json_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")

    if live_dashboard:
        html_path = output_dir / "dashboard_latest.html"
        html_path.write_text(build_html(output_data), encoding="utf-8")
        print(f"    → 로컬 저장: {json_path.name}, {html_path.name}")
    else:
        print(f"    → 로컬 저장: {json_path.name}")


def run_comparison(
    runs: list[dict],
    n_games: int,
    seed: int | None = None,
    output_dir: Path | None = None,
    live_dashboard: bool = True,
) -> dict: #runs -> 우리가 돌릴 main algorithm, n_games (횟수)
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    results = []
    output_data = None
    code_snapshot = capture_feature_snapshot()
    git_info = _git_commit_info()

    # 저번에 이 폴더에서 실행한 결과가 있으면, 그때 코드/weight랑 지금 걸 자동으로 비교해서 보여줌
    prev_path = output_dir / "comparison_latest.json"
    if prev_path.exists():
        try:
            prev_data = json.loads(prev_path.read_text(encoding="utf-8"))
            fake_new = {"feature_snapshot": code_snapshot, "runs": runs}  # runs: 이번에 쓸 name+weights
            print("\n[저번 실행 대비 feature 코드 변경사항]")
            diff_features.diff_features(prev_data, fake_new)
            print("\n[저번 실행 대비 weight 변경사항]")
            diff_features.diff_weights(prev_data, fake_new)
            print()
        except Exception as e:  # noqa: BLE001
            print(f"(이전 결과와 비교 실패, 무시하고 진행: {e})")

    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] '{run['name']}' 실행 중... ({n_games} games)")
        t0 = time.time()

        # weight 하나로 n_games 실행하는 책임은 simulator.py의 run_single()이 전담
        # (seed 리셋도 그 안에서 처리)
        summary = run_single(run["weights"], n_games, seed=seed)

        elapsed = time.time() - t0
        print(f"    완료 ({elapsed:.1f}s) | avg_score={summary.avg_score:.2f} "
              f"std={summary.std_score:.2f} clear_rate={summary.clear_rate:.2%}")

        results.append({
            "name": run["name"],
            "weights": run["weights"],
            "summary": {
                "n_games": summary.n_games,
                "max_score": summary.max_score,
                "min_score": summary.min_score,
                "avg_score": float(summary.avg_score),
                "std_score": float(summary.std_score),
                "avg_turn": float(summary.avg_turn),
                "avg_time": float(summary.avg_time),
                "avg_max_score_ratio": float(summary.avg_max_score_ratio),
                "clear_rate": float(summary.clear_rate),
            },
        })

        # run 하나 끝날 때마다 지금까지의 누적 결과를 즉시 저장 (전체 루프 안 기다림)
        output_data = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_games": n_games,
            "git": git_info,
            "feature_snapshot": code_snapshot,
            "runs": results,
        }
        save_snapshot(output_data, output_dir, live_dashboard=live_dashboard)

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Apple Game feature weight 버전 비교")
    parser.add_argument("--n-games", type=int, default=20,
                         help="버전당 시뮬레이션 게임 수 (기본 20, 현재 성능 이슈로 크게 잡으면 오래 걸림)")
    parser.add_argument("--config", type=str, default=None,
                         help="RUNS 리스트를 담은 JSON 파일 경로. 지정 안 하면 DEFAULT_RUNS 사용")
    parser.add_argument("--output", type=str, default=None,
                         help="최종 파일 경로를 직접 지정 (파일명까지). 지정 안 하면 "
                              "--output-dir(또는 DEFAULT_OUTPUT_DIR) 아래 comparison_1.json, "
                              "comparison_2.json ... 식으로 번호가 자동으로 늘어남")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="결과를 저장할 폴더. 지정 안 하면 스크립트 상단의 DEFAULT_OUTPUT_DIR 사용 "
                              "(구글 드라이브 동기화 폴더 경로도 가능)")
    parser.add_argument("--seed", type=int, default=42,
                         help="재현성을 위한 랜덤 시드. 모든 버전이 동일 시드로 리셋되어 공정하게 비교됨 "
                              "(고정하고 싶지 않으면 --seed -1)")
    parser.add_argument("--no-live-dashboard", action="store_true",
                         help="run이 끝날 때마다 대시보드 HTML을 재생성하지 않음 (기본은 매 run마다 재생성)")
    args = parser.parse_args()

    if args.config:
        runs = json.loads(Path(args.config).read_text(encoding="utf-8"))
    else:
        runs = DEFAULT_RUNS

    seed = None if args.seed < 0 else args.seed

    if args.output:
        out_path = Path(args.output)
        output_dir = out_path.parent
    else:
        output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
        out_path = next_numbered_path(output_dir, prefix="comparison", suffix=".json")

    output_data = run_comparison(
        runs, args.n_games, seed=seed,
        output_dir=output_dir, live_dashboard=not args.no_live_dashboard,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n최종 저장 완료: {out_path}")
    print(f"(중간 저장본: {output_dir / 'comparison_latest.json'}, "
          f"{output_dir / 'dashboard_latest.html'})")
    return out_path


if __name__ == "__main__":
    main()