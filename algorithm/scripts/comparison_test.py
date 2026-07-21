
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
import diff_features

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_dashboard import build_html  # noqa: E402

def find_project_root(start: Path) -> Path: #simulator.simulator import Simulator의 경로를 올바르게 잡아줌
    for p in [start, *start.parents]:
        if (p / "game").is_dir():
            return p
    raise RuntimeError("프로젝트 루트를 찾을 수 없음 (game/ 폴더 기준)")


# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.simulator.simulator import run_single  # noqa: E402
from algorithm.feature_assistance.feature_spec import FEATURES  # noqa: E402
from algorithm.simulator.simulator import DEFAULT_WEIGHTS
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
# 비교하고 싶은 weight 설정(버전)을 여기에 정의한다.
# 새 feature 버전을 실험할 때마다 이 리스트에 추가하면 됨.
# weights에 없는 feature는 feature_spec.py의 기본 weight(1.0)가 적용된다.
# ---------------------------------------------------------------------------
DEFAULT_RUNS = [
    {"name": "from_simulator_default", "weights": DEFAULT_WEIGHTS},
]

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
    output_dir = output_dir or (PROJECT_ROOT / "results")
    results = []
    output_data = None
    code_snapshot = capture_feature_snapshot()
    git_info = _git_commit_info()
    prev_path = output_dir / "comparison_latest.json"
    if prev_path.exists():
        try:
            prev_data = json.loads(prev_path.read_text(encoding="utf-8"))
            fake_new = {"feature_snapshot": code_snapshot}
            print("\n[저번 실행 대비 feature 코드 변경사항]")
            diff_features.diff_features(prev_data, fake_new)
            print()
        except Exception as e:
            print(f"(이전 결과와 비교 실패, 무시하고 진행: {e})")
    
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] '{run['name']}' 실행 중... ({n_games} games)")
        t0 = time.time()

        # 버전(run)마다 같은 시드로 리셋 -> 모든 weight 설정이 동일한 게임 보드 시퀀스를
        # 마주치게 되어 재현 가능하고 공정한 A/B 비교가 된다.
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
                         help="결과 저장 디렉토리 아래 최종 파일 경로. 지정 안 하면 "
                              "results/comparison_<timestamp>.json (구글 드라이브 동기화 폴더 경로도 가능)")
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
        output_dir = PROJECT_ROOT / "results"
        out_path = output_dir / f"comparison_{int(time.time())}.json"

    output_data = run_comparison(
        runs, args.n_games, seed=seed,
        output_dir=output_dir, live_dashboard=not args.no_live_dashboard,
    )

    # 최종 결과를 타임스탬프 붙은 별도 파일로도 저장
    # (comparison_latest.json / dashboard_latest.html은 run마다 이미 갱신되어 있음)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n최종 저장 완료: {out_path}")
    print(f"(중간 저장본: {output_dir / 'comparison_latest.json'}, "
          f"{output_dir / 'dashboard_latest.html'})")
    return out_path


if __name__ == "__main__":
    main()