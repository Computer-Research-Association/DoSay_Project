#!/usr/bin/env python3
"""
comparison_test.py가 만든 두 결과 JSON을 비교해서, 그 사이에
feature 함수 코드/weight가 정확히 뭐가 바뀌었는지 보여준다.

git log를 뒤질 필요 없이 "예전 결과 vs 지금 결과"만 있으면 바로 확인 가능
(각 JSON 안에 그 시점의 feature 함수 소스코드가 그대로 박혀있기 때문).

사용법:
    python diff_features.py old_comparison.json new_comparison.json
"""
import difflib
import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def print_header(text: str):
    print(f"\n{'=' * 60}\n{text}\n{'=' * 60}")


def diff_git(old: dict, new: dict):
    old_git, new_git = old.get("git") or {}, new.get("git") or {}
    if old_git.get("commit") != new_git.get("commit"):
        print(f"git 커밋: {old_git.get('commit', '?')[:8]} → {new_git.get('commit', '?')[:8]}")
    if old_git.get("dirty") or new_git.get("dirty"):
        print("⚠️  커밋 안 된 변경사항이 있는 상태에서 실행됨 (feature/features.py 기준) "
              "— 결과가 실제 git 히스토리와 정확히 안 맞을 수 있음")


def diff_features(old: dict, new: dict):
    old_snap = old.get("feature_snapshot", {})
    new_snap = new.get("feature_snapshot", {})
    all_names = sorted(set(old_snap) | set(new_snap))

    added = [n for n in all_names if n not in old_snap]
    removed = [n for n in all_names if n not in new_snap]
    common = [n for n in all_names if n in old_snap and n in new_snap]
    changed = [n for n in common if old_snap[n].get("source_hash") != new_snap[n].get("source_hash")]
    unchanged = [n for n in common if n not in changed]

    if added:
        print_header(f"새로 추가된 feature ({len(added)}개)")
        for n in added:
            print(f"  + {n}  (weight={new_snap[n].get('default_weight')})")

    if removed:
        print_header(f"삭제된 feature ({len(removed)}개)")
        for n in removed:
            print(f"  - {n}")

    if changed:
        print_header(f"코드가 바뀐 feature ({len(changed)}개)")
        for n in changed:
            old_src = (old_snap[n].get("source") or "").splitlines(keepends=True)
            new_src = (new_snap[n].get("source") or "").splitlines(keepends=True)
            print(f"\n--- {n} ---")
            diff_lines = list(difflib.unified_diff(
                old_src, new_src, fromfile=f"old/{n}", tofile=f"new/{n}"
            ))
            if diff_lines:
                print("".join(diff_lines))
            else:
                print("  (소스 텍스트는 같은데 hash가 다름 — 인코딩/공백 문제일 수 있음, 확인 필요)")

    if unchanged:
        print_header(f"변경 없음 ({len(unchanged)}개)")
        print("  " + ", ".join(unchanged))


def diff_performance(old: dict, new: dict):
    old_runs = {r["name"]: r["summary"] for r in old.get("runs", [])}
    new_runs = {r["name"]: r["summary"] for r in new.get("runs", [])}
    common = sorted(set(old_runs) & set(new_runs))

    if not common:
        return

    print_header("성능 비교 (같은 run 이름 기준)")
    print(f"{'run':<28}{'avg_score':>12}{'std_score':>12}{'clear_rate':>12}")
    for name in common:
        o, n = old_runs[name], new_runs[name]
        d_avg = n["avg_score"] - o["avg_score"]
        print(f"{name:<28}{n['avg_score']:>8.2f}({d_avg:+.2f}) "
              f"{n['std_score']:>8.2f}    {n['clear_rate']:>8.2%}")


def main():
    if len(sys.argv) != 3:
        print("사용법: python diff_features.py <old_comparison.json> <new_comparison.json>")
        sys.exit(1)

    old, new = load(sys.argv[1]), load(sys.argv[2])

    print(f"비교 대상: {sys.argv[1]} ({old.get('generated_at', '?')})"
          f" vs {sys.argv[2]} ({new.get('generated_at', '?')})")

    diff_git(old, new)
    diff_features(old, new)
    diff_performance(old, new)


if __name__ == "__main__":
    main()