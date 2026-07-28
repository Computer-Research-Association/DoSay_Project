"""
comparison_test.py가 만든 두 결과 JSON을 비교해서:
  1. feature 함수 코드가 바뀐 함수만 diff로 보여주고
  2. run별 weight 값이 바뀐 것만 보여준다
 
comparison_test.py의 자동 실행 로직과 무관하게, 순수하게 파일 두 개만 있으면 동작한다.
 
사용법:
    python check_changes.py old.json new.json
"""
import difflib
import json
import sys
from pathlib import Path
 
 
def load(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "feature_snapshot" not in data:
        print(f"⚠️  경고: {path} 에 'feature_snapshot' 필드가 없음 — "
              f"comparison_test.py로 만든 파일이 맞는지 확인 필요")
    return data
 
 
def check_code_changes(old: dict, new: dict) -> bool:
    """feature 함수 코드가 바뀐 것만 골라서 diff로 보여준다. 하나라도 바뀌었으면 True 반환."""
    old_snap = old.get("feature_snapshot", {})
    new_snap = new.get("feature_snapshot", {})
 
    print("\n" + "=" * 60)
    print("1. 함수 코드 변경 체크")
    print("=" * 60)
 
    if not old_snap or not new_snap:
        print("비교할 feature_snapshot이 비어있음 — 두 JSON 다 comparison_test.py로 만든 게 맞는지 확인")
        return False
 
    all_names = sorted(set(old_snap) | set(new_snap))
    any_changed = False
 
    for name in all_names:
        old_entry = old_snap.get(name)
        new_entry = new_snap.get(name)
 
        if old_entry is None:
            print(f"\n[+ 새로 추가됨] {name}")
            any_changed = True
            continue
        if new_entry is None:
            print(f"\n[- 삭제됨] {name}")
            any_changed = True
            continue
 
        old_hash = old_entry.get("source_hash")
        new_hash = new_entry.get("source_hash")
 
        if old_hash == new_hash:
            continue  # 안 바뀐 건 조용히 넘어감 (바뀐 것만 보여주는 게 목적)
 
        any_changed = True
        print(f"\n[코드 바뀜] {name}  (hash: {old_hash} → {new_hash})")
        old_lines = (old_entry.get("source") or "").splitlines(keepends=True)
        new_lines = (new_entry.get("source") or "").splitlines(keepends=True)
        diff = list(difflib.unified_diff(old_lines, new_lines, fromfile="old", tofile="new"))
        if diff:
            print("".join(diff))
        else:
            print("  (hash는 다른데 텍스트가 같게 보임 — 공백/인코딩 문제 의심, source 필드 직접 확인 필요)")
 
    if not any_changed:
        print("코드가 바뀐 feature 없음")
 
    return any_changed
 
 
def check_weight_changes(old: dict, new: dict) -> bool:
    """run별 weight 값이 바뀐 것만 보여준다. 하나라도 바뀌었으면 True 반환."""
    old_runs = {r["name"]: r.get("weights", {}) for r in old.get("runs", [])}
    new_runs = {r["name"]: r.get("weights", {}) for r in new.get("runs", [])}
 
    print("\n" + "=" * 60)
    print("2. weight 변경 체크")
    print("=" * 60)
 
    if not old_runs or not new_runs:
        print("비교할 runs가 비어있음 — 두 JSON 다 실제로 시뮬레이션을 돌려서 만든 게 맞는지 확인")
        return False
 
    all_names = sorted(set(old_runs) | set(new_runs))
    any_changed = False
 
    for name in all_names:
        if name not in old_runs:
            print(f"\n[+ 새로 추가된 run] {name}: {new_runs[name]}")
            any_changed = True
            continue
        if name not in new_runs:
            print(f"\n[- 삭제된 run] {name}")
            any_changed = True
            continue
 
        old_w, new_w = old_runs[name], new_runs[name]
        if old_w == new_w:
            continue  # 안 바뀐 run은 조용히 넘어감
 
        any_changed = True
        print(f"\n[weight 바뀜] {name}")
        keys = sorted(set(old_w) | set(new_w))
        for k in keys:
            ov = old_w.get(k, "(미지정 → 기본값)")
            nv = new_w.get(k, "(미지정 → 기본값)")
            if ov != nv:
                print(f"  {k}: {ov} → {nv}")
 
    if not any_changed:
        print("weight가 바뀐 run 없음")
 
    return any_changed
 
 
def main():
    if len(sys.argv) != 3:
        print("사용법: python check_changes.py <old.json> <new.json>")
        sys.exit(1)
 
    old_path, new_path = sys.argv[1], sys.argv[2]
    old, new = load(old_path), load(new_path)
 
    print(f"비교: {old_path} ({old.get('generated_at', '?')})  vs  {new_path} ({new.get('generated_at', '?')})")
 
    code_changed = check_code_changes(old, new)
    weight_changed = check_weight_changes(old, new)
 
    print("\n" + "=" * 60)
    print(f"결과: 코드 변경 = {code_changed}, weight 변경 = {weight_changed}")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()