"""체크포인트 파일명을 (알고리즘)_V(버전)_(스텝) -> V(버전)_(알고리즘)_(스텝) 으로 바꾼다.

학습이 도는 중에도 안전하다. 이미 있는 파일만 바꾸고, 도는 프로세스가 새로
만드는 파일은 건드리지 않는다. 학습이 끝난 뒤 한 번 더 돌리면 된다.

    python runs/rename_checkpoints.py           # 무엇이 바뀔지만 보여준다
    python runs/rename_checkpoints.py --apply   # 실제로 바꾼다
"""

import argparse
import re
from pathlib import Path

import game

ROOT_DIR = Path(game.__file__).resolve().parent.parent
MODEL_ROOT = ROOT_DIR / "ai" / "models"
LOG_ROOT = ROOT_DIR / "ai" / "logs"

OLD_PATTERN = re.compile(r"^(?P<name>[0-9A-Za-z]+)_V(?P<version>[0-9A-Za-z.]+)_(?P<steps>\d+)$")


def new_stem(stem: str) -> str | None:
    m = OLD_PATTERN.match(stem)
    if m is None:
        return None
    return f"V{m.group('version')}_{m.group('name')}_{m.group('steps')}"


def collect() -> list[tuple[Path, Path]]:
    """(원래 경로, 새 경로) 목록. 체크포인트와 텐서보드 런 폴더 둘 다."""
    pairs: list[tuple[Path, Path]] = []

    for path in MODEL_ROOT.glob("*/*/models/*.zip"):
        renamed = new_stem(path.stem)
        if renamed:
            pairs.append((path, path.with_name(f"{renamed}.zip")))

    for path in MODEL_ROOT.glob("*/*/*.zip"):          # models/ 도입 이전 위치
        renamed = new_stem(path.stem)
        if renamed:
            pairs.append((path, path.with_name(f"{renamed}.zip")))

    if LOG_ROOT.is_dir():
        for path in LOG_ROOT.iterdir():
            if not path.is_dir():
                continue
            # 텐서보드 런은 뒤에 _1, _2 가 붙는다
            base, _, suffix = path.name.rpartition("_")
            renamed = new_stem(base) if suffix.isdigit() else new_stem(path.name)
            if renamed:
                pairs.append((path, path.with_name(
                    f"{renamed}_{suffix}" if suffix.isdigit() else renamed)))

    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="체크포인트/로그 파일명 규격 변경")
    parser.add_argument("--apply", action="store_true", help="실제로 이름을 바꾼다")
    args = parser.parse_args()

    pairs = collect()
    if not pairs:
        print("바꿀 파일이 없습니다. (이미 새 규격이거나 파일이 없습니다)")
        return 0

    for old, new in pairs:
        mark = "->"
        if new.exists():
            mark = "!! 이미 있음, 건너뜀"
        print(f"  {old.relative_to(ROOT_DIR)}\n    {mark} {new.name}")

    if not args.apply:
        print(f"\n{len(pairs)}개. 실제로 바꾸려면 --apply 를 붙이세요.")
        return 0

    changed = 0
    for old, new in pairs:
        if new.exists():
            continue
        old.rename(new)
        changed += 1
    print(f"\n{changed}개 변경 완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
