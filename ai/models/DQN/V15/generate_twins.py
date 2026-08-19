"""V15 의 쌍둥이 폴더(V15G / V15H / V15W)의 `model.py` 를 **생성**한다.

    python ai/models/DQN/V15/generate_twins.py            # 생성
    python ai/models/DQN/V15/generate_twins.py --check    # 어긋났는지만 본다 (CI용)

──────────────────────────────────────────────────────────────────────────
왜 스크립트인가
──────────────────────────────────────────────────────────────────────────
쌍둥이는 `V15/model.py` 와 **상수 두 개**만 다르다. 그런데 지금까지는 손으로
복사해 왔고, 그래서 2026-08-13 에 사고가 났다 — `V15` 의 `plan()` 을 고쳤는데
`V15G`/`V15H` 는 옛 구현을 그대로 돌고 있었고, 벤치마크가 그 사실을 모른 채
숫자를 냈다 (search-history.md §7.1).

파일 머리에 "손으로 고치지 말 것" 을 적어 두는 것으로는 부족했다. 고칠 사람이
그 문장을 읽는 시점이 이미 고친 뒤이기 때문이다. **생성으로 바꾸면 어긋날 수가
없고, `--check` 가 어긋난 상태를 실패로 만든다.**

체크포인트는 건드리지 않는다. 쌍둥이는 가중치가 완전히 같고 배포 상수만 다르므로
`model.py` 만 다시 만들면 된다.
"""

import argparse
import io
import sys
from pathlib import Path

VERSION_DIR = Path(__file__).resolve().parent
DQN_DIR = VERSION_DIR.parent
SOURCE = VERSION_DIR / "model.py"

# 폴더 -> (설명 두 줄, {상수: 값})
TWINS: dict[str, tuple[str, dict[str, str]]] = {
    "V15G": ("배포 기록용 (131.77, 25초/판). 폭 1024 + 정책 top-8.",
             {"BEAM_WIDTH": "1024", "POLICY_TOPK": "8"}),
    "V15H": ("예산 상한 측정용 (132.91). 폭 2048 + 정책 top-8.",
             {"BEAM_WIDTH": "2048", "POLICY_TOPK": "8"}),
    "V15W": ("교사 폭 실험용. 학습 전용이므로 top-k 는 0.",
             {"BEAM_WIDTH": "128", "POLICY_TOPK": "0"}),
}


def render(name: str) -> str:
    """`V15/model.py` 에서 쌍둥이 하나의 내용을 만든다."""
    note, constants = TWINS[name]
    body = io.open(SOURCE, encoding="utf-8").read()

    for key, value in constants.items():
        old = next((line for line in body.splitlines()
                    if line.startswith(f"{key} =")), None)
        if old is None:
            raise SystemExit(f"{SOURCE.name} 에 '{key} =' 줄이 없다. 생성기를 고쳐야 한다.")
        # 주석은 그대로 두고 값만 바꾼다
        head, _, comment = old.partition("#")
        new = f"{key} = {value}" + (f"            #{comment}" if comment else "")
        body = body.replace(old, new, 1)

    changed = ", ".join(f"{k}={v}" for k, v in constants.items())
    header = (f'"""DQN {name} — **V15 의 쌍둥이.** 이 파일은 손으로 고치지 않는다.\n'
              f'\n'
              f'    {note}\n'
              f'    구조는 V15 와 같고 {changed} 만 다르다.\n'
              f'\n'
              f'고칠 일이 있으면 `ai/models/DQN/V15/model.py` 를 고치고\n'
              f'`python ai/models/DQN/V15/generate_twins.py` 를 다시 돌린다.\n'
              f'"""\n\n')
    return header + body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="쓰지 않고 어긋난 것만 보고한다 (어긋나면 종료코드 1)")
    args = parser.parse_args()

    stale = []
    for name in TWINS:
        target = DQN_DIR / name / "model.py"
        if not target.parent.is_dir():
            print(f"  건너뜀  {name} (폴더가 없다)")
            continue
        want = render(name)
        have = io.open(target, encoding="utf-8").read() if target.exists() else None
        if have == want:
            print(f"  같음    {name}/model.py")
            continue
        stale.append(name)
        if args.check:
            print(f"  어긋남  {name}/model.py")
        else:
            io.open(target, "w", encoding="utf-8").write(want)
            print(f"  생성    {name}/model.py")

    if args.check and stale:
        print(f"\n{', '.join(stale)} 가 V15 와 어긋나 있다. "
              f"generate_twins.py 를 --check 없이 돌릴 것.")
        return 1
    print("\n쌍둥이가 V15 와 일치한다." if not stale else "\n생성 완료.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
