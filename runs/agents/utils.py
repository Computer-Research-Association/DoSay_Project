from dataclasses import fields
from pathlib import Path

BOX_WIDTH = 68

def format_box(title: str, lines: list[str]) -> str:
    candidates = [BOX_WIDTH, len(title) + 4, *(len(l) + 2 for l in lines)]
    width = max(candidates)

    header = f"┌─ {title} " + "─" * (width - len(title) - 3)
    footer = "└" + "─" * (len(header) - 1)
    body = [f"│ {line}" for line in lines]
    return "\n".join([header, *body, footer])

def print_box(title: str, lines: list[str]) -> None:
    print(format_box(title, lines))

def prompt_index(count: int, prompt: str = "> ") -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw) - 1
        print(f"  1~{count} 사이의 숫자를 입력해주세요.")

def select_from(title: str, options: list[str]) -> int:
    print_box(title, [f"[{i}] {opt}" for i, opt in enumerate(options, 1)])
    return prompt_index(len(options))

def prompt_int(label: str, default: int, minimum: int = 1) -> int:
    """Enter 만 누르면 기본값. 1_000_000 / 1,000,000 표기 모두 허용."""
    while True:
        raw = input(f"{label}  (기본 {default:,}) > ").strip().replace(",", "").replace("_", "")
        if not raw:
            return default
        if raw.isdigit() and int(raw) >= minimum:
            return int(raw)
        print(f"  {minimum:,} 이상의 정수를 입력해주세요. (Enter = {default:,})")

def prompt_yes_no(question: str, default: bool = True) -> bool:
    suffix = "(Y/n)" if default else "(y/N)"
    while True:
        raw = input(f"{question} {suffix} > ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  y 또는 n 을 입력해주세요.")


########################################################

def format_value(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, Path):
        return str(value)
    return str(value)


def format_dataclass_box(title: str, obj) -> str:
    """dataclass 필드를 label/value로 정렬해 박스 문자열로 반환.

    값이 None이거나 metadata에 hidden=True인 필드는 건너뛴다.
    (원자료는 JSON에 남기되 화면에는 안 띄우고 싶을 때 hidden을 쓴다.)
    """
    rows = [
        (f.metadata.get("label", f.name), format_value(value))
        for f in fields(obj)
        if not f.metadata.get("hidden") and (value := getattr(obj, f.name)) is not None
    ]
    label_width = max(len(label) for label, _ in rows)
    lines = [f"{label:<{label_width}} : {value}" for label, value in rows]
    return format_box(title, lines)
