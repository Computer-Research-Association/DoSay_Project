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



########################################################

from dataclasses import fields
from pathlib import Path

def format_value(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, Path):
        return str(value)
    return str(value)


def format_dataclass_box(title: str, obj) -> str:
    """dataclass 필드를 label/value로 정렬해 박스 문자열로 반환"""
    rows = [(f.metadata.get("label", f.name), format_value(getattr(obj, f.name))) for f in fields(obj)]
    label_width = max(len(label) for label, _ in rows)
    lines = [f"{label:<{label_width}} : {value}" for label, value in rows]
    return format_box(title, lines)