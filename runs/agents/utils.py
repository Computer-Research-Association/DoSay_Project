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