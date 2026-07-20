from pathlib import Path
from typing import Tuple, Type

from agents.base import Agent
from agents.ai.agent import AIAgent
from agents.algorithm.agent import AlgoAgent

AGENTS_DIR = Path(__file__).parent

AGENT_REGISTRY: dict[str, tuple[Type[Agent], str]] = {
    "ai": (AIAgent, "ai/models/*.zip"),
    "algorithm": (AlgoAgent, "algorithm/models/*.py"),
}

def _prompt_index(prompt: str, count: int) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw) - 1
        print(f"1~{count} 사이의 숫자를 입력해주세요.")


def select_agent(agents_dir: Path = AGENTS_DIR) -> Tuple[Type[Agent], Path]:
    types = list(AGENT_REGISTRY.keys())

    print("== 에이전트 타입 선택 ==")
    for i, t in enumerate(types, 1):
        print(f"  [{i}] {t}")
    agent_type = types[_prompt_index("타입 번호 입력: ", len(types))]

    agent_cls, pattern = AGENT_REGISTRY[agent_type]
    candidates = sorted(agents_dir.glob(pattern))
    if not candidates:
        raise RuntimeError(f"'{agent_type}' 타입에 사용 가능한 에이전트가 없습니다.")

    print(f"\n== {agent_type} 에이전트 선택 ==")
    for i, p in enumerate(candidates, 1):
        print(f"  [{i}] {p.stem}")
    model_path = candidates[_prompt_index("에이전트 번호 입력: ", len(candidates))]

    return (agent_cls, model_path)