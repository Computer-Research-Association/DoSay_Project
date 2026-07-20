from pathlib import Path
from typing import Tuple, Type

from agents.base import Agent
from agents.ai.agent import AIAgent
from agents.algorithm.agent import AlgoAgent
from agents.utils import select_from

AGENTS_DIR = Path(__file__).parent

AGENT_REGISTRY: dict[str, tuple[Type[Agent], str]] = {
    "ai": (AIAgent, "ai/models/*.zip"),
    "algorithm": (AlgoAgent, "algorithm/models/*.py"),
}


def select_agent(agents_dir: Path = AGENTS_DIR) -> Tuple[Type[Agent], Path]:
    types = list(AGENT_REGISTRY.keys())
    agent_type = types[select_from("Select Agent Type", types)]

    agent_cls, pattern = AGENT_REGISTRY[agent_type]
    candidates = sorted(agents_dir.glob(pattern))
    if not candidates:
        raise RuntimeError(f"'{agent_type}' 타입에 사용 가능한 에이전트가 없습니다.")

    model_path = candidates[
        select_from(f"Select {agent_type.upper()} Agent", [p.stem for p in candidates])
    ]
    print()

    return (agent_cls, model_path)