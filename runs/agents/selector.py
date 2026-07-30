from pathlib import Path
from typing import Callable, Tuple, Type

from agents.base import Agent
from agents.ai.agent import AIAgent
from agents.ai.model_loader import find_checkpoint
from agents.algorithm.agent import AlgoAgent
from agents.utils import select_from

AI_VERSION_DIR = "ai/models/version"


def _find_ai_versions(root: Path) -> list[Path]:
    """체크포인트(.zip)가 들어 있는 버전 폴더만"""
    base = root / AI_VERSION_DIR
    if not base.is_dir():
        return []

    versions = sorted(d for d in base.iterdir() if d.is_dir())
    trained = [d for d in versions if find_checkpoint(d) is not None]

    untrained = [d.name for d in versions if d not in trained]
    if untrained:
        print(f"  (학습 전이라 제외된 버전: {', '.join(untrained)})")

    return trained


def _find_algorithms(root: Path) -> list[Path]:
    return sorted(root.glob("algorithm/models/version/*.py"))


AGENT_REGISTRY: dict[str, tuple[Type[Agent], Callable[[Path], list[Path]]]] = {
    "ai": (AIAgent, _find_ai_versions),
    "algorithm": (AlgoAgent, _find_algorithms),
}


def select_agent(agents_dir: Path) -> Tuple[Type[Agent], Path]:
    types = list(AGENT_REGISTRY.keys())
    agent_type = types[select_from("Select Agent Type", types)]

    agent_cls, find_candidates = AGENT_REGISTRY[agent_type]
    candidates = find_candidates(agents_dir)
    if not candidates:
        hint = (
            f"\n{agents_dir / AI_VERSION_DIR} 아래 버전 폴더에서 train.py 를 먼저 실행해 주세요."
            if agent_type == "ai" else ""
        )
        raise RuntimeError(f"'{agent_type}' 타입에 사용 가능한 에이전트가 없습니다.{hint}")

    labels = [p.name if p.is_dir() else p.stem for p in candidates]
    selected = candidates[select_from(f"Select {agent_type.upper()} Agent", labels)]
    print()

    return (agent_cls, selected)
