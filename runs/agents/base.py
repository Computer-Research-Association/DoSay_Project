from agents.utils import format_box
from ai.envs.apple_env import AppleGameEnv
from game.board import Board

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import fields

class Agent(ABC):
    def __init__(self, grid_shape: Tuple[int, int], model_path: Path) -> None:
        super().__init__()

    @abstractmethod
    def get_info(self) -> Any:
        """
        에이전트에 대한 정보 반환.
        """
        pass

    @abstractmethod
    def run_episode(self, board_source: int|list[int]|NDArray[np.int8], render: bool, delay: float) -> Tuple[int, int]:
        """
        에피소드 1회 실행. (steps, score) 반환.

        board       : Board 객체
        render      : True면 매 수 기보 출력
        delay       : 기보 출력 시 수 간 대기(초). render=False면 무시.
        """
        pass

    def get_info_formatted(self) -> str:
        info = self.get_info()
        rows = [(f.metadata.get("label", f.name), _format_value(getattr(info, f.name))) for f in fields(info)]

        label_width = max(len(label) for label, _ in rows)
        lines = [f"{label:<{label_width}} : {value}" for label, value in rows]

        return format_box("Agent Info", lines)

def _format_value(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)