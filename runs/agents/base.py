from agents.utils import format_dataclass_box

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Tuple, Any
from abc import ABC, abstractmethod

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
        return format_dataclass_box("Agent Info", self.get_info())