from agents.utils import format_dataclass_box

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Tuple, Any
from abc import ABC, abstractmethod

from game.action import Action
from game.board import Board

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

    @abstractmethod
    def select_action(self, board: Board) -> Action | None:
        """
        주어진 판에서 둘 수 하나를 고른다. 합법수가 없으면 None.

        run_episode 가 '판을 처음부터 끝까지 스스로 굴리는' 인터페이스라면
        이쪽은 '판이 밖에 있을 때' 쓴다. 실 기기 조작(runs/play_device.py)처럼
        판의 주인이 에이전트가 아닌 경우가 그렇다.

        board 를 바꾸지 않는다. 수를 두는 것은 부르는 쪽의 몫이다.
        """
        pass

    def get_info_formatted(self) -> str:
        return format_dataclass_box("Agent Info", self.get_info())