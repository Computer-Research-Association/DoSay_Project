import time

from agents.base import Agent
from agents.dtos import AlgoInfo

from pathlib import Path
from typing import Tuple, Type

from algorithm.models.model_executor import (
    AlgoExecutor, AnnealExecutor, BeamSearchExecutor, GreedyExecutor,
)
from game.action import Action
from game.board import Board


class AlgoAgent(Agent):
    """알고리즘 트랙 에이전트. 판정은 전부 executor 가 갖고, 여기는 껍데기다.

    종류별로 다른 것은 executor 하나뿐이라 run_episode / select_action / get_info 는
    여기 한 곳에만 둔다. 예전에는 종류마다 run_episode 를 복사해 두었는데, 그러면
    한쪽만 고쳐지는 일이 생긴다 (실제로 어닐링 쪽에만 무한 루프 위험이 있었다).
    """

    EXECUTOR: Type[AlgoExecutor]

    def __new__(cls, grid_shape: Tuple[int, int], model_path: Path, **kwargs):
        if cls is AlgoAgent:  # 정확히 AlgoAgent로 생성 요청된 경우만
            model_type = model_path.stem.split("_")[0]
            if model_type not in AGENT_REGISTRY:
                raise KeyError(
                    f"알 수 없는 알고리즘 종류입니다: '{model_type}' ({model_path.name})\n"
                    f"  등록됨: {', '.join(sorted(AGENT_REGISTRY))}\n"
                    f"  파일명 규격은 <종류>_<버전>.py 이고, 종류가 "
                    f"{__file__} 의 AGENT_REGISTRY 에 있어야 합니다.")
            cls = AGENT_REGISTRY[model_type]
            return cls(grid_shape, model_path, **kwargs)
        return super().__new__(cls)

    def __init__(self, grid_shape: Tuple[int, int], model_path: Path) -> None:
        super().__init__(grid_shape, model_path)
        self.executor = self.EXECUTOR(grid_shape, model_path)

    def get_info(self) -> AlgoInfo:
        return self.executor.model_info

    def select_action(self, board: Board) -> Action | None:
        return self.executor.select_action(board)

    def run_episode(self, board_source, render, delay):
        board, score = self.executor.reset(board_source)
        steps = 0

        while not board.is_done()[0]:
            cleared = self.executor.do_step()
            if cleared == 0:
                # 합법수는 반드시 사과를 1개 이상 지운다. 판이 안 끝났는데 0 이
                # 나왔다는 것은 executor 가 수를 못 내고 있다는 뜻이고, 그대로
                # 두면 같은 자리를 영원히 돈다.
                raise RuntimeError(
                    f"{type(self.executor).__name__} 이 {steps}수째에서 멈췄습니다. "
                    f"판에는 아직 합법수가 {len(board.get_valid_actions())}개 있습니다.")
            steps += 1
            score += cleared

            if render:
                print(f"\n[step {steps:>3}] score={score}")
                if delay > 0: time.sleep(delay)

        return steps, score


class HeuristicAgent(AlgoAgent):
    EXECUTOR = GreedyExecutor


class AnnealAgent(AlgoAgent):
    EXECUTOR = AnnealExecutor


class BeamSearchAgent(AlgoAgent):
    EXECUTOR = BeamSearchExecutor


AGENT_REGISTRY: dict[str, Type[AlgoAgent]] = {
    "Greedy": HeuristicAgent,
    "Anneal": AnnealAgent,
    "Beam": BeamSearchAgent,
}
