import time

from agents.base import Agent
from agents.dtos import AlgoInfo

from pathlib import Path
from typing import Tuple

from algorithm.models.model_executor import AlgoExecutor, GreedyExecutor, AnnealExecutor

class AlgoAgent(Agent):
    def __new__(cls, grid_shape: Tuple[int, int], model_path: Path, **kwargs):
        if cls is AlgoAgent:  # 정확히 AlgoAgent로 생성 요청된 경우만
            model_type = model_path.name.split("_")[0]
            cls = AGENT_REGISTRY[model_type]
            return cls(grid_shape, model_path, **kwargs)
        return super().__new__(cls)

    def __init__(self, grid_shape, model_path) -> None:
        super().__init__(grid_shape, model_path)

class HeuristicAgent(AlgoAgent):
    def __init__(self, grid_shape: Tuple[int, int], model_path: Path) -> None:
        super().__init__(grid_shape, model_path)
        self.executor = GreedyExecutor(grid_shape, model_path)

    def get_info(self) -> AlgoInfo:
        return self.executor.model_info

    def run_episode(self, board_source, render, delay):
        board, score = self.executor.reset(board_source)
        steps = 0

        while not board.is_done()[0]:
            cleared = self.executor.do_step()
            steps += 1
            score += cleared

            if render:
                print(f"\n[step {steps:>3}] reward={-1:>5.1f}  score={score}")
                if delay > 0: time.sleep(delay)

        return steps, score


class AnnealAgent(AlgoAgent):
    def __init__(self, grid_shape: Tuple[int, int], model_path: Path) -> None:
        super().__init__(grid_shape, model_path)
        self.executor = AnnealExecutor(grid_shape, model_path)

    def get_info(self) -> AlgoInfo:
        return self.executor.model_info

    def run_episode(self, board_source, render, delay):
        board, score = self.executor.reset(board_source)
        steps = 0
        while not board.is_done()[0]:
            cleared = self.executor.do_step()
            steps += 1
            score += cleared
            if render:
                print(f"\n[step {steps:>3}] score={score}")
                if delay > 0: time.sleep(delay)
        return steps, score


AGENT_REGISTRY = {
    "Greedy": HeuristicAgent,
    "Anneal": AnnealAgent,
}