import importlib
from pathlib import Path
from abc import ABC
from types import ModuleType
from typing import Tuple, cast

import numpy as np
from numpy.typing import NDArray

from algorithm.models.utils import HeuristicRegistry
from game.board import Board
from agents.dtos import AlgoInfo  # type: ignore

from algorithm.feature_assistance.feature_context import FeatureContext

class AlgoExecutor(ABC): 
    def __init__(self, grid_shape: Tuple[int, int], model_path: Path):
        super().__init__()
        self.grid_shape = grid_shape
        self.model_path = model_path
        self.model_info: AlgoInfo
        self.cls: ModuleType
        self.board: Board
        self._load_model()

    def reset(self, board_source: int | list[int] | NDArray[np.int8]):
        if isinstance(board_source, (np.ndarray, list)):
            _board = np.asarray(board_source, dtype=np.int8)
            self.board = Board.from_board(_board)
            self._score = _board.size - int(np.count_nonzero(_board))
        elif board_source is None or isinstance(board_source, int):
            self.board = Board.from_seed(self.grid_shape, board_source)
            self._score = 0
        else:
            raise TypeError(f"board_source must be ndarray, list, or int, got {type(board_source).__name__}")
        
        return self.board, self._score

    def _load_model(self):
        model_type, model_version = self.model_path.name.split("_")
        self.model_info = AlgoInfo("Algorithm", model_type, model_version, self.model_path)
        self.cls = importlib.import_module(f"algorithm.models.version.{self.model_path.stem}")
        

class GreedyExecutor(AlgoExecutor):
    def __init__(self, grid_shape: Tuple[int, int], model_path: Path):
        super().__init__(grid_shape, model_path)
    
    def do_step(self) -> int:
        if self.board is None:
            raise RuntimeError("Executor reset needed: call reset() before do_step().")

        is_over, _ = self.board.is_done()
        if is_over:
            return 0

        actions = self.board.get_valid_actions()
        registry = cast(HeuristicRegistry, self.cls.registry)

        def score_of(action) -> float:
            ctx = FeatureContext.from_board(self.board.grid, action, valid_actions=actions)
            return registry.evaluate(ctx)

        best_action = max(actions, key=score_of)
        _, cleared = self.board.do_action(best_action)
        return cleared
    
    
class BeamSearchExecutor(AlgoExecutor):
    def __init__(self, grid_shape: Tuple[int, int], model_path: Path):
        super().__init__(grid_shape, model_path)
    
    def do_game(self):
        pass