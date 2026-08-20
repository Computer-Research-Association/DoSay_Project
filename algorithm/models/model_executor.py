import importlib
from pathlib import Path
from abc import ABC
from types import ModuleType
from typing import Tuple, cast
import copy

import numpy as np
from numpy.typing import NDArray

from algorithm.models.utils import HeuristicRegistry
from game.board import Board
from agents.dtos import AlgoInfo  # type: ignore

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
        self.model_info = AlgoInfo("Algorithm", model_type, model_version, self.model_path.relative_to(Path.cwd()))
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
            child_board = copy.deepcopy(self.board)
            child_board.do_action(action)
            return registry.evaluate(child_board)

        best_action = max(actions, key=score_of)
        _, cleared = self.board.do_action(best_action)
        return cleared
    
    
class BeamSearchExecutor(AlgoExecutor):
    def __init__(self, grid_shape: Tuple[int, int], model_path: Path):
        super().__init__(grid_shape, model_path)
    
    def do_game(self):
        pass

class AnnealExecutor(AlgoExecutor):
    """어닐링(best 모델): reset 시 전체 수순을 계산해두고, do_step으로 한 수씩 재생."""
    def __init__(self, grid_shape: Tuple[int, int], model_path: Path):
        super().__init__(grid_shape, model_path)
        self.iters = int(getattr(self.cls, "ITERS", 8000))
        self.instances = int(getattr(self.cls, "INSTANCES", 16))
        self._plan: list = []
        self._ptr = 0

    def _solve(self, grid: NDArray[np.int8]) -> list:
        from game.action import Action
        best_sc, best_seq = -1, []
        try:  # 빠른 경로: 전체-Cython (9x18)
            import sys as _s
            _s.path.insert(0, str(self.model_path.resolve().parents[3] / "bench"))
            from anneal_cy import anneal_cy  # type: ignore
            b = np.ascontiguousarray(grid, dtype=np.int8)
            for j in range(self.instances):
                sc, seq = anneal_cy(b, self.iters, j * 100 + 1)
                if sc > best_sc:
                    best_sc, best_seq = sc, seq
        except Exception:  # 폴백: numba anneal_once
            from models.anneal import anneal_once
            for j in range(self.instances):
                sc, seq = anneal_once(np.asarray(grid), iters=self.iters, rng_seed=j)
                if sc > best_sc:
                    best_sc, best_seq = sc, seq
        return [Action(top_left=(m[0], m[1]), bottom_right=(m[2], m[3])) for m in best_seq]

    def reset(self, board_source):
        board, score = super().reset(board_source)
        self._plan = self._solve(self.board.grid)
        self._ptr = 0
        return board, score

    def do_step(self) -> int:
        if self.board is None:
            raise RuntimeError("Executor reset needed: call reset() before do_step().")
        if self._ptr >= len(self._plan):
            return 0
        action = self._plan[self._ptr]
        self._ptr += 1
        _, cleared = self.board.do_action(action)
        return cleared
