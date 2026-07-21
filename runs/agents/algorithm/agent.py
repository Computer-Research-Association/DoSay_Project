from game.board import Board
from agents.base import Agent

from ai.envs.apple_env import AppleGameEnv

from algorithm.evaluator.evaluator import pick_best_action

import time
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from typing import cast, Tuple
from dataclasses import dataclass, field


@dataclass
class AlgoInfo:
    algorithm_name: str
    agent_version: float
    beam_width: int  # 빔서치 폭
    weights: dict[str, float] # 피처 - 가중치 딕셔너리
    max_depth: int  # 탐색 깊이

class AlgoAgent(Agent):
    def __init__(self, grid_shape: Tuple[int, int], weights: dict[str, float], version: float, beam_width: int = 1, max_depth: int = 1) -> None:
        super().__init__(grid_shape, None)
        self.grid_shape = grid_shape
        self.weights = weights
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.agent_version = version

    def get_info(self) -> AlgoInfo:
        return AlgoInfo(
            algorithm_name=("greedy" if (self.beam_width == 1 and self.max_depth == 1) else "beam"),
            agent_version = self.agent_version,
            weights=self.weights,
            beam_width=self.beam_width,
            max_depth=self.max_depth,
        ) # type: ignore
    
    def set_board(self) -> None:
        pass
    
    def run_episode(self, board_source, render, delay) -> Tuple[int, int]: 
        board = Board.from_seed(size = self.grid_shape, seed=board_source)
        steps = 0
        score = 0

        if render: board.print_board()
        while True:
            is_over, _ = board.is_done()
            if is_over:
                break

            actions = board.get_valid_actions()
            best_action = pick_best_action(actions, board.grid, self.weights)
            _, cleared = board.do_action(best_action)
        
            score += cleared
            steps += 1
            if render:
                board.print_board()
                if delay > 0: time.sleep(delay)

        
        return (steps, score)