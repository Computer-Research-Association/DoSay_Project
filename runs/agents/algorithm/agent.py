from game.board import Board

from ai.envs.apple_env import AppleGameEnv

from agents.base import Agent
from agents.dtos import AlgoInfo

from pathlib import Path
from typing import Tuple


class AlgoAgent(Agent):
    def __init__(self, grid_shape: Tuple[int, int], model_path: Path) -> None:
        super().__init__(grid_shape, model_path)

    def get_info(self) -> AlgoInfo:
        return AlgoInfo() # type: ignore
    
    def set_board(self) -> None:
        pass
    
    def run_episode(self, board_source, render, delay) -> Tuple[int, int]: 
        return (0, 0)