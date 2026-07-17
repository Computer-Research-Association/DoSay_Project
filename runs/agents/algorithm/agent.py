from game.board import Board
from runs.agents.base import Agent

from ai.envs.apple_env import AppleGameEnv

from agents.dtos import AlgoInfo
from typing import cast, Tuple


class AlgoAgent(Agent):
    def __init__(self, grid_shape: Tuple[int, int], custom_param_gogo: str) -> None:
        super().__init__(grid_shape)

    def get_info(self) -> AlgoInfo:
        return AlgoInfo() # type: ignore
    
    def set_board(self) -> None:
        pass
    
    def run_episode(self, board_source, render, delay) -> Tuple[int, int]: 
        return (0, 0)