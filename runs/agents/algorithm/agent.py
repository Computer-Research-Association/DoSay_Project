from game.board import Board
from runs.agents.base import Agent

from ai.envs.apple_env import AppleGameEnv

import time
import gymnasium as gym
from typing import cast, Tuple
from dataclasses import dataclass, field


@dataclass
class AlgoInfo:
    algorithm_name: str
    heuristic_name: str  # 사용한 휴리스틱 종류
    agent_version: str
    source_path: str
    beam_width: int  # 빔서치 폭
    max_depth: int  # 탐색 깊이
    time_limit_sec: int


class AlgoAgent(Agent):
    def __init__(self, env: AppleGameEnv, model_path: str) -> None:
        super().__init__(env)
        self.env = env

    def get_info(self) -> AlgoInfo:
        return AlgoInfo() # type: ignore
    
    def set_board(self) -> None:
        pass
    
    def run_episode(self, render: bool, delay: float) -> Tuple[int, int]: 
        return (0, 0)