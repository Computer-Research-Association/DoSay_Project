import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from ai.envs.action_sets import get_all_action
from ai.envs.base_env import AppleGameEnvBase
from game.action import Action


class AppleGameEnv(AppleGameEnvBase):
    STEP_REWARD_TOTAL = 1.0
    TERMINAL_REWARD_TOTAL = 1.5
    REWARD_ALL_CLEAR_BONUS = 0.1  # 희소하므로 낮게 책정
    INVALID_ACTION_REWARD = -1.0

    def build_actions(self, rows: int, cols: int) -> list[Action]:
        return get_all_action(rows, cols)

    def build_observation_space(self, rows: int, cols: int) -> spaces.Space:
        return spaces.Box(low=0, high=9, shape=(rows, cols), dtype=np.int8)

    def _get_obs(self) -> NDArray[np.int8]:
        return self.board.grid.copy()

    def compute_reward(self, *, remove_count: int, terminated: bool, is_all_clear: bool) -> float:
        reward = (remove_count / self.total_cell_count) * self.STEP_REWARD_TOTAL
        if terminated:
            if is_all_clear:
                reward += self.REWARD_ALL_CLEAR_BONUS
            reward += (self.score / self.total_cell_count) * self.TERMINAL_REWARD_TOTAL
        return reward

    def invalid_action_reward(self) -> float:
        return self.INVALID_ACTION_REWARD


def make_env(rows: int, cols: int, render_mode: str | None = None) -> AppleGameEnv:
    return AppleGameEnv(rows=rows, cols=cols, render_mode=render_mode)
