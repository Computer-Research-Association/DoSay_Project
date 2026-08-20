from agents.base import Agent

from .utils import ignore_logs
from .model_loader import load_model
from agents.dtos import AIInfo
from .search import ValueBeamSearch

import time
from pathlib import Path
from typing import Tuple

from game.action import Action
from game.board import Board


class AIAgent(Agent):
    def __init__(self, grid_shape: Tuple[int, int], model_path: Path,
                 device: str = "cpu") -> None:
        super().__init__(grid_shape, model_path)

        ignore_logs()
        # model_path 는 체크포인트(.zip) 또는 버전 폴더.
        # 환경은 그 체크포인트가 속한 버전 폴더의 env.py 가 만들어 준다.
        self.model, self.env, self.model_info, search_config = load_model(
            model_path, grid_shape, device=device)
        self.search = (
            ValueBeamSearch(self.model, self.env.unwrapped, search_config)
            if search_config.enabled else None
        )

    def get_info(self) -> AIInfo:
        return self.model_info

    def select_action(self, board: Board) -> Action | None:
        """판 하나를 받아 둘 수를 고른다. env 는 관측 인코딩을 빌리는 용도로만 쓴다."""
        if not board.get_valid_actions():
            return None

        env = self.env.unwrapped
        if self.search is not None:
            # choose() 가 끝나면서 env.board 를 원래대로 되돌려 놓는다.
            index = self.search.choose(board)
        else:
            env.board = board   # type: ignore[attr-defined]
            index = self._policy_action(env._get_obs(), env.get_action_mask())  # type: ignore[attr-defined]

        return env.index_to_action[index]   # type: ignore[attr-defined]

    def _policy_action(self, obs, action_masks) -> int:
        if self.model_info.use_action_masking:
            action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
        else:
            action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def run_episode(self, board_source, render, delay):
        obs, info = self.env.reset(options={"board_source": board_source})
        terminated = truncated = False
        steps = 0
        score = info.get("score", 0)

        # 합법수는 반드시 사과를 1개 이상 지우므로 판 칸수보다 오래 갈 수 없다.
        # 이 한계를 넘었다는 것은 판이 진행되지 않고 있다는 뜻이다.
        max_steps = self.env.unwrapped.total_cell_count  # type: ignore[attr-defined]

        if render: self.env.render()
        while not (terminated or truncated):
            action_masks = self.env.unwrapped.get_action_mask()  # type: ignore[attr-defined]
            if not action_masks.any(): break  # 유효한 수 없음 -> 종료

            if self.search is not None:
                action = self.search.choose(self.env.unwrapped.board)  # type: ignore[attr-defined]
            else:
                action = self._policy_action(obs, action_masks)

            # 불법 수는 판을 바꾸지 않고 terminated 도 False 로 돌아온다. 그대로 두면
            # 같은 수를 영원히 반복하며 조용히 멈춘다. 조용히 도는 것보다 바로 죽는 편이 낫다.
            if not (0 <= action < len(action_masks)) or not action_masks[action]:
                raise RuntimeError(
                    f"에이전트가 불법 수를 선택했습니다 (action={action}, "
                    f"합법수 {int(action_masks.sum())}개, step={steps}).\n"
                    "그대로 두면 판이 진행되지 않아 무한 루프가 됩니다."
                )
            if steps >= max_steps:
                raise RuntimeError(
                    f"한 판이 {max_steps}수를 넘었습니다. 판이 진행되지 않고 있습니다."
                )

            obs, reward, terminated, truncated, info = self.env.step(action)
            steps += 1
            score = info.get("score", score)

            if render:
                print(f"\n[step {steps:>3}] reward={reward:>5.1f}  score={score}")
                self.env.render()
                if delay > 0: time.sleep(delay)

        return steps, score
