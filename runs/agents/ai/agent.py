from agents.base import Agent

from .utils import ignore_logs
from .model_loader import load_model
from agents.dtos import AIInfo
from .search import ValueBeamSearch

import time
from pathlib import Path
from typing import Tuple


class AIAgent(Agent):
    def __init__(self, grid_shape: Tuple[int, int], model_path: Path) -> None:
        super().__init__(grid_shape, model_path)

        ignore_logs()
        # model_path 는 체크포인트(.zip) 또는 버전 폴더.
        # 환경은 그 체크포인트가 속한 버전 폴더의 env.py 가 만들어 준다.
        self.model, self.env, self.model_info, search_config = load_model(model_path, grid_shape)
        self.search = (
            ValueBeamSearch(self.model, self.env.unwrapped, search_config)
            if search_config.enabled else None
        )

    def get_info(self) -> AIInfo:
        return self.model_info

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

        if render: self.env.render()
        while not (terminated or truncated):
            action_masks = self.env.unwrapped.get_action_mask()  # type: ignore[attr-defined]
            if not action_masks.any(): break  # 유효한 수 없음 -> 종료

            if self.search is not None:
                action = self.search.choose(self.env.unwrapped.board)  # type: ignore[attr-defined]
            else:
                action = self._policy_action(obs, action_masks)

            obs, reward, terminated, truncated, info = self.env.step(action)
            steps += 1
            score = info.get("score", score)

            if render:
                print(f"\n[step {steps:>3}] reward={reward:>5.1f}  score={score}")
                self.env.render()
                if delay > 0: time.sleep(delay)

        return steps, score
