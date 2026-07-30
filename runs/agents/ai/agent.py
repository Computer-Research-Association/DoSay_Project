from agents.base import Agent

from .utils import ignore_logs
from .model_loader import load_model
from agents.dtos import AIInfo

import time
from pathlib import Path
from typing import cast, Tuple

class AIAgent(Agent):
    def __init__(self, grid_shape: Tuple[int, int], model_path: Path) -> None:
        super().__init__(grid_shape, model_path)

        ignore_logs()
        # model_path 는 버전 폴더. 환경도 그 폴더의 env.py 가 만들어 준다.
        self.model, self.env, self.model_info = load_model(model_path, grid_shape)


    def get_info(self) -> AIInfo:
        return self.model_info

    def run_episode(self, board_source, render, delay):
        obs, info = self.env.reset(options={"board_source": board_source})
        terminated = truncated = False
        steps = 0
        score = info.get("score", 0)

        is_action_masking = self.model_info.use_action_masking

        if render: self.env.render()
        while not (terminated or truncated):
            if is_action_masking:
                action_masks = self.env.unwrapped.get_action_mask()  # type: ignore[attr-defined]
                if not action_masks.any(): break  # 유효한 수 없음 -> 종료
                action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
            else:
                action, _ = self.model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = self.env.step(int(action))
            steps += 1
            score = info.get("score", score)

            if render:
                print(f"\n[step {steps:>3}] reward={reward:>5.1f}  score={score}")
                self.env.render()
                if delay > 0: time.sleep(delay)

        return steps, score
