from game.board import Board
from agents.base import Agent

from .utils import ignore_logs
from .model_loader import load_model
from agents.dtos import AIInfo
from ai.envs.apple_env import AppleGameEnv

import time
from typing import cast

class AIAgent(Agent):
    def __init__(self, env: AppleGameEnv, model_path: str) -> None:
        super().__init__(env)
        self.env = env
        self.model, self.model_info = load_model(model_path, self.env)
        # self.MODEL_PATH = model_path

        ignore_logs()

    def get_info(self) -> AIInfo:
        return self.model_info
    
    def set_board(self) -> None:
        self.env.reset()

    def run_episode(self, render: bool, delay: float = 0.5):
        obs, info = self.env.reset("시드 혹은 init board로 초기화") # type: ignore
        terminated = truncated = False
        steps = 0
        score = info.get("score", 0)

        is_action_masking = self.model_info.use_action_masking

        if render: self.env.render()
        while not (terminated or truncated):
            if is_action_masking:
                action_masks = cast(AppleGameEnv, self.env.unwrapped).get_action_mask()  # type: ignore
                if not action_masks.any(): break  # 유효한 수 없음 -> 종료
                action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
            else:
                action, _ = self.model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = self.env.step(int(action))
            steps += 1
            score = info.get("score", score)

            if render:
                print(f"[step {steps:>3}] reward={reward:>5.1f}  score={score}")
                self.env.render()
                if delay > 0: time.sleep(delay)

        return steps, score
