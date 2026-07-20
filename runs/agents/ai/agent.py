from agents.base import Agent

from .utils import ignore_logs
from .model_loader import load_model
from agents.dtos import AIInfo
from ai.envs.apple_env import AppleGameEnv
from ai.wrappers.action_mask import wrap_with_mask

import time
import gymnasium as gym
from pathlib import Path
from typing import cast, Tuple

class AIAgent(Agent):
    def __init__(self, grid_shape: Tuple[int, int], model_path: Path) -> None:
        super().__init__(grid_shape, model_path)
        
        import ai.envs
        ignore_logs()
        env = gym.make("envs/AppleGame-v0", render_mode="ansi", rows=grid_shape[0], cols=grid_shape[1])
        self.env = cast(AppleGameEnv, wrap_with_mask(env))
        self.model, self.model_info = load_model(model_path, self.env)


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
                action_masks = cast(AppleGameEnv, self.env.unwrapped).get_action_mask()
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
