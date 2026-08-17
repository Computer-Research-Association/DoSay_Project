import sys
from pathlib import Path

VERSION_DIR = Path(__file__).resolve().parent
ROOT = VERSION_DIR.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VERSION_DIR))

import numpy as np
import torch
from sb3_contrib import QRDQN
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.training import (
    CheckpointSaver, ScoreCallback, TrainConfig, describe_device, model_filename,
    parse_train_args, save_model,
)
import env as env_mod
from model import POLICY_CLASS, make_policy_kwargs  # 같은 폴더

ALGORITHM = "QRDQN"
AI_VERSION = "9dB"
ROWS, COLS = 9, 18

# 관측 13x9x18 float32 = 8.4KB. 버퍼 1칸당 obs/next_obs 16.8KB -> 20만 칸에 약 3.4GB.
BUFFER_SIZE = 200_000


class MaskedQRDQN(QRDQN):
    """ε-greedy 탐험이 합법수 중에서만 뽑도록 고친 QR-DQN.

    원본은 action_space.sample() 로 7533개 중 균등추출한다. 합법수가 보통 50개
    안팎이라 거의 전부 불법 수가 나오고, 불법 수는 판을 진행시키지 못해 학습이
    성립하지 않는다. 합법 판정은 신경망이 관측에서 이미 계산하므로 그것을 쓴다.
    """

    def _masked_random(self, observation) -> np.ndarray:
        """합법수 중에서 균등추출. 합법 판정은 신경망이 관측에서 이미 계산한다."""
        with torch.no_grad():
            mask = self.policy.quantile_net.legal_mask(
                obs_as_tensor(observation, self.device)).cpu().numpy()

        actions = []
        for row in mask:
            legal = np.flatnonzero(row)
            actions.append(int(np.random.choice(legal)) if legal.size else 0)
        return np.array(actions)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        """워밍업 구간도 합법수만 뽑게 한다.

        SB3 원본은 num_timesteps < learning_starts 인 동안 predict() 를 통째로
        건너뛰고 action_space.sample() 로 7533개에서 균등추출한다. 합법수가 50개
        안팎이라 합법 확률이 0.66% 뿐이고 나머지는 판을 진행시키지 못한다.
        워밍업 2만 스텝이 통째로 버려지고 리플레이 버퍼도 쓰레기로 채워진다.
        """
        if self.num_timesteps < learning_starts:
            assert self._last_obs is not None
            action = self._masked_random(self._last_obs)
            return action, action          # Discrete 라 스케일링이 없다
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if deterministic or np.random.rand() >= self.exploration_rate:
            # super().predict() 를 부르면 안 된다. SB3 의 predict 가 ε 판정을 **한 번 더**
            # 하고, 그쪽 무작위 분기는 action_space.sample() 이라 마스킹이 없다.
            # 그러면 실제 불법 수 확률이 (1-ε)*ε 이 되어 ε=0.59 일 때 24% 에 달한다.
            # policy.predict 는 탐욕 경로라 신경망 안에서 이미 마스킹돼 있다.
            return self.policy.predict(observation, state, episode_start, deterministic)

        batched = self.policy.is_vectorized_observation(observation)
        obs = observation if batched else observation[None]
        action = self._masked_random(obs)
        return (action if batched else action[0]), state


def _env():
    def _init():
        factory = getattr(env_mod, "make_train_env", env_mod.make_env)
        return factory(ROWS, COLS, render_mode=None)   # 마스킹은 신경망 안에서 한다
    return _init


def generate_env(config: TrainConfig):
    return VecMonitor(SubprocVecEnv([_env() for _ in range(config.n_envs)]))


def train_model(env, config: TrainConfig):
    model = MaskedQRDQN(
        POLICY_CLASS,
        env,
        learning_rate=1e-4,
        buffer_size=BUFFER_SIZE,
        learning_starts=20_000,
        batch_size=256,          # (배치, 분위수 16, 행동 7533) = 123MB. VRAM 8GB 중 0.6GB 만 쓰고 있어 여유가 있다.
        tau=1.0,
        gamma=0.997,             # 81수 안에 끝나지만 max 연산 과대추정 누적을 감안해 1.0 은 피한다
        train_freq=4,
        # SB3 는 train_freq 를 '벡터 스텝' 으로 센다. n_envs=11 이면 환경 44 스텝마다
        # 경사 1회다. 한때 n_envs 에 비례해 늘려봤지만 GPU 가 이미 병목이라
        # 시간당 경사 갱신 횟수는 그대로인 채 fps 만 309 -> 73 으로 떨어졌다.
        # 12M 스텝 기준 11시간이 46시간이 되므로 되돌린다.
        gradient_steps=1,
        target_update_interval=5_000,
        exploration_fraction=0.25,   # total_timesteps 에 비례하므로 길게 돌리면 탐험도 길어진다
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        max_grad_norm=10.0,
        device=config.device,
        verbose=config.verbose,
        policy_kwargs=make_policy_kwargs(ROWS, COLS),
        tensorboard_log=str(config.log_dir),
    )

    model.learn(
        total_timesteps=config.total_timestep,
        tb_log_name=model_filename(ALGORITHM, AI_VERSION, config.total_timestep),
        callback=[CheckpointSaver(config, ALGORITHM, AI_VERSION), ScoreCallback()],
    )

    path = save_model(model, config, ALGORITHM, AI_VERSION, config.total_timestep)
    print(f"AI 모델 저장 완료! -> {path}")
    return model


def evaluate_model(model):
    eval_env = env_mod.make_env(ROWS, COLS, render_mode=None)
    scores = []
    for seed in range(10):
        obs, info = eval_env.reset(options={"board_source": 9000 + seed})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(int(action))
            done = terminated or truncated
        scores.append(info.get("score", 0))
    print(f"평균 점수: {np.mean(scores):.1f} +- {np.std(scores):.1f}  (10판, 만점 162)")


if __name__ == "__main__":
    config = parse_train_args(VERSION_DIR)
    print(describe_device(config.device))
    print(f"{config.n_envs}개 코어 병렬 처리 환경 구축")
    env = generate_env(config)
    model = train_model(env, config)
    evaluate_model(model)
