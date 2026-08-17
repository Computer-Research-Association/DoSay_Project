import sys
from pathlib import Path

VERSION_DIR = Path(__file__).resolve().parent
ROOT = VERSION_DIR.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VERSION_DIR))

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.training import (
    CheckpointSaver, ScoreCallback, TrainConfig, describe_device, model_filename,
    parse_train_args, save_model,
)
from env import make_env                          # 같은 폴더
from model import POLICY_CLASS, make_policy_kwargs  # 같은 폴더

ALGORITHM = "DQN"
AI_VERSION = "1.0"
ROWS, COLS = 9, 18

# 관측이 13x9x18 float32 = 8.4KB 라 버퍼 1칸당 obs/next_obs 16.8KB.
# 10만 칸이면 약 1.7GB 다. 서버 메모리에 맞춰 조절할 것.
BUFFER_SIZE = 100_000


class MaskedDQN(DQN):
    """ε-greedy 탐험이 합법수 중에서만 뽑도록 고친 DQN.

    SB3 원본은 self.action_space.sample() 로 7533개 중 균등추출한다. 이 게임에서
    합법수는 보통 50개 안팎이라 거의 전부 불법 수가 나오고, 불법 수는 판을
    진행시키지 못하므로 학습이 성립하지 않는다.

    합법 판정은 Q 신경망이 관측으로부터 이미 계산하고 있으므로 그것을 그대로 쓴다.
    """

    def _masked_random(self, observation) -> np.ndarray:
        """합법수 중에서 균등추출. 합법 판정은 신경망이 관측에서 이미 계산한다."""
        with torch.no_grad():
            mask = self.policy.q_net.legal_mask(
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
        return make_env(ROWS, COLS, render_mode=None)   # 마스킹은 신경망 안에서 한다
    return _init


def generate_env(config: TrainConfig):
    return VecMonitor(SubprocVecEnv([_env() for _ in range(config.n_envs)]))


def train_model(env, config: TrainConfig):
    model = MaskedDQN(
        POLICY_CLASS,
        env,
        learning_rate=1e-4,
        buffer_size=BUFFER_SIZE,
        learning_starts=20_000,
        batch_size=256,
        tau=1.0,
        # 에피소드가 81수 안에 끝나므로 1.0 도 성립하지만, DQN 은 max 연산의
        # 과대추정이 누적되면 발산한다. 46수 기준 끝단 할인이 0.87 인 0.997 로 절충.
        gamma=0.997,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=5_000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10.0,
        device=config.device,
        verbose=config.verbose,
        policy_kwargs=make_policy_kwargs(ROWS, COLS),
        tensorboard_log=str(config.log_dir),
    )

    model.learn(
        total_timesteps=config.total_timestep,
        tb_log_name=model_filename(ALGORITHM, AI_VERSION, config.total_timestep),
        callback=[
            CheckpointSaver(config, ALGORITHM, AI_VERSION),
            ScoreCallback(),
        ],
    )

    path = save_model(model, config, ALGORITHM, AI_VERSION, config.total_timestep)
    print(f"AI 모델 저장 완료! -> {path}")
    return model


def evaluate_model(model):
    eval_env = make_env(ROWS, COLS, render_mode=None)
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
