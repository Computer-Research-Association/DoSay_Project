import sys
from collections import deque
from pathlib import Path

VERSION_DIR = Path(__file__).resolve().parent
ROOT = VERSION_DIR.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VERSION_DIR))

import numpy as np
import torch
from sb3_contrib import QRDQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.training import (
    CheckpointSaver, ScoreCallback, TrainConfig, describe_device, model_filename,
    parse_train_args, save_model,
)
import env as env_mod
from model import POLICY_CLASS, make_policy_kwargs  # 같은 폴더

ALGORITHM = "QRDQN"
AI_VERSION = "10a"
ROWS, COLS = 9, 18

BUFFER_SIZE = 200_000
N_STEP = 6              # 부트스트랩 사슬을 30홉에서 6홉으로 줄인다
STEP_GAMMA = 0.997      # 한 스텝당 할인. 목표에는 STEP_GAMMA**N_STEP 이 곱해진다


class NStepReplayBuffer(ReplayBuffer):
    """n-step 반환값을 저장하는 리플레이 버퍼.

    **왜 필요한가.** V9 계열 다섯 모델이 알고리즘을 바꿔도 전부 114점에 수렴했고,
    점/수가 2.24~2.27 로 똑같았다. 즉 모두 같은 정책을 학습했다.

    합법수 개수를 재 보면 이유가 보인다. 판이 비어갈수록 47 -> 34 -> 17 -> 6 -> 3 으로
    붕괴한다. 즉 판을 가르는 결정은 선택지가 40개대인 **초반**에 내려지고, 그 대가는
    30수 뒤 후반에 나타난다. 1-step TD 로 30홉을 거슬러 올라가려면 그 사이 29개
    부트스트랩을 모두 통과해야 하는데, 형제 수 간 가치 차이가 2점 남짓이라
    그 신호가 잡음에 묻힌다.

    n-step 은 이 사슬을 6홉으로 줄인다. 실제 보상을 6수치 그대로 흘려보내므로
    부트스트랩 오차가 개입할 여지가 그만큼 줄어든다.

    **구현 메모.** 벡터 환경이 보조를 맞춰 돌기 때문에 매 벡터 스텝마다 환경마다
    정확히 하나씩 n-step 전이가 나온다. 그래서 SB3 의 배치 add 규약이 그대로 유지된다.
    창 안에서 에피소드가 끝나면 거기서 잘라 done=True 로 내보내므로,
    부트스트랩이 살아 있는 전이는 언제나 정확히 n 스텝짜리다.
    -> 알고리즘의 gamma 를 STEP_GAMMA**N_STEP 으로 주면 목표식이 정확해진다.
    """

    def __init__(self, *args, n_step: int = N_STEP, step_gamma: float = STEP_GAMMA, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_step = n_step
        self.step_gamma = step_gamma
        self._pending = [deque() for _ in range(self.n_envs)]

    def add(self, obs, next_obs, action, reward, done, infos):
        for i in range(self.n_envs):
            self._pending[i].append((obs[i], next_obs[i], action[i], reward[i], done[i]))

        if len(self._pending[0]) < self.n_step:
            return

        o = np.empty_like(obs); no = np.empty_like(next_obs)
        a = np.empty_like(action); r = np.empty_like(reward); d = np.empty_like(done)

        for i, window in enumerate(self._pending):
            total, discount = 0.0, 1.0
            for step, (_, w_next, _, w_reward, w_done) in enumerate(window):
                total += discount * w_reward
                discount *= self.step_gamma
                if w_done or step == self.n_step - 1:
                    no[i], d[i] = w_next, w_done
                    break
            o[i], a[i], r[i] = window[0][0], window[0][2], total
            window.popleft()

        super().add(o, no, a, r, d, infos)


class MaskedQRDQN(QRDQN):
    """ε-greedy 탐험이 합법수 중에서만 뽑도록 고친 QR-DQN."""

    def _masked_random(self, observation) -> np.ndarray:
        with torch.no_grad():
            mask = self.policy.quantile_net.legal_mask(
                obs_as_tensor(observation, self.device)).cpu().numpy()
        actions = []
        for row in mask:
            legal = np.flatnonzero(row)
            actions.append(int(np.random.choice(legal)) if legal.size else 0)
        return np.array(actions)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # SB3 는 워밍업 동안 predict() 를 건너뛰고 7533개에서 균등추출한다 (합법 확률 0.66%)
        if self.num_timesteps < learning_starts:
            assert self._last_obs is not None
            action = self._masked_random(self._last_obs)
            return action, action
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if deterministic or np.random.rand() >= self.exploration_rate:
            # super().predict() 는 ε 판정을 한 번 더 하고 그쪽은 마스킹이 없다
            return self.policy.predict(observation, state, episode_start, deterministic)
        batched = self.policy.is_vectorized_observation(observation)
        obs = observation if batched else observation[None]
        action = self._masked_random(obs)
        return (action if batched else action[0]), state


def _env():
    def _init():
        factory = getattr(env_mod, "make_train_env", env_mod.make_env)
        return factory(ROWS, COLS, render_mode=None)
    return _init


def generate_env(config: TrainConfig):
    return VecMonitor(SubprocVecEnv([_env() for _ in range(config.n_envs)]))


def train_model(env, config: TrainConfig):
    model = MaskedQRDQN(
        POLICY_CLASS,
        env,
        learning_rate=1e-4,
        buffer_size=BUFFER_SIZE,
        replay_buffer_class=NStepReplayBuffer,
        learning_starts=20_000,
        batch_size=256,
        tau=1.0,
        # 버퍼가 이미 n 스텝치 보상을 누적해 두었으므로 부트스트랩 항의 할인은 gamma^n 이다
        gamma=STEP_GAMMA ** N_STEP,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=5_000,
        exploration_fraction=0.25,
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
