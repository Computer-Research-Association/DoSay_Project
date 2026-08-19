import sys
from pathlib import Path

VERSION_DIR = Path(__file__).resolve().parent
ROOT = VERSION_DIR.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VERSION_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.training import (
    CheckpointSaver, ScoreCallback, TrainConfig, describe_device, model_filename,
    parse_train_args, save_model,
)
import env as env_mod
from model import POLICY_CLASS, make_policy_kwargs  # 같은 폴더

ALGORITHM = "DQN"
AI_VERSION = "10b"
ROWS, COLS = 9, 18

BUFFER_SIZE = 200_000

ELITE_CAPACITY = 60_000    # 잘 풀린 판에서 나온 (상태, 행동) 만 따로 모으는 곳
ELITE_PERCENTILE = 80      # 최근 판 중 상위 20% 만 채택
ELITE_LOSS_WEIGHT = 0.5    # 모방 손실 가중치
ELITE_BATCH = 128
ELITE_WARMUP = 200         # 이만큼 쌓이기 전에는 모방하지 않는다


class EliteBuffer:
    """상위 성적 에피소드의 (관측, 행동) 만 모아 두는 저장소.

    **왜 이걸 하는가.** V9 계열 다섯 모델이 전부 114점, 점/수 2.24 로 같은 정책에
    수렴했다. 그런데 같은 모델의 판별 점수는 표준편차 15, 상위 10% 가 133, 최고가
    157 이다. 즉 **좋은 수순은 이미 우연히 나오고 있는데 학습에 남지 않는다.**

    TD 학습은 그 좋은 판을 다른 판과 똑같이 취급한다. 30수 뒤에 드러나는 이득이
    부트스트랩 사슬을 타고 오는 동안 잡음에 묻히기 때문이다. 자기모방은 그 경로를
    건너뛴다 — 결과가 좋았던 판에서 실제로 둔 수를 그대로 정답으로 삼는다.

    Q 값을 직접 건드리지 않고 **argmax 가 그 수를 고르도록** 하는 손실만 준다.
    (마스킹된 Q 를 로짓처럼 보고 교차엔트로피) Q 의 스케일을 망가뜨리지 않으려는 것이다.
    """

    def __init__(self, capacity: int = ELITE_CAPACITY):
        self.capacity = capacity
        self.obs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.recent_scores: list[float] = []

    def threshold(self) -> float:
        if len(self.recent_scores) < 30:
            return float("inf")
        return float(np.percentile(self.recent_scores[-400:], ELITE_PERCENTILE))

    def offer(self, episode, score: float) -> bool:
        """에피소드 하나를 제출. 상위권이면 채택한다."""
        self.recent_scores.append(score)
        if score < self.threshold():
            return False

        for obs, action in episode:
            self.obs.append(obs)
            self.actions.append(action)
        overflow = len(self.obs) - self.capacity
        if overflow > 0:                      # 오래된 것부터 버린다
            del self.obs[:overflow]
            del self.actions[:overflow]
        return True

    def sample(self, size: int):
        idx = np.random.randint(0, len(self.obs), size=size)
        return (np.stack([self.obs[i] for i in idx]),
                np.array([self.actions[i] for i in idx]))

    def __len__(self) -> int:
        return len(self.obs)


class SelfImitationDQN(DQN):
    """ε-greedy 를 마스킹하고, 잘 풀린 판을 따라 하는 손실을 더한 DQN."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elite = EliteBuffer()
        self._episodes = None                 # 환경별로 진행 중인 (관측, 행동) 기록

    # ── 탐험 (마스킹) ────────────────────────────────────────────────────
    def _masked_random(self, observation) -> np.ndarray:
        with torch.no_grad():
            mask = self.policy.q_net.legal_mask(
                obs_as_tensor(observation, self.device)).cpu().numpy()
        actions = []
        for row in mask:
            legal = np.flatnonzero(row)
            actions.append(int(np.random.choice(legal)) if legal.size else 0)
        return np.array(actions)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts:
            assert self._last_obs is not None
            action = self._masked_random(self._last_obs)
            return action, action
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if deterministic or np.random.rand() >= self.exploration_rate:
            return self.policy.predict(observation, state, episode_start, deterministic)
        batched = self.policy.is_vectorized_observation(observation)
        obs = observation if batched else observation[None]
        action = self._masked_random(obs)
        return (action if batched else action[0]), state

    # ── 에피소드 수집 ────────────────────────────────────────────────────
    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        if self._episodes is None:
            self._episodes = [[] for _ in range(self.n_envs)]

        last_obs = self._last_obs
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

        for i in range(self.n_envs):
            self._episodes[i].append((last_obs[i].copy(), int(buffer_action[i])))
            if dones[i]:
                score = float(infos[i].get("score", 0.0))
                self.elite.offer(self._episodes[i], score)
                self._episodes[i] = []

    # ── 학습 (TD 손실 + 모방 손실) ────────────────────────────────────────
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        super().train(gradient_steps, batch_size)
        if len(self.elite) < ELITE_WARMUP:
            return

        losses = []
        for _ in range(gradient_steps):
            obs, actions = self.elite.sample(min(ELITE_BATCH, len(self.elite)))
            q = self.policy.q_net(obs_as_tensor(obs, self.device))   # 불법 수는 이미 -1e8
            target = torch.as_tensor(actions, device=self.device, dtype=torch.long)
            loss = ELITE_LOSS_WEIGHT * F.cross_entropy(q, target)

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(loss.item())

        self.logger.record("train/elite_loss", float(np.mean(losses)))
        self.logger.record("train/elite_size", len(self.elite))
        self.logger.record("train/elite_threshold", self.elite.threshold())


class EliteStats(BaseCallback):
    def _on_step(self) -> bool:
        return True


def _env():
    def _init():
        factory = getattr(env_mod, "make_train_env", env_mod.make_env)
        return factory(ROWS, COLS, render_mode=None)
    return _init


def generate_env(config: TrainConfig):
    return VecMonitor(SubprocVecEnv([_env() for _ in range(config.n_envs)]))


def train_model(env, config: TrainConfig):
    model = SelfImitationDQN(
        POLICY_CLASS,
        env,
        learning_rate=1e-4,
        buffer_size=BUFFER_SIZE,
        learning_starts=20_000,
        batch_size=512,
        tau=1.0,
        gamma=0.997,
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
        callback=[CheckpointSaver(config, ALGORITHM, AI_VERSION), ScoreCallback(), EliteStats()],
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
