import sys
from pathlib import Path

VERSION_DIR = Path(__file__).resolve().parent
ROOT = VERSION_DIR.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VERSION_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.training import (
    CheckpointSaver, ScoreCallback, TrainConfig, describe_device, model_filename,
    parse_train_args, save_model,
)
from game.board import Board
import env as env_mod
from model import POLICY_CLASS, make_policy_kwargs  # 같은 폴더

ALGORITHM = "QRDQN"
AI_VERSION = "10c"
ROWS, COLS = 9, 18

BUFFER_SIZE = 200_000

# ── 전문가 반복(expert iteration) 설정 ──────────────────────────────────
EXPERT_INTERVAL = 250_000   # 이 스텝마다 탐색으로 교사 데이터를 새로 만든다
EXPERT_EPISODES = 24        # 한 번에 만들 판 수
EXPERT_CAPACITY = 120_000
EXPERT_BATCH = 128
EXPERT_LOSS_WEIGHT = 1.0
SEARCH_TOP_K = 32           # 루트에서 펼칠 후보 수
SEARCH_DEPTH = 3            # 30수짜리 신용할당에 깊이 2 는 너무 얕았다
SEARCH_BEAM = 4


class ExpertIterationQRDQN(QRDQN):
    """탐색이 만든 수를 교사로 삼아 정책을 끌어올리는 QR-DQN.

    **왜 이 방향인가.** V9 계열 다섯 모델(DQN / QR-DQN / PPO, 커리큘럼 유무)이
    12M 스텝을 먹고 전부 114점, 점/수 2.24 로 수렴했다. 판별 점수를 짝지어 비교하면
    분포 강화학습의 효과가 -0.09 ± 1.33 으로 정확히 0 이다. 알고리즘을 바꿔도 같은
    정책에 도달한다는 것은, 학습 신호 자체가 모두를 같은 국소최적으로 보낸다는 뜻이다.

    빠져나오려면 **현재 정책보다 나은 목표**가 필요하다. 탐색이 그것을 만든다.
    중요한 것은 한 번 쓰고 버리는 탐색이 아니라 **순환**이다.
    (탐색이 더 나은 수를 찾는다 -> 신경망이 그걸 배운다 -> 탐색의 잎 평가가 좋아진다
     -> 더 나은 수를 찾는다)
    V8 에서 잰 '빔탐색 +0.46' 은 이 순환 없이 한 번만 얹었을 때의 값이었다.

    **배포는 탐색 없이 한다.** 탐색은 학습 중 교사로만 쓰고, 최종 모델은 신경망의
    argmax 만으로 둔다. 손으로 만든 규칙은 어디에도 없다. 탐색이 쓰는 평가값도
    전부 신경망 자신의 Q 다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_obs: list[np.ndarray] = []
        self.expert_actions: list[int] = []
        self._next_expert_at = 0
        self._search_env = None

    def _excluded_save_params(self) -> list[str]:
        """학습 전용 상태는 체크포인트에 넣지 않는다.

        넣으면 두 가지가 터진다.

        1) `_search_env` 는 env.py 의 AppleGameEnv 인스턴스다. 학습 중 이 파일은
           최상위 모듈 `env` 로 import 돼 있어서 cloudpickle 이 모듈 참조로 저장하는데,
           measure.py 에는 `env` 모듈이 없어 로드가
           ModuleNotFoundError: No module named 'env' 로 죽는다.
        2) `expert_obs` 는 관측(8.4KB) 수만 개다. 6M 스텝 시점에 25,696개였으니
           base64 로 250MB 넘는 zip 이 만들어진다.

        셋 다 추론에 쓰이지 않는다. 배포 모델은 신경망 argmax 만으로 둔다.
        (이 수정 이전에 저장된 체크포인트는 env.py 의 LOAD_CUSTOM_OBJECTS 로 살린다.)
        """
        return super()._excluded_save_params() + [
            "_search_env", "expert_obs", "expert_actions",
        ]

    # ── 탐험 (마스킹) ────────────────────────────────────────────────────
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

    # ── 탐색 (교사) ──────────────────────────────────────────────────────
    def _q_values(self, boards: list[Board]) -> np.ndarray:
        """여러 판의 마스킹된 Q. 불법 수는 신경망 안에서 이미 큰 음수다."""
        obs = []
        for board in boards:
            self._search_env.board = board
            obs.append(self._search_env._get_obs())
        with torch.no_grad():
            tensor = obs_as_tensor(np.stack(obs), self.device)
            return self.policy.quantile_net(tensor).mean(dim=1).cpu().numpy()

    def _search_best_action(self, root: Board) -> int:
        """깊이 SEARCH_DEPTH 빔 탐색. 잎의 가치는 max_a Q(s,a) 다."""
        cell_count = self._search_env.total_cell_count
        frontier = [(root, 0.0, -1)]                       # (판, 누적 사과, 첫 수)
        best_action, best_score = -1, -np.inf

        for depth in range(SEARCH_DEPTH):
            q = self._q_values([node[0] for node in frontier])
            width = SEARCH_TOP_K if depth == 0 else max(SEARCH_TOP_K // 4, 2)

            children = []
            for (board, gained, first), row in zip(frontier, q):
                legal = np.flatnonzero(row > -1e6)
                if legal.size == 0:
                    continue
                for index in legal[np.argsort(-row[legal])[:width]]:
                    child = Board.from_board(board.grid)
                    _, removed = child.do_action(self._search_env.index_to_action[int(index)])
                    children.append((child, gained + removed / cell_count,
                                     int(index) if first < 0 else first))
            if not children:
                break

            leaf_q = self._q_values([c[0] for c in children])
            values = leaf_q.max(axis=1)
            terminal = np.array([not c[0].get_valid_actions() for c in children])
            values[terminal | (values < -1e6)] = 0.0

            scores = np.array([c[1] for c in children]) + values
            top = int(np.argmax(scores))
            if scores[top] > best_score:
                best_score, best_action = float(scores[top]), children[top][2]

            alive = [i for i in np.argsort(-scores) if not terminal[i]]
            frontier = [children[i] for i in alive[:SEARCH_BEAM]]
            if not frontier:
                break

        return best_action

    def _collect_expert(self) -> float:
        """탐색으로 판을 두면서 (관측, 탐색이 고른 수)를 모은다."""
        if self._search_env is None:
            self._search_env = env_mod.make_env(ROWS, COLS, render_mode=None)

        scores = []
        for _ in range(EXPERT_EPISODES):
            self._search_env.reset()
            while True:
                board = self._search_env.board
                if not board.get_valid_actions():
                    break
                action = self._search_best_action(board)
                if action < 0:
                    break
                self._search_env.board = board          # 탐색이 갈아끼운 판을 되돌린다
                self.expert_obs.append(self._search_env._get_obs())
                self.expert_actions.append(action)
                _, _, terminated, truncated, info = self._search_env.step(action)
                if terminated or truncated:
                    break
            scores.append(float(info.get("score", 0)))

        overflow = len(self.expert_obs) - EXPERT_CAPACITY
        if overflow > 0:
            del self.expert_obs[:overflow]
            del self.expert_actions[:overflow]
        return float(np.mean(scores))

    # ── 학습 (TD + 교사 모방) ────────────────────────────────────────────
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        super().train(gradient_steps, batch_size)

        if self.num_timesteps >= self._next_expert_at:
            self._next_expert_at = self.num_timesteps + EXPERT_INTERVAL
            mean_score = self._collect_expert()
            self.logger.record("expert/search_score", mean_score)
            self.logger.record("expert/buffer_size", len(self.expert_obs))
            print(f"[{self.num_timesteps:>12,} steps] 탐색 교사 {EXPERT_EPISODES}판 "
                  f"평균 {mean_score:.1f}점, 누적 {len(self.expert_obs):,}개", flush=True)

        if len(self.expert_obs) < EXPERT_BATCH:
            return

        losses = []
        for _ in range(gradient_steps):
            idx = np.random.randint(0, len(self.expert_obs), size=EXPERT_BATCH)
            obs = np.stack([self.expert_obs[i] for i in idx])
            target = torch.as_tensor([self.expert_actions[i] for i in idx],
                                     device=self.device, dtype=torch.long)
            # 마스킹된 Q 를 로짓처럼 보고 교차엔트로피. Q 스케일은 건드리지 않고
            # argmax 가 탐색의 선택을 향하게만 민다.
            q = self.policy.quantile_net(obs_as_tensor(obs, self.device)).mean(dim=1)
            loss = EXPERT_LOSS_WEIGHT * F.cross_entropy(q, target)

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(loss.item())

        self.logger.record("expert/imitation_loss", float(np.mean(losses)))


def _env():
    def _init():
        factory = getattr(env_mod, "make_train_env", env_mod.make_env)
        return factory(ROWS, COLS, render_mode=None)
    return _init


def generate_env(config: TrainConfig):
    return VecMonitor(SubprocVecEnv([_env() for _ in range(config.n_envs)]))


def train_model(env, config: TrainConfig):
    model = ExpertIterationQRDQN(
        POLICY_CLASS,
        env,
        learning_rate=1e-4,
        buffer_size=BUFFER_SIZE,
        learning_starts=20_000,
        batch_size=256,
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
    print(f"평균 점수: {np.mean(scores):.1f} +- {np.std(scores):.1f}  (10판, 탐색 미사용)")


if __name__ == "__main__":
    config = parse_train_args(VERSION_DIR)
    print(describe_device(config.device))
    print(f"{config.n_envs}개 코어 병렬 처리 환경 구축")
    env = generate_env(config)
    model = train_model(env, config)
    evaluate_model(model)
