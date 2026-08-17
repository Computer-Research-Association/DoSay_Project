"""DQN V15 — **정책이 탐색을 좁힌다.** V14 에서 빠져 있던 알파제로 구조를 넣는다.

    데이터:  빔(폭 32)으로 한 판을 끝까지 계획한다. 그 과정에서 두 종류 라벨이 나온다.
    정책:    펼친 부모마다 자식들의 **가치 순위 분포** (교차엔트로피)
    가치:    **죽은 빔 전부**의 경로 위 모든 판의 실제 결과 (Huber)
    반복:    정책↑ -> top-k 가지치기 안전 -> 탐색이 **싸진다** -> 폭↑ -> 가치↑ -> 정책↑

──────────────────────────────────────────────────────────────────────────
V14 의 세 가지 결함 (2026-08-12 실측, model.py 머리말에 근거)
──────────────────────────────────────────────────────────────────────────
1. 빔이 정책 헤드를 한 번도 쓰지 않았다 -> 폭으로만 점수를 샀다 (판당 40만회 호출)
2. 정책 목표가 argmax 하나 + λ=3 margin -> 퇴화 해(손실 = λ)에 300k 스텝 정지
3. 가치 손실이 162^2 로 나눠져 있었다 -> 정책 손실의 1/4500, 가치가 600스텝에 즉사

V15 는 셋을 각각 고친다. **폭은 V14 와 같은 32 로 둔다** — 이 버전의 질문은
"폭 없이 점수가 오르는가" 이고, 폭을 같이 바꾸면 그 답을 못 얻는다.

기준: V12b+빔W32 **123.66** (학습 0) / V14@300k+빔W32 **124.80** (+1.14) /
      V14@300k+빔W256 129.15 / 어닐링 ~137 / 이론상 최대 137~140
"""

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
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.training import (
    CheckpointSaver, ScoreCallback, TrainConfig, describe_device, model_filename,
    parse_train_args, save_model,
)
import env as env_mod
from model import LABEL_TOPK, POLICY_CLASS, make_policy_kwargs

ALGORITHM = "DQN"
AI_VERSION = "15"
ROWS, COLS = 9, 18

# SB3 껍데기용. 학습에 쓰이지 않으므로 작게 둔다 (obs 가 13채널이라 20만이면 3.4GB).
SHELL_BUFFER_SIZE = 10_000
BATCH_SIZE = 256

# 라벨 버퍼. 판당 가치 라벨 최대 4,096개 / 정책 라벨 약 1,760개가 나온다.
VALUE_CAPACITY = 400_000
POLICY_CAPACITY = 200_000

# **가치 손실은 사과 단위로 둔다.** V14 는 /162^2 이라 정책 손실의 1/4500 이었고,
# 그래서 워밍스타트한 가치가 600스텝 만에 MAE 0.679 -> 4.48 로 무너졌다.
# Huber 를 쓰는 이유: 초반 목표가 ~125 사과라 MSE 면 기울기가 폭주한다.
VALUE_LOSS_WEIGHT = 1.0

# **워밍스타트.** V12b 의 인코더와 가치 헤드를 물려받아 1스텝째부터 123.66 짜리
# 빔 교사를 쓴다. 없으면 초반 빔이 "사과 많이 먹기"(91점) 근처에서 논다.
INIT_FROM = ROOT / "ai/models/DQN/V12b/models/V12b_DQN_6000000.zip"


class LabelStore:
    """빔이 만든 라벨의 링 버퍼. 판(grid)만 들고 obs 는 학습 때 GPU 에서 만든다.

    obs 는 13채널이라 판의 13배다. 판당 6천 개 라벨을 obs 로 들고 있으면
    버퍼만 수십 GB 가 된다.
    """

    def __init__(self, rows: int, cols: int, k: int):
        self.v_grid = np.zeros((VALUE_CAPACITY, rows, cols), dtype=np.int8)
        self.v_remain = np.zeros(VALUE_CAPACITY, dtype=np.float32)
        self.p_grid = np.zeros((POLICY_CAPACITY, rows, cols), dtype=np.int8)
        self.p_action = np.zeros((POLICY_CAPACITY, k), dtype=np.int16)
        self.p_prob = np.zeros((POLICY_CAPACITY, k), dtype=np.float32)
        self.v_pos = self.v_size = 0
        self.p_pos = self.p_size = 0

    @staticmethod
    def _put(dst: list[np.ndarray], src: list[np.ndarray], pos: int, size: int,
             capacity: int) -> tuple[int, int]:
        n = src[0].shape[0]
        if n == 0:
            return pos, size
        if n >= capacity:                       # 한 판이 버퍼보다 크면 뒤쪽만
            src = [s[-capacity:] for s in src]
            n = capacity
        end = pos + n
        for d, s in zip(dst, src):
            if end <= capacity:
                d[pos:end] = s
            else:
                cut = capacity - pos
                d[pos:] = s[:cut]
                d[:end - capacity] = s[cut:]
        return end % capacity, min(size + n, capacity)

    def add(self, labels: dict) -> None:
        self.v_pos, self.v_size = self._put(
            [self.v_grid, self.v_remain],
            [labels["value_grids"].numpy(), labels["value_remain"].numpy()],
            self.v_pos, self.v_size, VALUE_CAPACITY)
        self.p_pos, self.p_size = self._put(
            [self.p_grid, self.p_action, self.p_prob],
            [labels["policy_grids"].numpy(), labels["policy_actions"].numpy(),
             labels["policy_probs"].numpy()],
            self.p_pos, self.p_size, POLICY_CAPACITY)

    def ready(self, batch_size: int) -> bool:
        return self.v_size >= batch_size and self.p_size >= batch_size

    def sample_value(self, batch_size: int):
        idx = np.random.randint(0, self.v_size, size=batch_size)
        return self.v_grid[idx], self.v_remain[idx]

    def sample_policy(self, batch_size: int):
        idx = np.random.randint(0, self.p_size, size=batch_size)
        return self.p_grid[idx], self.p_action[idx], self.p_prob[idx]


class GuidedBeamDQN(DQN):
    """SB3 DQN 의 껍데기만 쓴다. TD 도, 타깃망도, 보상도 학습에 안 들어간다.

    껍데기를 쓰는 이유는 `runs/measure.py` 가 MODEL_REGISTRY 를 통해 SB3 클래스로
    로드하기 때문이다. 행동 선택(빔 계획)은 `q_net.forward()` 안에 있으므로
    하네스는 한 줄도 바꿀 필요가 없다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = LabelStore(ROWS, COLS, LABEL_TOPK)
        self._recalls: list[float] = []
        self.policy.q_net.collect_labels = True
        self.policy.q_net_target.collect_labels = False

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["labels", "_recalls"]

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        """항상 빔의 계획을 따른다. ε 탐험을 쓰지 않는다.

        판이 정해지면 무작위성이 없어 계획에서 벗어날 이유가 없고, 계획을 벗어나면
        캐시가 무효가 될 뿐이다. 다양성은 매 판 새로 뽑히는 판이 제공한다.
        """
        assert self._last_obs is not None
        chosen, _ = self.policy.predict(self._last_obs, deterministic=True)
        self._drain_labels()
        action = np.asarray(chosen).reshape(-1)
        return action, action

    def _drain_labels(self) -> None:
        pending = self.policy.q_net.pending_labels
        if not pending:
            return
        for labels in pending:
            self.labels.add(labels)
            recall = labels.get("topk_recall")
            if recall is not None and not np.isnan(recall):
                self._recalls.append(float(recall))
        pending.clear()

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        return self.policy.predict(observation, state, episode_start, deterministic=True)

    # ── 학습 (정책 분포 증류 + 가치 회귀) ─────────────────────────────────
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if not self.labels.ready(batch_size):
            return
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        q_net = self.policy.q_net
        index = q_net.index

        pol_losses, agrees, top8, value_losses, value_maes = [], [], [], [], []
        for _ in range(gradient_steps):
            # ── 정책: 자식들의 가치 순위 분포를 교차엔트로피로 ──────────────
            p_grid, p_action, p_prob = self.labels.sample_policy(batch_size)
            grids = torch.as_tensor(p_grid, device=self.device, dtype=torch.float32)
            actions = torch.as_tensor(p_action.astype(np.int64), device=self.device)
            target = torch.as_tensor(p_prob, device=self.device)

            logits = q_net.policy_logits(index.observation(grids))
            logp = torch.log_softmax(logits, dim=1).gather(1, actions)
            # 라벨에 없는 합법수는 목표 0 이다 (log_softmax 가 전체를 정규화하므로
            # 자동으로 눌린다). V14 의 argmax + margin 은 목표가 모호할 때
            # "전부 같게" 라는 퇴화 해로 갔다.
            policy_loss = -(target * logp).sum(dim=1).mean()

            # ── 가치: 죽은 빔 전부의 실제 결과를 Huber 로 (사과 단위) ───────
            v_grid, v_remain = self.labels.sample_value(batch_size)
            vgrids = torch.as_tensor(v_grid, device=self.device, dtype=torch.float32)
            vtarget = torch.as_tensor(v_remain, device=self.device)

            obs = index.observation(vgrids)
            leftover = q_net.expected_leftover(obs)
            occupied = (vgrids != 0).flatten(1).sum(dim=1)
            future = occupied - leftover
            value_loss = F.smooth_l1_loss(future, vtarget)

            loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            with torch.no_grad():
                best = actions.gather(1, target.argmax(dim=1, keepdim=True))
                ranked = logits.topk(LABEL_TOPK, dim=1).indices
                pol_losses.append(policy_loss.item())
                agrees.append((logits.argmax(dim=1, keepdim=True) == best).float().mean().item())
                top8.append((ranked == best).any(dim=1).float().mean().item())
                value_losses.append(value_loss.item())
                value_maes.append((future - vtarget).abs().mean().item())

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/policy_loss", float(np.mean(pol_losses)))
        # 정책 argmax 가 가치 1등과 일치하는 비율. **경량 모델의 실력을 좌우한다.**
        self.logger.record("train/teacher_agreement", float(np.mean(agrees)))
        # 정책 상위 8 안에 가치 1등이 드는 비율. **1 에 가까워지면 배포에서
        # POLICY_TOPK=8 로 탐색을 3배 싸게 할 수 있다** (별도 측정 없이 여기서 읽힌다).
        self.logger.record("train/topk_recall_train", float(np.mean(top8)))
        self.logger.record("train/value_loss", float(np.mean(value_losses)))
        # 빔의 가지치기 품질. V12b 워밍스타트 직후가 0.679, V14 는 4.48 로 무너졌다.
        self.logger.record("train/value_mae", float(np.mean(value_maes)))
        self.logger.record("buffer/value_rows", self.labels.v_size)
        self.logger.record("buffer/policy_rows", self.labels.p_size)
        if self._recalls:
            # 빔이 실제로 계획하는 동안 잰 것 (학습 배치가 아니라 현장 값).
            self.logger.record("beam/topk_recall", float(np.mean(self._recalls[-2000:])))


def warm_start(model, path: Path) -> None:
    """V12b 의 인코더 + 가치 헤드를 물려받는다. 정책 헤드는 무작위로 남는다."""
    if not path.exists():
        print(f"[워밍스타트 없음] {path} 를 찾지 못했습니다.\n"
              "  무작위 초기화로 시작합니다 — 초반 빔이 약해 학습이 훨씬 느립니다.")
        return
    _, params, _ = load_from_zip_file(path, device=model.device)
    incompatible = model.policy.load_state_dict(params["policy"], strict=False)  # type: ignore[index]
    loaded = len(params["policy"]) - len(incompatible.unexpected_keys)           # type: ignore[index]
    print(f"[워밍스타트] {path.name} 에서 {loaded}개 텐서를 물려받았습니다.")
    print(f"  새로 배우는 것: {len(incompatible.missing_keys)}개 (정책 헤드)")


def _env():
    def _init():
        factory = getattr(env_mod, "make_train_env", env_mod.make_env)
        return factory(ROWS, COLS, render_mode=None)
    return _init


def generate_env(config: TrainConfig):
    return VecMonitor(SubprocVecEnv([_env() for _ in range(config.n_envs)]))


def train_model(env, config: TrainConfig):
    model = GuidedBeamDQN(
        POLICY_CLASS,
        env,
        learning_rate=2e-4,
        buffer_size=SHELL_BUFFER_SIZE,
        replay_buffer_class=ReplayBuffer,
        learning_starts=2_000,
        batch_size=BATCH_SIZE,
        # 아래 넷은 상속 때문에 필요할 뿐 손실에 들어가지 않는다.
        tau=1.0, gamma=1.0, target_update_interval=10 ** 9,
        exploration_initial_eps=0.0, exploration_final_eps=0.0, exploration_fraction=1.0,
        train_freq=4,
        # 빔이 만든 라벨은 비싸지만(판당 2.7초) 판당 6천 개가 나온다. V14 는 55개였다.
        gradient_steps=4,
        max_grad_norm=10.0,
        device=config.device,
        verbose=config.verbose,
        policy_kwargs=make_policy_kwargs(ROWS, COLS),
        tensorboard_log=str(config.log_dir),
    )
    warm_start(model, INIT_FROM)

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
    for seed in range(5):
        obs, info = eval_env.reset(options={"board_source": 9000 + seed})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(int(action))
            done = terminated or truncated
        scores.append(info.get("score", 0))
    print(f"평균 점수: {np.mean(scores):.1f} +- {np.std(scores):.1f}  (5판, 만점 162)")


if __name__ == "__main__":
    config = parse_train_args(VERSION_DIR)
    print(describe_device(config.device))
    print(f"{config.n_envs}개 코어 병렬 처리 환경 구축")
    env = generate_env(config)
    model = train_model(env, config)
    evaluate_model(model)
