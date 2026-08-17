"""DQN V11c — 결과 예측 학습 (learned lookahead + 잔존맵). **새 시도 ②**

한 줄 요약: **빔서치가 추론 시점에 60배 비용으로 사 오는 정보를, 학습 시점에
사서 가중치에 넣는다.** 추론은 그대로 1회 forward(판당 0.4초)다.

──────────────────────────────────────────────────────────────────────────
왜 이 방향인가
──────────────────────────────────────────────────────────────────────────
문서의 기준선 표에 설명되지 않은 줄이 하나 있다.

    1수 앞 탐색 (한 줄 휴리스틱)   118.1
    DQN V1.0 (12M 스텝)          114.5

"수를 둬 보고 남는 합법수를 세는" 규칙이 12M 스텝 신경망보다 3.6점 높다.
신경망이 못 배운 게 아니라 **결정 시점에 그 정보가 없다.** 사각형을 지운 뒤의
판을 볼 수 없으니까. 빔서치(V9dB, 121.45)는 그 정보를 추론 시점에 사서 쓰는데,
대가가 판당 0.38초 -> 22.6초다.

그래서 V11c 는 신경망에게 **그 결과를 예측하도록 가르친다.** 상태당 감독 정보가
이렇게 늘어난다.

    지금까지:  TD 목표 스칼라 1개
    V11c:      look 8개 + survive 162개 + TD 1개  = 약 170개

**QR-DQN(V9a)의 분위수 16개와는 다르다.** 분위수는 같은 스칼라를 여러 각도로
볼 뿐이라 상태당 정보가 늘지 않았고, 실제로 짝비교 결과가 -0.09 ± 1.33 (정확히
0) 이었다. 여기서는 **다른 것**을 배운다.

──────────────────────────────────────────────────────────────────────────
보조 목표 ① look — "이 수를 두면 합법수가 몇 개 남는가" (행동마다)
──────────────────────────────────────────────────────────────────────────
라벨은 게임 규칙 그대로 GPU 에서 정확히 계산한다 (model.py 의
RectIndex.afterstate_legal_counts). 손으로 튜닝한 값이 하나도 없고, 게임 엔진도
부르지 않는다.

Q 와의 연결은 model.py 참고: Q = base + look_weight · look.detach() 이고
look_weight 는 0 에서 시작하는 학습 파라미터다. **`aux/look_weight` 가 이 실험의
핵심 지표다** — 0 근처에 머물면 TD 가 "1수 앞 정보는 쓸모없다"고 판단한 것이고,
그러면 이 가설 자체가 틀린 것이다.

──────────────────────────────────────────────────────────────────────────
보조 목표 ② survive — "이 사과가 끝까지 남을 확률" (칸마다)
──────────────────────────────────────────────────────────────────────────
라벨은 에이전트 자신의 에피소드 결말에서 나온다 (env.py 가 final_grid 를 낸다).

이 게임은 한 수가 **항상 합 10** 을 가져가고 판 합이 810 으로 고정이라
점수 = 162 - 남은 사과 수 다. 그리고 초기 재고(각 숫자 18개)는 9+1, 8+2, 7+3,
6+4, 5+5 로 짝지으면 정확히 81수 x 2칸 = 162칸 으로 딱 맞는다. 즉 **작은 숫자는
큰 숫자를 녹이는 용매**이고, 한 수에 사과를 많이 먹는 것은 그 용매를 태워
큰 숫자를 좌초시키는 짓이다 (탐욕 91점 < 무작위 96.2점 이 여기서 설명된다).

실제로 잔여 구성을 재보면 1 은 14%, 9 는 52% 가 남는다. 즉 지금 모델들은
전부 좌초를 만들고 있는데, **어떤 사과가 좌초되는가를 가리키는 학습 신호가
지금까지 하나도 없었다.** survive 는 그 구조를 트렁크가 표현하도록 만든다.

survive 는 Q 출력에 직접 붙지 않는다. 공유 트렁크를 통해서만 영향을 준다.
Q 출력에 보조 손실을 걸면 V10b/V10c 처럼 Q 의 눈금이 망가진다 (train/loss 가
각각 5180배, 953배로 폭주했다).

──────────────────────────────────────────────────────────────────────────
비용과 감시 지표
──────────────────────────────────────────────────────────────────────────
보조 손실은 TD 배치와 **별개의 배치**로 트렁크를 한 번 더 통과한다. 그만큼
fps 가 떨어진다(대략 -25~35%). 너무 느리면 AUX_BATCH 를 줄이거나 AUX_EVERY 를
2 로 올릴 것.

로그에서 반드시 볼 것:

    train/loss          TD 손실. V1.0 이 3.4e-5 였다. 이것이 100배 이상 커지면
                        보조 손실이 트렁크를 잡아먹은 것이니 가중치를 낮춰야 한다.
    aux/look_weight     Q 가 1수 앞 예측을 얼마나 쓰는가. 0 이면 가설 기각.
    aux/look_corr       예측과 정답의 상관. 1수 앞을 실제로 배웠는지.
    aux/survive_acc     잔존 예측 정확도. 0.5 근처면 아무것도 못 배운 것.
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
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.training import (
    CheckpointSaver, ScoreCallback, TrainConfig, describe_device, model_filename,
    parse_train_args, save_model,
)
import env as env_mod
from model import LOOK_LOG_SCALE, POLICY_CLASS, grid_from_obs, make_policy_kwargs

ALGORITHM = "DQN"
AI_VERSION = "11c"
ROWS, COLS = 9, 18

BUFFER_SIZE = 100_000      # V1.0 과 같은 값

# ── 보조 손실 ────────────────────────────────────────────────────────────
AUX_BATCH = 128            # 보조 손실 전용 배치. 줄이면 fps 가 오른다.
AUX_EVERY = 1              # N 번의 경사 스텝마다 한 번만 보조 손실을 건다
LOOK_SAMPLES = 8           # 상태마다 몇 개의 합법수에 대해 1수 앞을 계산할 것인가
LOOK_LOSS_WEIGHT = 1.0
SURVIVE_LOSS_WEIGHT = 0.2  # 162칸이라 항이 많다. look 보다 낮게 잡는다.
LABEL_CHUNK = 512          # afterstate 라벨 계산의 청크 (메모리 조절용)


class OutcomeReplayBuffer(ReplayBuffer):
    """리플레이 버퍼 + **에피소드 결말 라벨**.

    잔존맵의 정답은 에피소드가 끝나야 알 수 있다. 그래서 전이를 넣을 때는 슬롯
    번호만 기억해 두고, 그 에피소드가 끝나는 순간 되짚어 라벨을 채운다.
    (SB3 의 ReplayBuffer 는 사후 수정 훅이 없어서 add 를 감싸는 수밖에 없다.)

    메모리는 (칸수 x uint8) 만 든다 — 관측 하나가 8.4KB 인데 라벨은 162바이트다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_cells = int(np.prod(self.obs_shape[1:]))          # (채널, R, C) -> R*C
        self.outcome = np.zeros((self.buffer_size, self.n_envs, n_cells), dtype=np.uint8)
        self.outcome_valid = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self._slots: list[list[int]] = [[] for _ in range(self.n_envs)]

    def add(self, obs, next_obs, action, reward, done, infos):
        pos = self.pos
        super().add(obs, next_obs, action, reward, done, infos)

        for i in range(self.n_envs):
            # 새로 덮어쓴 슬롯은 아직 결말을 모른다. 이전 에피소드의 라벨이
            # 남아 있으면 그 라벨이 엉뚱한 전이에 붙는다.
            self.outcome_valid[pos, i] = False
            self._slots[i].append(pos)

            if not done[i]:
                continue

            slots = self._slots[i]
            self._slots[i] = []
            final = infos[i].get("final_grid")
            if final is None:
                continue                      # 불법 수로 truncated 된 경우 등
            if len(slots) > self.buffer_size:
                continue                      # 에피소드가 버퍼보다 길면 앞부분이 이미 덮였다

            survived = (np.asarray(final).reshape(-1) != 0).astype(np.uint8)
            self.outcome[slots, i] = survived
            self.outcome_valid[slots, i] = True

    def sample_outcome(self, batch_size: int):
        """결말 라벨이 붙은 전이에서만 (관측, 잔존라벨) 을 뽑는다."""
        upper = self.buffer_size if self.full else self.pos
        if upper == 0:
            return None
        valid = np.flatnonzero(self.outcome_valid[:upper].reshape(-1))
        if valid.size == 0:
            return None

        pick = valid[np.random.randint(0, valid.size, size=batch_size)]
        pos, env_idx = np.unravel_index(pick, (upper, self.n_envs))
        return self.observations[pos, env_idx], self.outcome[pos, env_idx]


class OutcomePredictionDQN(DQN):
    """ε-greedy 를 마스킹하고, TD 손실에 결과 예측 보조 손실을 더한 DQN."""

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
        # SB3 는 워밍업 동안 predict() 를 건너뛰고 7533개에서 균등추출한다 (합법 확률 0.66%)
        if self.num_timesteps < learning_starts:
            assert self._last_obs is not None
            action = self._masked_random(self._last_obs)
            return action, action
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if deterministic or np.random.rand() >= self.exploration_rate:
            # super().predict() 를 부르면 안 된다. SB3 의 predict 가 ε 판정을 한 번 더
            # 하고 그쪽 무작위 분기는 마스킹이 없다 (문서 §4 버그 5번).
            return self.policy.predict(observation, state, episode_start, deterministic)
        batched = self.policy.is_vectorized_observation(observation)
        obs = observation if batched else observation[None]
        action = self._masked_random(obs)
        return (action if batched else action[0]), state

    # ── 학습 (TD + 보조 손실을 한 번의 backward 로) ───────────────────────
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        td_losses, look_losses, survive_losses = [], [], []
        look_corrs, survive_accs = [], []

        for step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with torch.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())
            td_loss = F.smooth_l1_loss(current_q_values, target_q_values)
            td_losses.append(td_loss.item())

            loss = td_loss
            if step % AUX_EVERY == 0:
                auxiliary = self._auxiliary_losses()
                if auxiliary is not None:
                    look_loss, survive_loss, look_corr, survive_acc = auxiliary
                    loss = (loss + LOOK_LOSS_WEIGHT * look_loss
                            + SURVIVE_LOSS_WEIGHT * survive_loss)
                    look_losses.append(look_loss.item())
                    survive_losses.append(survive_loss.item())
                    look_corrs.append(look_corr)
                    survive_accs.append(survive_acc)

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(td_losses))
        self.logger.record("aux/look_weight", float(self.q_net.look_weight.item()))
        if look_losses:
            self.logger.record("aux/look_loss", float(np.mean(look_losses)))
            self.logger.record("aux/look_corr", float(np.mean(look_corrs)))
            self.logger.record("aux/survive_loss", float(np.mean(survive_losses)))
            self.logger.record("aux/survive_acc", float(np.mean(survive_accs)))

    def _auxiliary_losses(self):
        """(look 손실, survive 손실, look 상관, survive 정확도). 데이터가 없으면 None."""
        batch = self.replay_buffer.sample_outcome(AUX_BATCH)  # type: ignore[union-attr]
        if batch is None:
            return None
        obs_np, survive_np = batch
        obs = obs_as_tensor(obs_np, self.device)

        _, look, survive_logit, mask = self.q_net.heads(obs)

        # ── ① 1수 앞 합법수 ──────────────────────────────────────────────
        keep = mask.any(dim=1)
        look_loss = torch.zeros((), device=self.device)
        look_corr = 0.0
        if bool(keep.any()):
            picks = torch.multinomial(mask[keep].float(), LOOK_SAMPLES, replacement=True)
            with torch.no_grad():
                counts = self.q_net.index.afterstate_legal_counts(
                    grid_from_obs(obs[keep]), picks, chunk=LABEL_CHUNK)
                target = torch.log1p(counts) / LOOK_LOG_SCALE
            predicted = look[keep].gather(1, picks)
            look_loss = F.mse_loss(predicted, target)
            look_corr = _correlation(predicted.detach(), target)

        # ── ② 잔존맵 ────────────────────────────────────────────────────
        label = torch.as_tensor(survive_np, device=self.device, dtype=torch.float32)
        label = label.view(-1, self.q_net.rows, self.q_net.cols)
        occupied = obs[:, 0] < 0.5                       # 0번 평면 = 빈칸
        survive_loss = torch.zeros((), device=self.device)
        survive_acc = 0.0
        if bool(occupied.any()):
            logit = survive_logit[occupied]
            truth = label[occupied]
            survive_loss = F.binary_cross_entropy_with_logits(logit, truth)
            survive_acc = float(((logit > 0).float() == truth).float().mean().item())

        return look_loss, survive_loss, look_corr, survive_acc


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """피어슨 상관. 두 텐서 모두 평평하게 편다. 표준편차가 0 이면 0 을 돌려준다."""
    x = a.reshape(-1).float()
    y = b.reshape(-1).float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    return float((x @ y / denom).item()) if denom > 0 else 0.0


def _env():
    def _init():
        factory = getattr(env_mod, "make_train_env", env_mod.make_env)
        return factory(ROWS, COLS, render_mode=None)   # 마스킹은 신경망 안에서 한다
    return _init


def generate_env(config: TrainConfig):
    return VecMonitor(SubprocVecEnv([_env() for _ in range(config.n_envs)]))


def train_model(env, config: TrainConfig):
    # 하이퍼파라미터는 V1.0(114.48점) 과 완전히 동일하다. 바꾼 것은 보조 목표 둘과
    # (그에 따라) 셰이핑 제거뿐이다.
    model = OutcomePredictionDQN(
        POLICY_CLASS,
        env,
        learning_rate=1e-4,
        buffer_size=BUFFER_SIZE,
        replay_buffer_class=OutcomeReplayBuffer,
        learning_starts=20_000,
        batch_size=256,
        tau=1.0,
        gamma=0.997,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=5_000,
        exploration_fraction=0.3,
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
