"""DQN V13 — **롤아웃 정책반복.** 가치를 추정하지 않고 자기 정책으로 끝까지 두어 본다.

    매 수:   후보 K개를 각각 끝까지 롤아웃 -> 최종 점수 최고를 둔다  (= 정책 개선)
    학습:    그 선택을 정답으로 정책 헤드에 margin 손실             (= 정책 평가/증류)
    순환:    정책이 좋아지면 롤아웃이 좋아지고 -> 선택이 좋아지고 -> 다시 학습

──────────────────────────────────────────────────────────────────────────
왜 이 방향인가 — V6~V12 가 남긴 것
──────────────────────────────────────────────────────────────────────────
**끝난 줄기(평가함수 개선).** V12b 는 최종 잔여 사과를 평균 0.81개 오차로 맞히는
가치를 갖고도 조잡한 휴리스틱을 쓰는 V1.0 과 통계적으로 같은 점수를 냈다
(−0.19 ± 0.86). V6~V12 가 전부 이 줄기였고, 이제 닫혔다는 것이 실측으로 확정됐다.

**열린 줄기(평가의 편향).** 같은 시드 20판:

    min-area (base 정책)                      ~107
    1수앞 휴리스틱                             118.20
    빔 깊이 2 / 3 (평가 고정)                  119.20 / 119.00   <- 깊이만으로는 +1
    **rollout-1ply (후보마다 끝까지 두어 봄)    122.90**          <- 평가를 정확히 하면 +16

롤아웃은 K=1 이라 잡음이 std ~5 로 V12b 의 MAE 0.81 보다 6배 크다. 그런데도 이긴다.
**정밀하지만 편향된 추정보다, 잡음이 있어도 편향 없는 추정이 낫다.**

편향의 정체는 **자기확인 고리**다. V^π 는 자기 정책이 지나간 상태에서만 학습되는데
(ε=0.05 면 데이터의 95%), 개선 연산자가 묻는 형제 27개는 분포 밖이다. 가치가 자기
선택을 계속 1등으로 매기면 정책은 영원히 안 바뀐다. 롤아웃은 그 형제를 **실제로
두어 보므로** 고리를 끊는다.

──────────────────────────────────────────────────────────────────────────
과거 실패와 무엇이 다른가
──────────────────────────────────────────────────────────────────────────
| 버전 | 실패 원인 | V13 |
|---|---|---|
| V10c 탐색교사 | Φ 를 빠뜨려 교사가 **반대 방향**을 가리킴 | 롤아웃은 실제 점수라 Φ 개념 자체가 없다 |
| V10c/V10b 모방 | Q 에 CE 를 걸어 눈금 파괴 (TD 손실 5,180배) | 로짓은 **정책**이라 눈금이 없다. 게다가 margin 손실 |
| V11a 자기모방 | **교사가 학생과 같았다** (margin_loss 0.039 < λ) | 교사가 측정으로 +8 강하다 (122.9 vs 114.3) |
| V12 가치개선 | 자기확인 고리 | 롤아웃이 분포 밖을 직접 평가 |

──────────────────────────────────────────────────────────────────────────
비용
──────────────────────────────────────────────────────────────────────────
롤아웃은 전부 GPU 텐서 연산이고 후보 전체가 **한 배치로 lockstep** 으로 돈다.
후보 16개면 한 번에 16판을 동시에 진행하므로, 한 수 결정에 드는 것은
`남은 수 x (배치 16 forward)` 다. 게임 엔진을 부르지 않는다.

느리면 model.py 의 `ROLLOUT_TOPK` 를 줄인다 (비용이 정비례한다).

**경량 모델**은 같은 가중치를 `DQN/V13L`(ROLLOUT_TOPK=0)로 로드하면 된다 —
정책 로짓 argmax, 1회 forward. V9c/V9cB 와 같은 방식이다.
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
from model import POLICY_CLASS, make_policy_kwargs

ALGORITHM = "DQN"
AI_VERSION = "13"
ROWS, COLS = 9, 18

BUFFER_SIZE = 100_000
BATCH_SIZE = 256

# margin 손실. 로짓은 **정책**이라 눈금이 없으므로 V11a(0.05, 보상 단위)와 값이 다르다.
# 3.0 이면 고른 수가 나머지보다 확률 20배 앞선 지점에서 손실이 0 이 된다 —
# argmax 는 확실히 바뀌면서도 롤아웃이 쓸 다양성은 남는다.
POLICY_MARGIN = 3.0
VALUE_LOSS_WEIGHT = 0.5


class ImprovedActionBuffer(ReplayBuffer):
    """리플레이 버퍼 + (이 수가 롤아웃이 고른 것인가, 이 판의 최종 점수는 얼마였나).

    두 가지를 더 들고 있어야 한다.
      greedy  — ε 탐험으로 무작위로 둔 수는 **모방하면 안 된다.** 그건 교사의
                선택이 아니다. V10b 가 '운 좋은 무작위 수' 를 모방해 실패한 것과
                같은 함정이다.
      remain  — 이 상태에서 끝까지 실제로 얻은 점수. 가치 헤드의 몬테카를로 목표.
                에피소드가 끝나야 알 수 있으므로 슬롯을 기억해 두고 되짚어 채운다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.greedy = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.remain = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.labelled = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self._slots: list[list[int]] = [[] for _ in range(self.n_envs)]
        self._before: list[list[float]] = [[] for _ in range(self.n_envs)]
        self._prev_score = np.zeros(self.n_envs, dtype=np.float32)
        self._greedy_flags: np.ndarray | None = None      # 학습 루프가 매 스텝 채운다

    def add(self, obs, next_obs, action, reward, done, infos):
        pos = self.pos
        flags = (self._greedy_flags if self._greedy_flags is not None
                 else np.ones(self.n_envs, dtype=bool))
        super().add(obs, next_obs, action, reward, done, infos)

        for i in range(self.n_envs):
            self.labelled[pos, i] = False        # 덮어쓴 슬롯은 아직 결말을 모른다
            self.greedy[pos, i] = bool(flags[i])
            self._slots[i].append(pos)
            # info["score"] 는 이 수를 **둔 뒤**의 점수다. 이 상태(obs)에서 앞으로
            # 얻을 양은 (최종 점수 - 이 수를 두기 **전**의 점수) 이므로 직전 값을 쓴다.
            self._before[i].append(float(self._prev_score[i]))
            self._prev_score[i] = float(infos[i].get("score", 0.0))

            if not done[i]:
                continue

            slots, before = self._slots[i], self._before[i]
            self._slots[i], self._before[i] = [], []
            final = float(infos[i].get("score", 0.0))
            self._prev_score[i] = 0.0            # 다음 에피소드는 0 점에서 시작한다
            if len(slots) > self.buffer_size:
                continue
            self.remain[slots, i] = final - np.asarray(before, dtype=np.float32)
            self.labelled[slots, i] = True

    def sample_improved(self, batch_size: int):
        """롤아웃이 고른 수 + 결말 라벨이 둘 다 있는 전이만."""
        upper = self.buffer_size if self.full else self.pos
        if upper == 0:
            return None
        usable = self.labelled[:upper] & self.greedy[:upper]
        valid = np.flatnonzero(usable.reshape(-1))
        if valid.size == 0:
            return None
        pick = valid[np.random.randint(0, valid.size, size=batch_size)]
        pos, env_idx = np.unravel_index(pick, (upper, self.n_envs))
        return (self.observations[pos, env_idx],
                self.actions[pos, env_idx, 0],
                self.remain[pos, env_idx])


class RolloutPolicyIterationDQN(DQN):
    """SB3 DQN 의 껍데기만 쓴다. TD 도, 타깃망도, 보상도 학습에 안 들어간다.

    껍데기를 쓰는 이유는 `runs/measure.py` 가 MODEL_REGISTRY 를 통해 SB3 클래스로
    로드하기 때문이다. 행동 선택은 `q_net.forward()` 안에 있으므로 하네스는
    한 줄도 바꿀 필요가 없다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._greedy_flags: np.ndarray | None = None

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_greedy_flags"]

    # ── 탐험 ─────────────────────────────────────────────────────────────
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
        """워밍업/ε 구간은 무작위, 나머지는 **롤아웃이 고른 수**.

        어느 쪽이었는지 기록해 둔다 — 무작위로 둔 수를 모방하면 안 되기 때문이다.
        """
        assert self._last_obs is not None
        n = self._last_obs.shape[0]

        if self.num_timesteps < learning_starts:
            self._greedy_flags = np.zeros(n, dtype=bool)
            action = self._masked_random(self._last_obs)
            return action, action

        explore = np.random.rand(n) < self.exploration_rate
        action = np.empty(n, dtype=np.int64)
        if explore.any():
            action[explore] = self._masked_random(self._last_obs[explore])
        if (~explore).any():
            # policy.predict -> q_net.forward -> 롤아웃 (model.py 의 ROLLOUT_TOPK)
            chosen, _ = self.policy.predict(self._last_obs[~explore], deterministic=True)
            action[~explore] = np.asarray(chosen).reshape(-1)
        self._greedy_flags = ~explore
        return action, action

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if deterministic or np.random.rand() >= self.exploration_rate:
            # super().predict() 는 ε 판정을 한 번 더 하고 그쪽은 마스킹이 없다 (문서 §4 버그 5번)
            return self.policy.predict(observation, state, episode_start, deterministic)
        batched = self.policy.is_vectorized_observation(observation)
        obs = observation if batched else observation[None]
        action = self._masked_random(obs)
        return (action if batched else action[0]), state

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        replay_buffer._greedy_flags = self._greedy_flags   # type: ignore[attr-defined]
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    # ── 학습 (정책 증류 + 가치 보조) ─────────────────────────────────────
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        margins, agrees, value_losses, value_mae = [], [], [], []
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample_improved(batch_size)  # type: ignore[union-attr]
            if batch is None:
                return
            obs_np, action_np, remain_np = batch
            obs = obs_as_tensor(obs_np, self.device)
            teacher = torch.as_tensor(action_np, device=self.device, dtype=torch.long)
            target = torch.as_tensor(remain_np, device=self.device, dtype=torch.float32)

            q_net = self.policy.q_net
            logits = q_net.policy_logits(obs)

            # DQfD large-margin: 교사 행동이 λ 만큼 앞서면 정확히 0 이 된다.
            # 불법 수는 -1e8 이라 max 후보로 올라오지 못한다.
            margin = torch.full_like(logits, POLICY_MARGIN)
            margin.scatter_(1, teacher[:, None], 0.0)
            chosen = logits.gather(1, teacher[:, None]).squeeze(1)
            margin_loss = ((logits + margin).max(dim=1).values - chosen).mean()

            value = q_net.predict_value(obs)
            value_loss = F.mse_loss(value, target / q_net.n_cells)

            loss = margin_loss + VALUE_LOSS_WEIGHT * value_loss
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            with torch.no_grad():
                margins.append(margin_loss.item())
                agrees.append((logits.argmax(dim=1) == teacher).float().mean().item())
                value_losses.append(value_loss.item())
                value_mae.append(((value * q_net.n_cells) - target).abs().mean().item())

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/margin_loss", float(np.mean(margins)))
        # 정책이 롤아웃의 선택을 얼마나 따라잡았는가. 1 에 가까우면 **경량 모델이
        # 무거운 모델을 거의 재현**한다는 뜻이고, 낮으면 증류가 덜 된 것이다.
        self.logger.record("train/teacher_agreement", float(np.mean(agrees)))
        self.logger.record("train/value_loss", float(np.mean(value_losses)))
        # V12b 의 leftover_mae(0.808)와 같은 자. 가치는 진단용일 뿐 선택에 안 쓰인다.
        self.logger.record("train/value_mae", float(np.mean(value_mae)))


def _env():
    def _init():
        factory = getattr(env_mod, "make_train_env", env_mod.make_env)
        return factory(ROWS, COLS, render_mode=None)
    return _init


def generate_env(config: TrainConfig):
    return VecMonitor(SubprocVecEnv([_env() for _ in range(config.n_envs)]))


def train_model(env, config: TrainConfig):
    model = RolloutPolicyIterationDQN(
        POLICY_CLASS,
        env,
        learning_rate=2e-4,
        buffer_size=BUFFER_SIZE,
        replay_buffer_class=ImprovedActionBuffer,
        # 롤아웃이 비싸므로 워밍업을 짧게 잡는다. 무작위 구간은 교사가 없어 정책
        # 학습에 기여하지 못한다 (가치 헤드만 배운다).
        learning_starts=5_000,
        batch_size=BATCH_SIZE,
        tau=1.0, gamma=1.0, target_update_interval=10 ** 9,   # 전부 안 쓰인다
        train_freq=4,
        gradient_steps=1,
        # ε 는 낮게. 롤아웃 자체가 이미 정책보다 나은 수를 찾아 주므로 무작위 탐험이
        # 할 일이 적고, 무작위로 둔 수는 교사 데이터가 되지 못한다.
        exploration_fraction=0.1,
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
