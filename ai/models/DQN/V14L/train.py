"""DQN V14L — **학습용이 아니다.** V14 체크포인트를 다른 배포 모드로 재기 위한 폴더.

    cp ai/models/DQN/V14/models/V14_DQN_<스텝>.zip \n       ai/models/DQN/V14L/models/V14L_DQN_<스텝>.zip
    python runs/measure.py --checkpoint ai/models/DQN/V14L/models/V14L_DQN_<스텝>.zip

model.py 의 MODE 만 "policy" 로 다르다 (1회 forward (정책 로짓 argmax). **경량 모델.** 0.006초/수).
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
from model import POLICY_CLASS, make_policy_kwargs

ALGORITHM = "DQN"
AI_VERSION = "14L"
ROWS, COLS = 9, 18

BUFFER_SIZE = 200_000
BATCH_SIZE = 256

# 정책 로짓은 **가치가 아니라 정책**이라 눈금이 없다. 3.0 이면 고른 수가 나머지보다
# 확률 20배 앞선 지점에서 손실이 0 이 된다.
POLICY_MARGIN = 3.0
VALUE_LOSS_WEIGHT = 0.5

# **워밍스타트.** V12b 의 인코더와 가치 헤드를 그대로 물려받아 1스텝째부터 강한
# 빔 교사를 쓴다. 없으면 무작위 초기화로 시작하는데, 그러면 초반 빔이 "사과 많이
# 먹기"(91점) 근처에서 놀다가 아주 천천히 올라온다.
INIT_FROM = ROOT / "ai/models/DQN/V12b/models/V12b_DQN_6000000.zip"


class BeamTeacherBuffer(ReplayBuffer):
    """리플레이 버퍼 + (이 수가 빔이 고른 것인가, 이 상태에서 앞으로 얻은 점수).

    V13 의 ImprovedActionBuffer 와 같다. 무작위로 둔 수를 모방하면 안 되므로
    (V10b 가 '운 좋은 무작위 수' 를 모방해 실패했다) 교사 여부를 따로 들고 있고,
    가치 목표는 에피소드가 끝나야 알 수 있으므로 슬롯을 기억해 두고 되짚어 채운다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.remain = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.labelled = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self._slots: list[list[int]] = [[] for _ in range(self.n_envs)]
        self._before: list[list[float]] = [[] for _ in range(self.n_envs)]
        self._prev_score = np.zeros(self.n_envs, dtype=np.float32)
        self._teacher_flags: np.ndarray | None = None

    def add(self, obs, next_obs, action, reward, done, infos):
        pos = self.pos
        flags = (self._teacher_flags if self._teacher_flags is not None
                 else np.ones(self.n_envs, dtype=bool))
        super().add(obs, next_obs, action, reward, done, infos)

        for i in range(self.n_envs):
            self.labelled[pos, i] = False       # 덮어쓴 슬롯은 아직 결말을 모른다
            self.teacher[pos, i] = bool(flags[i])
            self._slots[i].append(pos)
            # info["score"] 는 이 수를 둔 **뒤**의 점수다. 이 상태(obs)에서 앞으로 얻을
            # 양은 (최종 점수 - 이 수를 두기 전의 점수) 이므로 직전 값을 쓴다.
            self._before[i].append(float(self._prev_score[i]))
            self._prev_score[i] = float(infos[i].get("score", 0.0))
            if not done[i]:
                continue

            slots, before = self._slots[i], self._before[i]
            self._slots[i], self._before[i] = [], []
            final = float(infos[i].get("score", 0.0))
            self._prev_score[i] = 0.0
            if len(slots) > self.buffer_size:
                continue
            self.remain[slots, i] = final - np.asarray(before, dtype=np.float32)
            self.labelled[slots, i] = True

    def sample_teacher(self, batch_size: int):
        upper = self.buffer_size if self.full else self.pos
        if upper == 0:
            return None
        valid = np.flatnonzero((self.labelled[:upper] & self.teacher[:upper]).reshape(-1))
        if valid.size == 0:
            return None
        pick = valid[np.random.randint(0, valid.size, size=batch_size)]
        pos, env_idx = np.unravel_index(pick, (upper, self.n_envs))
        return (self.observations[pos, env_idx],
                self.actions[pos, env_idx, 0],
                self.remain[pos, env_idx])


class BeamTeacherDQN(DQN):
    """SB3 DQN 의 껍데기만 쓴다. TD 도, 타깃망도, 보상도 학습에 안 들어간다.

    껍데기를 쓰는 이유는 `runs/measure.py` 가 MODEL_REGISTRY 를 통해 SB3 클래스로
    로드하기 때문이다. 행동 선택(빔 계획)은 `q_net.forward()` 안에 있으므로
    하네스는 한 줄도 바꿀 필요가 없다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._teacher_flags: np.ndarray | None = None

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_teacher_flags"]

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
        """항상 빔의 계획을 따른다. ε 탐험을 쓰지 않는다.

        판이 정해지면 무작위성이 없어 계획에서 벗어날 이유가 없고, 무작위로 둔 수는
        교사 라벨이 되지도 못한다. 다양성은 매 판 새로 뽑히는 판이 제공한다.
        """
        assert self._last_obs is not None
        chosen, _ = self.policy.predict(self._last_obs, deterministic=True)
        action = np.asarray(chosen).reshape(-1)
        self._teacher_flags = np.ones(action.shape[0], dtype=bool)
        return action, action

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        return self.policy.predict(observation, state, episode_start, deterministic=True)

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        replay_buffer._teacher_flags = self._teacher_flags   # type: ignore[attr-defined]
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    # ── 학습 (정책 증류 + 가치 회귀) ─────────────────────────────────────
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        margins, agrees, value_losses, value_mae = [], [], [], []
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample_teacher(batch_size)  # type: ignore[union-attr]
            if batch is None:
                return
            obs_np, action_np, remain_np = batch
            obs = obs_as_tensor(obs_np, self.device)
            teacher = torch.as_tensor(action_np, device=self.device, dtype=torch.long)
            target = torch.as_tensor(remain_np, device=self.device, dtype=torch.float32)

            q_net = self.policy.q_net
            logits = q_net.policy_logits(obs)

            # DQfD large-margin. 교사 행동이 λ 만큼 앞서면 정확히 0 이 된다.
            margin = torch.full_like(logits, POLICY_MARGIN)
            margin.scatter_(1, teacher[:, None], 0.0)
            chosen = logits.gather(1, teacher[:, None]).squeeze(1)
            margin_loss = ((logits + margin).max(dim=1).values - chosen).mean()

            # 가치는 '이 상태에서 앞으로 얻을 점수' 를 맞힌다. 빔이 이것으로 가지를 친다.
            leftover_pred = q_net.expected_leftover(obs)
            occupied = (obs[:, 0] < 0.5).flatten(1).sum(dim=1)
            value_loss = F.mse_loss((occupied - leftover_pred) / q_net.n_cells,
                                    target / q_net.n_cells)

            loss = margin_loss + VALUE_LOSS_WEIGHT * value_loss
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            with torch.no_grad():
                margins.append(margin_loss.item())
                agrees.append((logits.argmax(dim=1) == teacher).float().mean().item())
                value_losses.append(value_loss.item())
                value_mae.append(((occupied - leftover_pred) - target).abs().mean().item())

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/margin_loss", float(np.mean(margins)))
        # 정책이 빔의 선택을 얼마나 따라잡았는가. **경량 모델의 실력을 좌우한다.**
        self.logger.record("train/teacher_agreement", float(np.mean(agrees)))
        self.logger.record("train/value_loss", float(np.mean(value_losses)))
        # 빔의 가지치기 품질. V12b 는 0.808 이었다. **이 값이 내려가면 빔이 강해진다.**
        self.logger.record("train/value_mae", float(np.mean(value_mae)))


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
    model = BeamTeacherDQN(
        POLICY_CLASS,
        env,
        learning_rate=2e-4,
        buffer_size=BUFFER_SIZE,
        replay_buffer_class=BeamTeacherBuffer,
        learning_starts=2_000,
        batch_size=BATCH_SIZE,
        # 아래 넷은 상속 때문에 필요할 뿐 손실에 들어가지 않는다.
        tau=1.0, gamma=1.0, target_update_interval=10 ** 9,
        exploration_initial_eps=0.0, exploration_final_eps=0.0, exploration_fraction=1.0,
        train_freq=4,
        # 빔이 만든 라벨은 비싸다(판당 2.7초). 한 번 모은 것을 여러 번 학습한다.
        gradient_steps=2,
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
    raise SystemExit(
        "V14L 은 측정 전용 폴더입니다. "
        "학습은 DQN/V14 에서 하고, 그 체크포인트를 복사해 measure.py 로 재세요."
    )

    config = parse_train_args(VERSION_DIR)
    print(describe_device(config.device))
    print(f"{config.n_envs}개 코어 병렬 처리 환경 구축")
    env = generate_env(config)
    model = train_model(env, config)
    evaluate_model(model)
