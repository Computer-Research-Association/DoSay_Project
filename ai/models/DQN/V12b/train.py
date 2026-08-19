"""DQN V12b — **V12a 의 대조군. 가치 헤드만 스칼라로 바꿨다.**

    (V12a)  칸마다 '남을 확률' 162개를 BCE 로 배우고, 그 합을 가치로 쓴다.
    (V12b)  '최종 잔여 사과 수' **스칼라 하나**를 바로 회귀한다.

애프터스테이트 평가, 몬테카를로 라벨, 트렁크, 하이퍼파라미터, 환경 — 전부 같다.
**단 하나의 변인이 감독의 밀도다.**

probe_v11c 에서 잔존맵을 애프터스테이트에 씌우면 Q argmax 대비 +4.50 이 나왔는데,
그 이득의 출처가 두 갈래다.

    (가) 조밀한 감독 (상태당 라벨 162개)
    (나) 애프터스테이트 평가 (예측 대신 실제 판을 만들어 잼)

    V12a − V12b       = (가) 조밀한 감독의 순효과
    V12b − V1.0(114.49) = (나) 애프터스테이트 평가의 순효과

이 프로젝트가 V4 부터 계속 아쉬웠던 것이 이런 대조군이다. V9c 는 "이긴 조합 둘을
합쳤는데 최저" 였고 원인을 아직 모른다. V11a 는 변인이 둘(듀얼링 + 그룹 엘리트)이라
귀속이 안 됐다. 이번엔 한 변인만 다르다.

**둘 다 V1.0 보다 낮게 나오면** 애프터스테이트 평가 자체가 답이 아니라는 뜻이고,
그때는 probe 의 +4.50 이 "V11c 의 Q 가 유독 나빴던 것"에 불과했다는 결론이 된다.
그것도 확실한 정보다.

나머지 설명(왜 TD 를 버리는가, 감시할 지표)은 V12a train.py 와 같다.
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
from model import POLICY_CLASS, grid_from_obs as model_grid_from_obs, make_policy_kwargs

ALGORITHM = "DQN"
AI_VERSION = "12b"
ROWS, COLS = 9, 18

BUFFER_SIZE = 100_000     # V1.0 과 같은 값. 관측 8.4KB x 2 x 10만 = 약 1.7GB
BATCH_SIZE = 256
SIBLING_EVERY = 200       # 이만큼의 경사 스텝마다 형제 서열 진단을 한 번


class OutcomeReplayBuffer(ReplayBuffer):
    """리플레이 버퍼 + **에피소드 결말 라벨**. V11c 에서 검증된 것을 그대로 쓴다.

    잔존맵의 정답은 에피소드가 끝나야 알 수 있다. 전이를 넣을 때는 슬롯 번호만
    기억해 두고, 그 에피소드가 끝나는 순간 되짚어 라벨을 채운다.
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
            # 새로 덮어쓴 슬롯은 아직 결말을 모른다. 이전 에피소드의 라벨이 남아
            # 있으면 그 라벨이 엉뚱한 전이에 붙는다.
            self.outcome_valid[pos, i] = False
            self._slots[i].append(pos)

            if not done[i]:
                continue
            slots = self._slots[i]
            self._slots[i] = []
            final = infos[i].get("final_grid")
            if final is None or len(slots) > self.buffer_size:
                continue                     # 불법 수로 truncated 됐거나 버퍼보다 긴 에피소드
            survived = (np.asarray(final).reshape(-1) != 0).astype(np.uint8)
            self.outcome[slots, i] = survived
            self.outcome_valid[slots, i] = True

    def sample_outcome(self, batch_size: int):
        """결말 라벨이 붙은 전이에서만 (관측, 잔존라벨)."""
        upper = self.buffer_size if self.full else self.pos
        if upper == 0:
            return None
        valid = np.flatnonzero(self.outcome_valid[:upper].reshape(-1))
        if valid.size == 0:
            return None
        pick = valid[np.random.randint(0, valid.size, size=batch_size)]
        pos, env_idx = np.unravel_index(pick, (upper, self.n_envs))
        return self.observations[pos, env_idx], self.outcome[pos, env_idx]


class AfterstateOutcomeDQN(DQN):
    """SB3 DQN 의 껍데기만 쓰고 학습은 통째로 갈아끼운다.

    껍데기를 쓰는 이유는 하나다 — `runs/measure.py` 가 MODEL_REGISTRY 를 통해
    SB3 클래스로 체크포인트를 로드하기 때문이다. 정책의 `q_net.forward()` 가
    애프터스테이트 평가로 Q 를 만들어 주므로 하네스는 아무것도 바꾸지 않아도 된다.

    상속받은 것 중 **쓰지 않는 것**: 타깃망, gamma, TD 손실. (타깃망은 만들어지고
    주기적으로 동기화되지만 손실에 한 번도 들어가지 않는다.)
    """

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

    # ── 학습 (순수 지도학습) ─────────────────────────────────────────────
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses, mae, bias = [], [], []
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample_outcome(batch_size)  # type: ignore[union-attr]
            if batch is None:
                return
            obs_np, survive_np = batch
            obs = obs_as_tensor(obs_np, self.device)
            target = torch.as_tensor(survive_np, device=self.device, dtype=torch.float32)
            target = target.view(-1, self.policy.q_net.rows, self.policy.q_net.cols)

            # 라벨은 V12a 와 같은 배열을 쓰되, 합쳐서 스칼라 하나로 만든다.
            # (칸별 라벨의 합 = 그 판에서 끝까지 남은 사과 수)
            actual = target.sum(dim=(1, 2))
            logit = self.policy.q_net.leftover_logit(obs)
            loss = F.binary_cross_entropy_with_logits(
                logit, actual / self.policy.q_net.n_cells)

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            with torch.no_grad():
                losses.append(loss.item())
                predicted = torch.sigmoid(logit) * self.policy.q_net.n_cells
                mae.append((predicted - actual).abs().mean().item())
                bias.append((predicted - actual).mean().item())

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/leftover_loss", float(np.mean(losses)))
        # 가치의 품질 그 자체. 형제 간 실제 차이가 약 2점이므로 2 아래로 내려가야 한다.
        self.logger.record("train/leftover_mae", float(np.mean(mae)))
        self.logger.record("train/leftover_bias", float(np.mean(bias)))

        if self._n_updates % SIBLING_EVERY < gradient_steps:
            with torch.no_grad():
                self._log_sibling_spread(obs[:8])

    # ── 형제 서열 진단 ───────────────────────────────────────────────────
    # 이 프로젝트는 "절대 정확도가 좋으니 서열도 좋겠지" 로 두 번 데였다:
    #   V7  explained_variance 0.983 -> 탐색이 +2점밖에 못 냄
    #   V11c look_corr 0.992         -> argmax 는 정확한 규칙보다 4.7점 낮음
    # 둘 다 **상태 간** 변동이 커서 나온 지표였고, 정작 필요한 것은 **한 국면 안에서**
    # 형제를 줄 세우는 능력이었다. 그래서 그것을 직접 잰다.
    def _log_sibling_spread(self, obs: torch.Tensor) -> None:
        q_net = self.policy.q_net
        grids = model_grid_from_obs(obs)
        mask = q_net.index.legal_mask_from_grid(grids)
        spreads, ranges = [], []
        for i in range(grids.shape[0]):
            legal = mask[i].nonzero(as_tuple=True)[0]
            if legal.numel() < 2:
                continue
            after = q_net.index.erase(grids[i][None].expand(legal.numel(), *grids.shape[1:]), legal)
            leftover = q_net.expected_leftover(q_net.index.observation(after))
            spreads.append(float(leftover.std()))
            ranges.append(float(leftover.max() - leftover.min()))
        if spreads:
            # 형제 간 실제 가치 차이는 약 2점이다. spread 가 그보다 훨씬 작으면
            # 모델이 형제를 사실상 구분하지 못하는 것이고, argmin 은 잡음이 된다.
            self.logger.record("train/sibling_spread", float(np.mean(spreads)))
            self.logger.record("train/sibling_range", float(np.mean(ranges)))



def _env():
    def _init():
        factory = getattr(env_mod, "make_train_env", env_mod.make_env)
        return factory(ROWS, COLS, render_mode=None)   # 마스킹은 신경망 안에서 한다
    return _init


def generate_env(config: TrainConfig):
    return VecMonitor(SubprocVecEnv([_env() for _ in range(config.n_envs)]))


def train_model(env, config: TrainConfig):
    model = AfterstateOutcomeDQN(
        POLICY_CLASS,
        env,
        # 순수 지도학습이라 TD 때보다 조금 크게 잡아도 안정적이다.
        learning_rate=2e-4,
        buffer_size=BUFFER_SIZE,
        replay_buffer_class=OutcomeReplayBuffer,
        learning_starts=20_000,
        batch_size=BATCH_SIZE,
        # 아래 셋은 상속 때문에 필요할 뿐 손실에 들어가지 않는다 (타깃망도 안 쓴다).
        tau=1.0,
        gamma=1.0,
        target_update_interval=10 ** 9,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.3,
        # 맵이 좋아지려면 다양한 국면을 봐야 한다. V1.0 과 같은 0.05 를 유지한다.
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
