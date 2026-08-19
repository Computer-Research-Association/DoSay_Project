"""DQN V11a — 기조 유지 버전. V10b(자기모방)에서 실측으로 드러난 결함 두 개를 고친다.

기준선은 **V1.0 = 114.48점**(12M, 탐색 없음)이고, V10b 는 같은 계열에서
6M 에 107.65점으로 오히려 떨어졌다. 로그를 되짚어 원인을 두 개 특정했다.

──────────────────────────────────────────────────────────────────────────
결함 1. 엘리트 필터가 '잘 둔 판'이 아니라 '쉬운 판'을 걸렀다
──────────────────────────────────────────────────────────────────────────
5개 모델을 같은 시드 100판으로 이원배치 분산분해하면 판(시드) 성분이 76.1%,
모델 성분이 12.6% 다. 그리고 어떤 모델이든 자기 상위 20% 에피소드를 골라
**다른 모델**의 같은 시드 점수를 보면 그쪽도 자기 평균보다 +14~16점 높다.

    V1.0  상위20% 판의 타모델 평균 129.52  vs 전체 115.20  (+14.32)
    V10a                       131.86  vs      116.05  (+15.81)
    V10b                       131.41  vs      116.91  (+14.50)

즉 V10b 의 EliteBuffer 가 통과시킨 데이터에는 정책 품질 신호가 사실상 없었다.
`train/elite_threshold` 가 107 -> 117 로 오른 것도 "쉬운 판을 점점 더 잘
골라냈다"는 뜻일 뿐이다.

-> **고침**: 엘리트를 전역 백분위가 아니라 **같은 판 그룹 안에서** 뽑는다.
   env.py 가 한 판을 GROUP_SIZE 번 반복해 주므로, 그 K개끼리만 비교하면 판 운이
   통계적으로 줄어드는 게 아니라 **정확히 상쇄**된다. 남는 차이는 정책의 차이뿐이다.
   실측 헤드룸도 충분하다 (같은 판 best-of-12 = 평균 +8.62, 판 안 std 5.41).

──────────────────────────────────────────────────────────────────────────
결함 2. Q 에 교차엔트로피를 걸어서 Q 를 부쉈다
──────────────────────────────────────────────────────────────────────────
V10b/V10c 는 마스킹된 Q 를 로짓처럼 보고 F.cross_entropy 를 걸었다. CE 는 로짓
간격을 **무한히 벌리려** 하는데 Q 는 보상 단위(에피소드 총합 ~0.7)로 눈금이
맞아 있어야 한다. 양립할 수 없고, 대가는 TD 손실이 치렀다.

    train/loss (최종)   V1.0  3.39e-05      1x
                        V10a  1.51e-03     44x
                        V10c  3.23e-02    953x   (CE 가중치 1.0)
                        V10b  1.76e-01   5180x   (CE 가중치 0.5)

CE 가중치 순서가 그대로 TD 손실 파괴 순서다.

-> **고침**: DQfD 의 large-margin 손실로 바꾼다.

       J_E = max_a [ Q(s,a) + λ·1(a != a_E) ] - Q(s,a_E)

   교사 행동의 Q 가 나머지 합법수보다 λ 만큼만 앞서면 손실이 정확히 0 이 되어
   더는 밀지 않는다. **유계이고 Q 의 눈금을 건드리지 않는다.**
   그리고 V10b 처럼 옵티마이저 스텝을 따로 밟지 않고 TD 손실과 **같은 backward** 에
   넣는다 (따로 밟으면 모방 목적함수의 실효 학습률만 두 배가 된다).

──────────────────────────────────────────────────────────────────────────
결함이 아니라 개선: 듀얼링
──────────────────────────────────────────────────────────────────────────
model.py 참고. Q = V(s) + (A - mean A) 로 공통항(0.5 규모)과 행동항(0.01 규모)을
갈라 놓는다. 문서 §3 의 "V 는 판 전체는 맞히는데 형제 서열을 못 매긴다"에 대한
구조적 대응이다.

**변인이 둘(듀얼링 + 그룹 엘리트)이므로 귀속이 애매해질 수 있다.** 그래서
아래 진단 지표를 전부 남긴다. 특히:

    elite/group_spread  그룹 안 점수 std. **0 에 가까워지면 신호가 사라진 것**이고,
                        그때는 모방이 아무 일도 안 하므로 남는 차이는 듀얼링뿐이다.
    train/loss          V10b 처럼 폭주하지 않는지. margin 손실이면 안 그래야 한다.
    elite/margin_loss   0 으로 수렴하면 정책이 이미 엘리트를 따라가고 있다는 뜻.
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
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.training import (
    CheckpointSaver, ScoreCallback, TrainConfig, describe_device, model_filename,
    parse_train_args, save_model,
)
import env as env_mod
from model import POLICY_CLASS, make_policy_kwargs  # 같은 폴더

ALGORITHM = "DQN"
AI_VERSION = "11a"
ROWS, COLS = 9, 18

# 관측이 13x9x18 float32 = 8.4KB 라 버퍼 1칸당 obs/next_obs 16.8KB. V1.0 과 같은 값.
BUFFER_SIZE = 100_000

# ── 그룹 엘리트 모방 ──────────────────────────────────────────────────────
ELITE_CAPACITY = 60_000    # 채택된 (관측, 행동) 저장 한도
ELITE_BATCH = 128
ELITE_WARMUP = 2_000       # 이만큼 쌓이기 전에는 모방하지 않는다
ELITE_LOSS_WEIGHT = 0.5
# 그룹 최고가 그룹 평균보다 이만큼(사과 개수)은 앞서야 채택한다. 동점에 가까운
# 그룹은 정보가 없는데 모방하면 잡음만 넣는 꼴이다.
ELITE_MIN_GAP = 2.0
# large-margin 손실의 λ. Q 는 보상 단위(에피소드 총합 ~0.7)이고 형제 간 실제
# 가치 차이가 약 0.012(2점)이므로, 그보다 뚜렷하되 Q 스케일을 망가뜨리지 않는 값.
ELITE_MARGIN = 0.05


class GroupEliteBuffer:
    """**같은 판** 그룹 안에서 가장 잘 풀린 롤아웃의 (관측, 행동)만 모은다.

    env.py 의 GroupBoardEnv 가 한 판을 GROUP_SIZE 번 반복하고 info 에 group_index 를
    실어 준다. group_index 가 바뀌는 순간이 그룹 마감이다.

    메모리는 환경당 에피소드 2개분(진행 중 + 현재 최고)만 든다. 50수 x 8.4KB x 2 x
    환경 11개 = 약 9MB.
    """

    def __init__(self, n_envs: int, capacity: int = ELITE_CAPACITY,
                 min_gap: float = ELITE_MIN_GAP) -> None:
        self.capacity = capacity
        self.min_gap = min_gap

        self.obs: list[np.ndarray] = []
        self.actions: list[int] = []

        self._current: list[list[tuple[np.ndarray, int]]] = [[] for _ in range(n_envs)]
        self._best: list[tuple[float, list[tuple[np.ndarray, int]]] | None] = [None] * n_envs
        self._group: list[int] = [-1] * n_envs
        self._scores: list[list[float]] = [[] for _ in range(n_envs)]

        # 진단용 (그룹이 마감될 때마다 갱신)
        self.groups_closed = 0
        self.groups_accepted = 0
        self.spread_sum = 0.0     # 그룹 안 점수 std 의 누적
        self.gap_sum = 0.0        # (최고 - 평균) 의 누적

    # ── 수집 ─────────────────────────────────────────────────────────────
    def record_step(self, env_idx: int, obs: np.ndarray, action: int) -> None:
        self._current[env_idx].append((obs, action))

    def end_episode(self, env_idx: int, score: float, group_index: int) -> None:
        if group_index != self._group[env_idx]:
            self._close_group(env_idx)
            self._group[env_idx] = group_index

        self._scores[env_idx].append(score)
        best = self._best[env_idx]
        if best is None or score > best[0]:
            self._best[env_idx] = (score, self._current[env_idx])
        self._current[env_idx] = []

    def _close_group(self, env_idx: int) -> None:
        scores = self._scores[env_idx]
        best = self._best[env_idx]
        self._scores[env_idx] = []
        self._best[env_idx] = None
        if best is None or len(scores) < 2:
            return                                  # 비교 대상이 없으면 신호도 없다

        mean = float(np.mean(scores))
        gap = best[0] - mean
        self.groups_closed += 1
        self.spread_sum += float(np.std(scores))
        self.gap_sum += gap
        if gap < self.min_gap:
            return

        self.groups_accepted += 1
        for obs, action in best[1]:
            self.obs.append(obs)
            self.actions.append(action)
        overflow = len(self.obs) - self.capacity
        if overflow > 0:                            # 오래된 것부터 버린다
            del self.obs[:overflow]
            del self.actions[:overflow]

    # ── 사용 ─────────────────────────────────────────────────────────────
    def sample(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.random.randint(0, len(self.obs), size=size)
        return (np.stack([self.obs[i] for i in idx]),
                np.array([self.actions[i] for i in idx], dtype=np.int64))

    def stats(self) -> dict[str, float]:
        closed = max(self.groups_closed, 1)
        return {
            "elite/size": float(len(self.obs)),
            "elite/groups_closed": float(self.groups_closed),
            "elite/accept_rate": self.groups_accepted / closed,
            "elite/group_spread": self.spread_sum / closed,
            "elite/group_gap": self.gap_sum / closed,
        }

    def __len__(self) -> int:
        return len(self.obs)


class GroupEliteDQN(DQN):
    """ε-greedy 를 마스킹하고, 같은 판 그룹의 최고 롤아웃을 margin 손실로 따라 한다."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elite: GroupEliteBuffer | None = None    # n_envs 를 알아야 만들 수 있다

    def _excluded_save_params(self) -> list[str]:
        """학습 전용 상태는 체크포인트에 넣지 않는다.

        elite 버퍼에는 관측 6만 개(8.4KB씩)가 들어 있어 그대로 저장하면 zip 이
        500MB 를 넘는다. 추론에는 전혀 쓰이지 않는다.
        (QRDQN/V10c 는 이걸 안 해서 로드 자체가 불가능한 체크포인트가 나왔다.)
        """
        return super()._excluded_save_params() + ["elite"]

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

    # ── 에피소드 수집 ────────────────────────────────────────────────────
    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        if self.elite is None:
            self.elite = GroupEliteBuffer(self.n_envs)

        last_obs = self._last_obs
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

        for i in range(self.n_envs):
            self.elite.record_step(i, last_obs[i].copy(), int(buffer_action[i]))
            if dones[i]:
                # 종료 스텝의 info 는 아직 reset 전이라 그 에피소드의 group_index 를 담고 있다
                self.elite.end_episode(
                    i,
                    float(infos[i].get("score", 0.0)),
                    int(infos[i].get("group_index", -1)),
                )

    # ── 학습 (TD 손실 + margin 모방 손실을 한 번의 backward 로) ───────────
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        use_elite = self.elite is not None and len(self.elite) >= ELITE_WARMUP
        td_losses, margin_losses = [], []

        for _ in range(gradient_steps):
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
            if use_elite:
                margin_loss = self._margin_loss()
                margin_losses.append(margin_loss.item())
                loss = loss + ELITE_LOSS_WEIGHT * margin_loss

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(td_losses))
        if margin_losses:
            self.logger.record("elite/margin_loss", float(np.mean(margin_losses)))
        if self.elite is not None:
            for key, value in self.elite.stats().items():
                self.logger.record(key, value)

    def _margin_loss(self) -> torch.Tensor:
        """DQfD large-margin. 교사 행동이 λ 만큼 앞서면 정확히 0 이 된다.

        불법 수는 신경망 안에서 이미 -1e8 이라 max 후보로 올라오지 못한다.
        """
        assert self.elite is not None
        obs, actions = self.elite.sample(min(ELITE_BATCH, len(self.elite)))
        expert = torch.as_tensor(actions, device=self.device, dtype=torch.long)

        q = self.q_net(obs_as_tensor(obs, self.device))                  # (B, A)
        margin = torch.full_like(q, ELITE_MARGIN)
        margin.scatter_(1, expert[:, None], 0.0)                         # 교사 행동만 0

        best = (q + margin).max(dim=1).values
        chosen = q.gather(1, expert[:, None]).squeeze(1)
        return (best - chosen).mean()


def _env():
    def _init():
        factory = getattr(env_mod, "make_train_env", env_mod.make_env)
        return factory(ROWS, COLS, render_mode=None)   # 마스킹은 신경망 안에서 한다
    return _init


def generate_env(config: TrainConfig):
    return VecMonitor(SubprocVecEnv([_env() for _ in range(config.n_envs)]))


def train_model(env, config: TrainConfig):
    # 하이퍼파라미터는 V1.0(114.48점) 과 완전히 동일하게 둔다. 바꾼 것은
    # 신경망(듀얼링)과 손실(margin 모방), 그리고 학습용 판 공급 방식(그룹)뿐이다.
    model = GroupEliteDQN(
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
        exploration_fraction=0.3,
        # 0.05 를 유지하는 데는 이유가 있다. 그룹 안에서 롤아웃이 서로 달라야
        # 'best-of-K' 라는 신호가 생긴다. ε 를 더 낮추면 K개가 전부 같은 판이 되어
        # elite/group_spread 가 0 으로 죽고 모방이 아무 일도 하지 않게 된다.
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
