"""MaskablePPO V11b — 그룹 상대 정책경사 (GRPO 계열). **새 시도 ①**

한 줄 요약: **알고리즘 팀의 어닐링이 추론 시점에 하는 '재시도'를, 우리는 학습
시점에 흡수해서 추론은 1회 forward 로 끝낸다.**

──────────────────────────────────────────────────────────────────────────
왜 지금까지 안 됐는가
──────────────────────────────────────────────────────────────────────────
V4 -> V10 까지 알고리즘(PPO/DQN/QR-DQN), 보상 셰이핑, 커리큘럼, n-step,
자기모방, 탐색 교사를 다 바꿔 봤는데 전부 114점 언저리, **점/수 2.24** 라는
같은 정책으로 수렴했다. 문서 §3 의 진단은 "30수 지연 신용할당을 1-step TD 가
못 넘는다"였다.

그런데 벤치마크 JSON 5개를 같은 시드로 정렬해 이원배치 분산분해를 해보면
더 근본적인 것이 나온다.

    판(시드) 성분   172.9   (76.1%)
    모델 성분        28.6   (12.6%)
    상호작용/잔차    25.7   (11.3%)

**점수 분산의 76%가 어떤 판을 뽑았느냐다.** 알고리즘을 무엇으로 바꾸든 그
경사는 판 운이라는 큰 잡음 위에 얹혀 있었다. 자기모방(V10b)이 실패한 이유도
같다 — 어떤 모델이든 자기 상위 20% 에피소드는 **다른 모델에게도** +14~16점
높은 판이었다. 즉 걸러낸 것이 '잘 둔 판'이 아니라 '쉬운 판'이었다.

──────────────────────────────────────────────────────────────────────────
그래서 무엇을 하는가
──────────────────────────────────────────────────────────────────────────
같은 판을 K(=8)번 둔다. 이점은 그 판 안에서의 상대값으로만 준다.

    A_i = (점수_i - 같은 판의 다른 롤아웃 평균) / 6.0

판 성분이 통계적으로 줄어드는 게 아니라 **정확히 상쇄**된다. 크리틱이 하는
베이스라인 역할과 다른 점이 여기다 — 크리틱의 베이스라인에는 함수근사 오차가
남고(V7 측정: 잔차 약 1.7점 = 형제 간 실제 가치 차이와 같은 크기),
그룹 평균에는 그런 오차가 없다.

그리고 이 방법이 오르는 경사의 크기를 직접 쟀다. 무작위 타이브레이크를 준
min-area 정책으로 12판 x 12회:

    판 안 std          5.41
    같은 판 best-of-12  평균 대비 **+8.62**

한심한 휴리스틱조차 재시도만으로 8.6점이 나온다. 그리고 이 방법이 하는 일이
정확히 "평범한 롤아웃을 best-of-K 쪽으로 끌어올리는 것"이다.

──────────────────────────────────────────────────────────────────────────
하이퍼파라미터에서 중요한 것 세 가지
──────────────────────────────────────────────────────────────────────────
1) **gae_lambda = 1.0.** λ<1 이면 크리틱으로 부트스트랩하는 항이 섞여 들어와
   문제의 원인이던 부트스트랩 사슬이 되살아난다. 1.0 이면 A_t = R - V(s_t) 인
   순수 몬테카를로 이점이라 사슬이 0홉이다.

2) **gamma = 1.0.** 에피소드가 반드시 81수 안에 끝나므로 할인이 필요 없고,
   할인하면 총점이 아니라 '빨리 먹기'를 최적화하게 된다 (V6.0 이후 유지).

3) **엔트로피를 V9c 보다 높게 유지한다 (0.02 -> 0.005).** 이 방법의 연료는
   **그룹 안 롤아웃의 다양성**이다. 정책이 결정적이 되면 K개 롤아웃이 전부
   같아지고 이점이 전부 0 이 되어 학습이 그 자리에서 멈춘다. V6.0 에서
   엔트로피가 -2.79 -> -0.79 로 붕괴한 전례가 있으므로 특히 조심해야 한다.
   -> `group/spread` 지표로 감시한다. **이 값이 0 으로 가면 학습이 죽은 것이다.**

──────────────────────────────────────────────────────────────────────────
로그 읽는 법 (이 버전은 지표 해석이 다르다)
──────────────────────────────────────────────────────────────────────────
    rollout/ep_rew_mean    0 근처를 맴돈다. **정의상 그렇다.** 보지 말 것.
    rollout/ep_score_mean  실제 점수. 이것이 학습 진행 지표다.
    group/spread           그룹 안 점수 std. 0 으로 죽으면 신호가 사라진 것.
    group/gap              그룹 최고 - 그룹 평균. 정책이 좋아지면 줄어드는 게 정상
                           (평균이 최고를 따라잡는 것이 이 방법의 목표다).
    train/explained_variance  크리틱이 '이 롤아웃이 그룹 평균 대비 어떤가'를
                           얼마나 맞히는가. 지금까지의 EV(판 난이도까지 포함해
                           0.98) 와 의미가 완전히 다르니 그 숫자와 비교하지 말 것.
"""

import sys
from collections import defaultdict
from pathlib import Path

VERSION_DIR = Path(__file__).resolve().parent
ROOT = VERSION_DIR.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VERSION_DIR))

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.training import (
    CheckpointSaver, ScoreCallback, TrainConfig, describe_device, model_filename,
    parse_train_args, save_model,
)
from ai.wrappers.action_mask import wrap_with_mask
import env as env_mod
from model import POLICY_CLASS, make_policy_kwargs  # 같은 폴더

ALGORITHM = "MaskablePPO"
AI_VERSION = "11b"
ROWS, COLS = 9, 18

# 그룹 안 다양성이 이 방법의 연료다. V9c(0.01 -> 0.002)보다 높게 유지한다.
ENT_COEF_START = 0.02
ENT_COEF_END = 0.005


class GroupStats(BaseCallback):
    """그룹 안 점수 분포를 기록한다. 이 버전에서 가장 중요한 진단 지표다.

    group/spread 가 0 으로 죽으면 K개 롤아웃이 전부 같아졌다는 뜻이고, 그러면
    이점이 전부 0 이라 정책 경사가 사라진다. 점수가 정체됐을 때 '수렴한 것'인지
    '신호가 죽은 것'인지를 이 지표 하나로 가른다.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scores: dict[tuple[int, int], list[float]] = defaultdict(list)
        self._env_group: dict[int, int] = {}

    def _close(self, env_idx: int, group_index: int) -> None:
        scores = self._scores.pop((env_idx, group_index), [])
        if len(scores) < 2:
            return
        arr = np.asarray(scores, dtype=np.float64)
        self.logger.record_mean("group/spread", float(arr.std()))
        self.logger.record_mean("group/gap", float(arr.max() - arr.mean()))
        self.logger.record_mean("group/best", float(arr.max()))
        self.logger.record_mean("group/mean", float(arr.mean()))

    def _on_step(self) -> bool:
        for env_idx, info in enumerate(self.locals["infos"]):
            if "episode" not in info or "score" not in info:
                continue
            group_index = int(info.get("group_index", -1))
            previous = self._env_group.get(env_idx)
            if previous is not None and previous != group_index:
                self._close(env_idx, previous)      # 그룹이 넘어갔으니 이전 그룹 마감
            self._env_group[env_idx] = group_index
            self._scores[(env_idx, group_index)].append(float(info["score"]))
            self.logger.record_mean("group/baseline", float(info.get("group_baseline", 0.0)))
        return True


class EntropyDecay(BaseCallback):
    """ent_coef 를 학습 진행에 따라 선형으로 줄인다."""

    def __init__(self, start: float, end: float) -> None:
        super().__init__()
        self.start, self.end = start, end

    def _on_step(self) -> bool:
        remaining = self.model._current_progress_remaining  # 1.0 -> 0.0
        self.model.ent_coef = self.end + (self.start - self.end) * remaining  # type: ignore[attr-defined]
        return True


def _env():
    def _init():
        factory = getattr(env_mod, "make_train_env", env_mod.make_env)
        return wrap_with_mask(factory(ROWS, COLS, render_mode=None))
    return _init


def generate_env(config: TrainConfig):
    return VecMonitor(SubprocVecEnv([_env() for _ in range(config.n_envs)]))


def linear_schedule(initial_value: float):
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func


def train_model(env, config: TrainConfig):
    model = MaskablePPO(
        POLICY_CLASS,
        env,
        learning_rate=linear_schedule(3e-4),
        # 에피소드가 47~55수라 512 면 환경당 약 10판, 즉 그룹(8판) 하나가 대체로
        # 한 롤아웃 안에 들어온다. 그룹이 롤아웃 경계를 걸치면 베이스라인이 한
        # 업데이트 전 정책의 것이 되는데, PPO 의 비율 클리핑이 감당하는 범위다.
        n_steps=512,
        batch_size=512,
        n_epochs=4,
        # V6.0 은 target_kl=0.03 이었고 approx_kl 이 상시로 컷오프를 넘어 업데이트가
        # 만성적으로 잘렸다. 클리핑이 이미 신뢰영역 역할을 하므로 풀어 둔다.
        target_kl=None,
        gamma=1.0,
        gae_lambda=1.0,     # 부트스트랩 0홉. 이 버전의 핵심이므로 절대 낮추지 말 것.
        ent_coef=ENT_COEF_START,
        # 정책과 가치가 트렁크를 공유하지 않으므로(share_features_extractor=False)
        # 가치 손실이 커도 정책 표현을 누르지 않는다.
        vf_coef=0.5,
        max_grad_norm=0.5,
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
            GroupStats(),
            EntropyDecay(ENT_COEF_START, ENT_COEF_END),
        ],
    )

    path = save_model(model, config, ALGORITHM, AI_VERSION, config.total_timestep)
    print(f"AI 모델 저장 완료! -> {path}")
    return model


def evaluate_model(model):
    # 학습 환경의 보상은 그룹 상대값이라 reward 로는 아무것도 알 수 없다.
    # 평가는 make_env()(원래 규칙)로, 점수로만 본다.
    eval_env = wrap_with_mask(env_mod.make_env(ROWS, COLS, render_mode=None))
    scores = []
    for seed in range(10):
        obs, info = eval_env.reset(options={"board_source": 9000 + seed})
        done = False
        while not done:
            masks = eval_env.unwrapped.get_action_mask()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
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
