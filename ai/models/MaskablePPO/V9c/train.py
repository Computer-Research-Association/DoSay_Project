import sys
from pathlib import Path

VERSION_DIR = Path(__file__).resolve().parent
ROOT = VERSION_DIR.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VERSION_DIR))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.training import (
    CheckpointSaver, ScoreCallback, TrainConfig, describe_device, model_filename,
    parse_train_args, save_model,
)
from ai.wrappers.action_mask import wrap_with_mask
import env as env_mod                             # 같은 폴더
from model import POLICY_CLASS, make_policy_kwargs  # 같은 폴더

ALGORITHM = "MaskablePPO"
AI_VERSION = "9c"
ROWS, COLS = 9, 18

ENT_COEF_START = 0.01
# V6.0 은 0.002 로 끝냈고 엔트로피가 0.79(유효 2.2수)까지 무너졌다. 그런데 V7.0 에서
# 정책의 역할은 '두는 것'이 아니라 탐색 후보를 **순위 매기는 것**이다. 너무 무너지면
# 상위 몇 수 밖의 순서가 의미를 잃어 2수째 가지치기(top-16)가 망가진다. 그래서 더 낮추지 않는다.
ENT_COEF_END = 0.002


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


class EntropyDecay(BaseCallback):
    """ent_coef 를 학습 진행에 따라 선형으로 줄인다."""

    def __init__(self, start: float, end: float) -> None:
        super().__init__()
        self.start, self.end = start, end

    def _on_step(self) -> bool:
        remaining = self.model._current_progress_remaining  # 1.0 -> 0.0
        self.model.ent_coef = self.end + (self.start - self.end) * remaining  # type: ignore[attr-defined]
        return True


def train_model(env, config: TrainConfig):
    model = MaskablePPO(
        POLICY_CLASS,
        env,
        learning_rate=linear_schedule(3e-4),
        n_steps=512,
        batch_size=512,
        n_epochs=4,
        # V6.0 은 target_kl=0.03 이었다. 행동이 7533개라 approx_kl 이 쉽게 튀는데,
        # SB3 는 이때 남은 에포크를 통째로 버린다. 업데이트가 만성적으로 잘려
        # 학습이 기어갔을 가능성이 크다. 클리핑이 이미 신뢰영역 역할을 하므로 풀어 준다.
        target_kl=None,
        gamma=1.0,          # 에피소드가 81수 안에 반드시 끝난다. 할인하면 총점이 아닌
                            # '빨리 먹기'를 최적화하게 되므로 목표와 어긋난다.
        gae_lambda=0.95,
        ent_coef=ENT_COEF_START,
        vf_coef=1.0,        # 탐색이 V 에 의존한다. 가치 정확도에 가중치를 더 준다.
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
            EntropyDecay(ENT_COEF_START, ENT_COEF_END),
        ],
    )

    # 체크포인트와 같은 models/ 폴더에 저장 -> measure.py 가 여기서 골라 쓴다
    path = save_model(model, config, ALGORITHM, AI_VERSION, config.total_timestep)
    print(f"AI 모델 저장 완료! -> {path}")
    return model


def evaluate_model(model):
    # 셰이핑 때문에 reward 는 더 이상 점수/162 가 아니다. 점수는 measure.py 로 잰다.
    eval_env = wrap_with_mask(env_mod.make_env(ROWS, COLS, render_mode="ansi"))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"평균 reward: {mean_reward:.2f} +- {std_reward:.2f}  (탐색 미적용, 셰이핑 포함)")


if __name__ == "__main__":
    config = parse_train_args(VERSION_DIR)
    print(describe_device(config.device))
    print(f"{config.n_envs}개 코어 병렬 처리 환경 구축")
    env = generate_env(config)
    model = train_model(env, config)
    evaluate_model(model)
