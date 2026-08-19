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
from env import make_env                          # 같은 폴더
from model import POLICY_CLASS, make_policy_kwargs  # 같은 폴더

ALGORITHM = "MaskablePPO"
AI_VERSION = "6.0"
ROWS, COLS = 9, 18

ENT_COEF_START = 0.02   # 합법수가 수십 개라 초반에는 넓게 훑어야 한다
ENT_COEF_END = 0.002    # 후반에는 좁혀서 수렴시킨다


def _env():
    def _init():
        return wrap_with_mask(make_env(ROWS, COLS, render_mode=None))
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
        target_kl=0.03,
        gamma=1.0,          # 에피소드가 81수 안에 반드시 끝난다. 할인하면 총점이 아닌
                            # '빨리 먹기'를 최적화하게 되므로 목표와 어긋난다.
        gae_lambda=0.95,
        ent_coef=ENT_COEF_START,
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
            EntropyDecay(ENT_COEF_START, ENT_COEF_END),
        ],
    )

    # 체크포인트와 같은 models/ 폴더에 저장 -> measure.py 가 여기서 골라 쓴다
    path = save_model(model, config, ALGORITHM, AI_VERSION, config.total_timestep)
    print(f"AI 모델 저장 완료! -> {path}")
    return model


def evaluate_model(model):
    eval_env = wrap_with_mask(make_env(ROWS, COLS, render_mode="ansi"))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"평균 reward: {mean_reward:.2f} +- {std_reward:.2f}  (x162 = 점수)")


if __name__ == "__main__":
    config = parse_train_args(VERSION_DIR)
    print(describe_device(config.device))
    print(f"{config.n_envs}개 코어 병렬 처리 환경 구축")
    env = generate_env(config)
    model = train_model(env, config)
    evaluate_model(model)
