import os, sys
from pathlib import Path

VERSION_DIR = Path(__file__).resolve().parent
ROOT = VERSION_DIR.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VERSION_DIR))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.wrappers.action_mask import wrap_with_mask
from env import make_env                          # 같은 폴더
from model import POLICY_CLASS, make_policy_kwargs  # 같은 폴더

AI_VERSION = "5.0"
TOTAL_TIMESTEP = 10_000_000
CHECKPOINT_TIMESTEP = 100_000
ROWS, COLS = 9, 18

N_ENVS = max(1, (os.cpu_count() or 0) - 1)


def _env():
    def _init():
        return wrap_with_mask(make_env(ROWS, COLS, render_mode=None))
    return _init


def generate_env():
    return VecMonitor(SubprocVecEnv([_env() for _ in range(N_ENVS)]))


def linear_schedule(initial_value: float):
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func


class ScoreCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info and "score" in info:
                self.logger.record_mean("rollout/ep_score_mean", info["score"])
        return True


def train_model(env):
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_TIMESTEP // N_ENVS, 1),
        save_path=str(VERSION_DIR / "checkpoints"),
        name_prefix=f"MaskablePPO_V{AI_VERSION}",
    )

    model = MaskablePPO(
        POLICY_CLASS,
        env,
        learning_rate=linear_schedule(3e-4),
        n_steps=512,
        batch_size=1024,
        n_epochs=4,        # 핵심 수정 1: 기본값 10 -> 4. rollout 과잉 재사용 방지
        target_kl=0.03,    # 핵심 수정 2: KL이 튀면 해당 업데이트 조기 중단 (안전장치)
        gamma=0.998,       # 핵심 수정 3: 40수 에피소드에서 후반 보상 할인 완화
        ent_coef=0.01,
        device="cuda",
        verbose=1,
        policy_kwargs=make_policy_kwargs(ROWS, COLS),
        tensorboard_log=str(ROOT / "ai" / "logs"),
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEP,
        tb_log_name=f"MaskablePPO_V{AI_VERSION}_{TOTAL_TIMESTEP}",
        callback=[checkpoint_callback, ScoreCallback()],
    )

    # 버전 폴더 안에 저장 -> measure.py 가 이 폴더를 선택 가능해진다
    model.save(str(VERSION_DIR / f"MaskablePPO_V{AI_VERSION}_{TOTAL_TIMESTEP}.zip"))
    print("AI 모델 저장 완료!")
    return model


def evaluate_model(model):
    eval_env = wrap_with_mask(make_env(ROWS, COLS, render_mode="ansi"))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"평균 reward: {mean_reward:.2f} +- {std_reward:.2f}")


if __name__ == "__main__":
    print(f"{N_ENVS}개 코어 병렬 처리 환경 구축")
    env = generate_env()
    model = train_model(env)
    evaluate_model(model)
