import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import gymnasium as gym
import ai.envs

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.wrappers.action_mask import wrap_with_mask

AI_VERSION = 3.3
TOTLAL_TIMESTEP = 10_000_000
CHECKPOINT_TIMESTEP = 100_000

N_ENVS = max(1, (os.cpu_count() or 0) - 1)

def _env():
    def _init():
        env = gym.make(
            "envs/AppleGame-v0",
            render_mode=None,
            rows=9,
            cols=18,
        )
        env = wrap_with_mask(env)
        return env
    return _init

def generate_env():
    env = SubprocVecEnv([_env() for _ in range(N_ENVS)])
    env = VecMonitor(env)
    return env

def linear_schedule(initial_value: float):
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func

def train_model(env):
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_TIMESTEP // N_ENVS, 1),
        save_path="ai\\checkpoints",
        name_prefix=f"MaskablePPO_V{AI_VERSION}",
    )

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(3e-4),
        device="cuda",          # GPU 사용. cuda 안 잡히면 "cpu"로 자동 대체할지 확인 필요
        verbose=1,
        ent_coef=0.01,
        n_steps=512,             # env당 스텝 수는 줄이고 (기존 2048 → 512)
        batch_size=1024,         # n_steps * N_ENVS 가 실제 rollout 크기이므로 그에 맞춰 조정
        policy_kwargs=policy_kwargs,
        tensorboard_log="ai\\logs",
    )  # gamma 값도 조정해보자!

    model.learn(
        total_timesteps=TOTLAL_TIMESTEP,
        tb_log_name=f"MaskablePPO_V{AI_VERSION}_{TOTLAL_TIMESTEP}",
        callback=checkpoint_callback,
    )

    model.save(f"ai\\MaskablePPO_V{AI_VERSION}_{TOTLAL_TIMESTEP}.zip")
    print("AI 모델 저장 완료!")

    return model

def evaluate_model(model):
    eval_env = gym.make("envs/AppleGame-v0", render_mode="ansi", rows=9, cols=18)
    eval_env = wrap_with_mask(eval_env)

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"평균 reward: {mean_reward:.2f} +- {std_reward:.2f}")

if __name__ == "__main__":
    print(f"{N_ENVS}개 코어 병렬 처리 환경 구축")
    env = generate_env()
    model = train_model(env)
    evaluate_model(model)
