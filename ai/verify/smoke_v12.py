"""V12 학습 -> 저장 -> measure.py 경로 로드 -> 1판 완주까지 짧게 돌려 본다.

    python ai/verify/smoke_v12.py --target v12a     (CPU, 수 분)
    python ai/verify/smoke_v12.py --target v12b

버전마다 env/model 을 같은 이름으로 import 하므로 **--target 하나씩 따로**.
학습 상수는 짧은 실행에서도 모든 분기가 돌도록 작게 덮어쓴다 — 목적은 성능이
아니라 **코드 경로가 실제로 실행되는지**다.

애프터스테이트 평가는 CPU 에서 느리다 (한 수마다 후보 수만큼 트렁크를 돈다).
여기서 재는 판당 시간을 그대로 믿지 말고, GPU 배치 기준으로는 훨씬 빠르다는
점만 염두에 둘 것. 다만 **CPU 판당 시간이 60초를 넘으면** 뭔가 잘못된 것이다.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS, COLS = 9, 18
STEPS = 1200


def load_version(rel: str):
    version_dir = ROOT / rel
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "runs"))
    sys.path.insert(0, str(version_dir))
    import env as env_mod
    import model as model_mod
    return env_mod, model_mod, version_dir


def save_and_reload(model, version_dir, version):
    from ai.training import model_filename
    from agents.ai.model_loader import load_model

    models_dir = version_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{model_filename('DQN', version, STEPS)}.zip"
    model.save(str(path))
    print(f"  저장: {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")

    try:
        loaded, env, info, search = load_model(path, (ROWS, COLS), device="cpu")
        print(f"  로드 성공: policy={type(loaded.policy).__name__}, 탐색={search.describe()}")

        start = time.time()
        obs, _ = env.reset(options={"board_source": 1234})
        steps = score = 0
        while steps < ROWS * COLS:
            masks = env.unwrapped.get_action_mask()
            if not masks.any():
                break
            action, _ = loaded.predict(obs, deterministic=True)
            action = int(action)
            assert masks[action], f"불법 수 선택: {action}"
            obs, _, terminated, truncated, step_info = env.step(action)
            steps += 1
            score = step_info.get("score", score)
            if terminated or truncated:
                break
        elapsed = time.time() - start
        print(f"  1판 플레이: {steps}수, {score}점, {elapsed:.1f}s (CPU)")
        assert elapsed < 60, "CPU 에서도 판당 60초를 넘으면 안 된다"
    finally:
        path.unlink(missing_ok=True)
        try:
            models_dir.rmdir()
        except OSError:
            pass


def smoke(target: str):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    rel = f"ai/models/DQN/V{target[1:]}"
    env_mod, model_mod, version_dir = load_version(rel)
    import train as train_mod

    def _init():
        return env_mod.make_env(ROWS, COLS, render_mode=None)

    venv = VecMonitor(DummyVecEnv([_init for _ in range(2)]))
    model = train_mod.AfterstateOutcomeDQN(
        model_mod.POLICY_CLASS, venv,
        learning_rate=2e-4, buffer_size=4000,
        replay_buffer_class=train_mod.OutcomeReplayBuffer,
        learning_starts=200, batch_size=32,
        tau=1.0, gamma=1.0, target_update_interval=10 ** 9,
        train_freq=4, gradient_steps=1,
        exploration_fraction=0.3, exploration_initial_eps=1.0, exploration_final_eps=0.05,
        max_grad_norm=10.0, device="cpu", verbose=0,
        policy_kwargs=model_mod.make_policy_kwargs(ROWS, COLS))

    start = time.time()
    model.learn(total_timesteps=STEPS)
    print(f"  학습 {STEPS} 스텝: {time.time() - start:.0f}s (CPU, {model.logger.name_to_value.get('time/fps', '?')} fps)")

    # DQN 은 에피소드마다 dump 하면서 name_to_value 를 비운다. 확실히 보려면 한 번 더.
    model.train(gradient_steps=3, batch_size=32)
    logged = model.logger.name_to_value
    keys = ["train/survive_loss", "train/leftover_loss", "train/survive_acc",
            "train/leftover_mae", "train/leftover_bias"]
    print("  " + "  ".join(f"{k.split('/')[1]}={logged[k]:.3f}" for k in keys if k in logged))
    assert logged.get("train/leftover_mae") is not None, "학습 루프가 한 번도 안 돌았다"

    valid = int(model.replay_buffer.outcome_valid.sum())
    print(f"  결말 라벨이 붙은 전이 {valid}개")
    assert valid > 0, "잔존 라벨이 하나도 안 채워졌다"

    save_and_reload(model, version_dir, target[1:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=("v12a", "v12b"))
    args = parser.parse_args()
    print(f"== 스모크: {args.target} ({STEPS} 스텝, CPU) ==")
    try:
        smoke(args.target)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    print("통과")
