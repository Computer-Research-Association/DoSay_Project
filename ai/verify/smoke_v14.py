"""V14 학습 -> 저장 -> measure.py 경로 로드 -> 1판 완주까지 짧게 돌려 본다.

    python ai/verify/smoke_v14.py

CPU 에서 빔은 느리므로 `BEAM_WIDTH` 를 작게 덮어쓴다. 목적은 성능이 아니라
**모든 코드 경로가 실제로 실행되는지**다 — 빔이 계획을 세우고, 그 수순이 교사로
버퍼에 들어가고, margin 손실과 가치 손실이 돌고, 저장/로드가 되는지.
"""

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS, COLS = 9, 18
STEPS = 500
SMOKE_WIDTH = 4


def main() -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "runs"))
    sys.path.insert(0, str(ROOT / "ai/models/DQN/V14"))
    import env as env_mod
    import model as model_mod
    import train as train_mod
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from ai.training import model_filename
    from agents.ai.model_loader import load_model

    model_mod.BEAM_WIDTH = SMOKE_WIDTH
    print(f"BEAM_WIDTH = {model_mod.BEAM_WIDTH} (스모크 전용), MODE = {model_mod.MODE}")

    venv = VecMonitor(DummyVecEnv([lambda: env_mod.make_env(ROWS, COLS) for _ in range(2)]))
    model = train_mod.BeamTeacherDQN(
        model_mod.POLICY_CLASS, venv,
        learning_rate=2e-4, buffer_size=4000,
        replay_buffer_class=train_mod.BeamTeacherBuffer,
        learning_starts=100, batch_size=32,
        tau=1.0, gamma=1.0, target_update_interval=10 ** 9,
        exploration_initial_eps=0.0, exploration_final_eps=0.0, exploration_fraction=1.0,
        train_freq=4, gradient_steps=2,
        max_grad_norm=10.0, device="cpu", verbose=0,
        policy_kwargs=model_mod.make_policy_kwargs(ROWS, COLS))

    train_mod.warm_start(model, train_mod.INIT_FROM)   # 없으면 안내만 하고 넘어간다

    start = time.time()
    model.learn(total_timesteps=STEPS)
    print(f"  학습 {STEPS} 스텝: {time.time() - start:.0f}s (CPU)")

    model.train(gradient_steps=3, batch_size=32)   # dump 없이 지표 확인
    logged = model.logger.name_to_value
    print("  " + "  ".join(f"{k.split('/')[1]}={logged[k]:.3f}" for k in (
        "train/margin_loss", "train/teacher_agreement",
        "train/value_loss", "train/value_mae") if k in logged))
    assert logged.get("train/margin_loss") is not None, "학습 루프가 한 번도 안 돌았다"

    buf = model.replay_buffer
    usable = int((buf.labelled & buf.teacher).sum())
    print(f"  교사 전이 {usable}개 / 라벨 {int(buf.labelled.sum())}개")
    assert usable > 0, "빔이 고른 수가 버퍼에 하나도 안 들어갔다"

    # 빔이 한 판에 한 번만 도는지 (캐시가 적중하는지)
    cache = len(model.policy.q_net._plan_cache)
    print(f"  계획 캐시 {cache}개 (판당 빔 1회면 스텝 수 근처여야 한다)")
    assert cache >= STEPS * 0.5, "계획이 자꾸 무효화되고 있다 (빔이 매 수 다시 돈다)"

    version_dir = ROOT / "ai/models/DQN/V14"
    models_dir = version_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{model_filename('DQN', '14', STEPS)}.zip"
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
            action = int(loaded.predict(obs, deterministic=True)[0])
            assert masks[action], f"불법 수 선택: {action}"
            obs, _, terminated, truncated, step_info = env.step(action)
            steps += 1
            score = step_info.get("score", score)
            if terminated or truncated:
                break
        elapsed = time.time() - start
        print(f"  1판 플레이: {steps}수, {score}점, {elapsed:.1f}s "
              f"({elapsed / max(steps, 1) * 1000:.0f} ms/수, CPU, W={SMOKE_WIDTH})")
    finally:
        path.unlink(missing_ok=True)
        try:
            models_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    print(f"== 스모크: V14 ({STEPS} 스텝, CPU) ==")
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    print("통과")
