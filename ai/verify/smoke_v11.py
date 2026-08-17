"""V11 학습 -> 저장 -> measure.py 경로 로드 -> 1판 플레이 까지 짧게 돌려 본다.

    python ai/verify/smoke_v11.py --target v11a     (CPU, 1~2분)
    python ai/verify/smoke_v11.py --target v11b
    python ai/verify/smoke_v11.py --target v11c

버전마다 env/model 을 같은 이름으로 import 하므로 **--target 하나씩 따로 실행**한다.

학습 상수(learning_starts, ELITE_WARMUP 등)는 짧은 실행에서도 핵심 코드가 한 번은
돌도록 작게 덮어쓴다. 목적은 성능이 아니라 **모든 분기가 실제로 실행되는지**다.
train.py 의 기본값(learning_starts=20000 등)으로 몇천 스텝만 돌리면 학습 루프가
한 번도 안 돌아서 아무것도 검증되지 않는다 — 이전 세션이 겪은 함정이다.

체크포인트는 버전 폴더의 models/ 에 잠깐 만들었다가 반드시 지운다.
"""
import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # ai/verify/ -> 저장소 루트
ROWS, COLS = 9, 18
STEPS = 1600


def load_version(rel):
    version_dir = ROOT / rel
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "runs"))
    sys.path.insert(0, str(version_dir))
    import env as env_mod
    import model as model_mod
    return env_mod, model_mod, version_dir


def build_vec(env_mod, maskable, n_envs=2):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    factory = getattr(env_mod, "make_train_env", env_mod.make_env)

    def _init():
        e = factory(ROWS, COLS, render_mode=None)
        if maskable:
            from ai.wrappers.action_mask import wrap_with_mask
            e = wrap_with_mask(e)
        return e

    return VecMonitor(DummyVecEnv([_init for _ in range(n_envs)]))


def save_and_reload(model, version_dir, algorithm, version, steps):
    from ai.training import model_filename
    from agents.ai.model_loader import load_model

    models_dir = version_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{model_filename(algorithm, version, steps)}.zip"
    model.save(str(path))
    size_mb = path.stat().st_size / 1e6
    print(f"  저장: {path.name}  ({size_mb:.1f} MB)")

    try:
        loaded, env, info, search = load_model(path, (ROWS, COLS), device="cpu")
        print(f"  로드 성공: policy={type(loaded.policy).__name__}, 탐색={search.describe()}")

        obs, _ = env.reset(options={"board_source": 1234})
        steps_done, score = 0, 0
        while steps_done < ROWS * COLS:
            masks = env.unwrapped.get_action_mask()
            if not masks.any():
                break
            if info.use_action_masking:
                action, _ = loaded.predict(obs, action_masks=masks, deterministic=True)
            else:
                action, _ = loaded.predict(obs, deterministic=True)
            action = int(action)
            assert masks[action], f"불법 수 선택: {action}"
            obs, _, term, trunc, step_info = env.step(action)
            steps_done += 1
            score = step_info.get("score", score)
            if term or trunc:
                break
        print(f"  1판 플레이: {steps_done}수, {score}점  (합법 수만 선택함)")
        return size_mb
    finally:
        path.unlink(missing_ok=True)
        try:
            models_dir.rmdir()
        except OSError:
            pass


def smoke_v11a():
    env_mod, model_mod, version_dir = load_version("ai/models/DQN/V11a")
    import train as train_mod
    train_mod.ELITE_WARMUP = 10          # 짧은 실행에서도 margin 손실이 돌게

    venv = build_vec(env_mod, maskable=False)
    model = train_mod.GroupEliteDQN(
        model_mod.POLICY_CLASS, venv,
        learning_rate=1e-4, buffer_size=4000, learning_starts=200, batch_size=64,
        tau=1.0, gamma=0.997, train_freq=4, gradient_steps=1,
        target_update_interval=500, exploration_fraction=0.3,
        exploration_initial_eps=1.0, exploration_final_eps=0.05, max_grad_norm=10.0,
        device="cpu", verbose=0,
        policy_kwargs=model_mod.make_policy_kwargs(ROWS, COLS))
    model.learn(total_timesteps=STEPS)

    elite = model.elite
    print(f"  그룹 마감 {elite.groups_closed}개, 채택 {elite.groups_accepted}개, "
          f"엘리트 전이 {len(elite)}개")
    assert elite.groups_closed > 0, "그룹이 한 번도 마감되지 않았다 (스텝을 늘릴 것)"
    assert len(elite) > 0, "엘리트가 하나도 채택되지 않았다"
    stats = elite.stats()
    print(f"  group_spread={stats['elite/group_spread']:.2f}, "
          f"group_gap={stats['elite/group_gap']:.2f}, "
          f"accept_rate={stats['elite/accept_rate']:.2f}")
    assert stats["elite/group_spread"] > 0, "그룹 안 점수가 전부 같다 (신호 없음)"

    save_and_reload(model, version_dir, "DQN", "11a", STEPS)


def smoke_v11b():
    env_mod, model_mod, version_dir = load_version("ai/models/MaskablePPO/V11b")
    import train as train_mod
    from sb3_contrib import MaskablePPO

    venv = build_vec(env_mod, maskable=True)
    model = MaskablePPO(
        model_mod.POLICY_CLASS, venv,
        learning_rate=3e-4, n_steps=256, batch_size=128, n_epochs=2,
        target_kl=None, gamma=1.0, gae_lambda=1.0, ent_coef=0.02, vf_coef=0.5,
        max_grad_norm=0.5, device="cpu", verbose=0,
        policy_kwargs=model_mod.make_policy_kwargs(ROWS, COLS))

    # logger.name_to_value 는 PPO 가 매 이터레이션 dump 할 때 비워지므로,
    # 콜백이 실제로 무엇을 마감했는지는 직접 잡아 둔다.
    class Probe(train_mod.GroupStats):
        def __init__(self):
            super().__init__()
            self.closed: list[list[float]] = []

        def _close(self, env_idx, group_index):
            scores = list(self._scores.get((env_idx, group_index), []))
            if len(scores) >= 2:
                self.closed.append(scores)
            super()._close(env_idx, group_index)

    stats = Probe()
    model.learn(total_timesteps=STEPS, callback=[stats])

    import numpy as np
    print(f"  마감된 그룹 {len(stats.closed)}개, 예: {stats.closed[:2]}")
    assert stats.closed, "GroupStats 가 그룹을 한 번도 마감하지 못했다"
    spreads = [float(np.std(s)) for s in stats.closed]
    print(f"  group/spread 평균 {np.mean(spreads):.2f}, "
          f"group/gap 평균 {np.mean([max(s) - np.mean(s) for s in stats.closed]):.2f}")
    assert max(spreads) > 0, "그룹 안 점수가 전부 같다 (신호 없음)"

    buf = model.rollout_buffer
    nonzero = int((buf.rewards != 0).sum())
    print(f"  롤아웃 버퍼: 0 이 아닌 보상 {nonzero}개 / {buf.rewards.size}칸 "
          f"(종료 스텝에서만 나와야 정상)")
    assert 0 < nonzero < buf.rewards.size * 0.2, "보상이 종료 스텝에만 있지 않다"
    assert np.isfinite(buf.advantages).all(), "이점에 NaN/inf 가 있다"

    save_and_reload(model, version_dir, "MaskablePPO", "11b", STEPS)


def smoke_v11c():
    env_mod, model_mod, version_dir = load_version("ai/models/DQN/V11c")
    import train as train_mod
    train_mod.AUX_BATCH = 32             # CPU 스모크라 작게

    venv = build_vec(env_mod, maskable=False)
    model = train_mod.OutcomePredictionDQN(
        model_mod.POLICY_CLASS, venv,
        learning_rate=1e-4, buffer_size=4000,
        replay_buffer_class=train_mod.OutcomeReplayBuffer,
        learning_starts=200, batch_size=64,
        tau=1.0, gamma=0.997, train_freq=4, gradient_steps=1,
        target_update_interval=500, exploration_fraction=0.3,
        exploration_initial_eps=1.0, exploration_final_eps=0.05, max_grad_norm=10.0,
        device="cpu", verbose=0,
        policy_kwargs=model_mod.make_policy_kwargs(ROWS, COLS))
    model.learn(total_timesteps=STEPS)

    # DQN 은 에피소드마다 로그를 dump 하면서 name_to_value 를 비운다.
    # 값을 확실히 보려면 dump 없이 train() 을 한 번 더 부른다 (보조 경로도 다시 탄다).
    model.train(gradient_steps=3, batch_size=64)
    logged = model.logger.name_to_value
    print(f"  train/loss={logged.get('train/loss'):.3e}  "
          f"aux/look_loss={logged.get('aux/look_loss')}  "
          f"aux/look_corr={logged.get('aux/look_corr')}")
    print(f"  aux/survive_loss={logged.get('aux/survive_loss')}  "
          f"aux/survive_acc={logged.get('aux/survive_acc')}  "
          f"aux/look_weight={logged.get('aux/look_weight')}")
    assert logged.get("aux/look_loss") is not None, "보조 손실이 한 번도 돌지 않았다"
    assert logged.get("aux/survive_loss") is not None

    valid = int(model.replay_buffer.outcome_valid.sum())
    print(f"  결말 라벨이 붙은 전이 {valid}개")
    assert valid > 0, "잔존 라벨이 하나도 채워지지 않았다"

    save_and_reload(model, version_dir, "DQN", "11c", STEPS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=("v11a", "v11b", "v11c"))
    args = parser.parse_args()

    print(f"== 스모크: {args.target} ({STEPS} 스텝, CPU) ==")
    try:
        {"v11a": smoke_v11a, "v11b": smoke_v11b, "v11c": smoke_v11c}[args.target]()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    print("통과")
