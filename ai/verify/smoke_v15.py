"""V15 학습 -> 저장 -> measure.py 경로 로드 -> 1판 완주까지 짧게 돌려 본다.

    python ai/verify/smoke_v15.py

CPU 에서 빔은 느리므로 `BEAM_WIDTH` 를 작게 덮어쓴다. 목적은 성능이 아니라
**모든 코드 경로가 실제로 실행되는지**다. V15 에서 새로 생긴 것이 넷이라
그 넷을 각각 확인한다.

    1. `plan(collect=True)` 이 가치 라벨(죽은 빔 전부)과 정책 라벨(자식 가치 순위)을
       실제로 만들어 내는가
    2. 그 라벨이 링 버퍼에 들어가고 교차엔트로피 + Huber 가 도는가
    3. `POLICY_TOPK > 0` 일 때 빔이 정책으로 좁혀서도 합법 수를 내는가
    4. 라벨 목표 분포가 확률 (합 1, 음수 없음) 인가
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROWS, COLS = 9, 18
STEPS = 500
SMOKE_WIDTH = 4


def check_labels(model_mod, q_net, device) -> None:
    """plan(collect=True) 이 만드는 라벨 자체를 먼저 검증한다."""
    import torch
    from game.board import Board

    grid = torch.as_tensor(Board.from_seed((ROWS, COLS), 1234).grid,
                           dtype=torch.float32, device=device)
    occupied0 = int((grid != 0).sum())

    path, score, labels = q_net.plan(grid, SMOKE_WIDTH, collect=True)
    assert labels is not None, "collect=True 인데 라벨이 안 나왔다"

    v_grid, v_remain = labels["value_grids"], labels["value_remain"]
    p_act, p_prob = labels["policy_actions"], labels["policy_probs"]
    print(f"  라벨: 가치 {v_grid.shape[0]}개, 정책 {p_act.shape[0]}개 "
          f"(V14 는 가치 {len(path)}개뿐이었다)")
    assert v_grid.shape[0] > len(path), "가치 라벨이 최고 수순 하나에서만 나왔다"

    # 목표 분포가 확률인가
    s = p_prob.sum(dim=1)
    assert bool((p_prob >= 0).all()), "목표 분포에 음수가 있다"
    assert float((s - 1.0).abs().max()) < 1e-4, f"목표 분포 합이 1이 아니다 ({s.min()}~{s.max()})"

    # 가치 목표가 회계와 맞는가: 남은 점수 = 최종 점수 - 지금까지 얻은 점수 이므로
    # 0 <= remain <= 지금 남은 사과 수 여야 한다.
    alive = (v_grid != 0).flatten(1).sum(dim=1).float()
    assert float(v_remain.min()) >= -1e-4, f"가치 목표가 음수다 ({v_remain.min()})"
    assert bool((v_remain <= alive + 1e-4).all()), "가치 목표가 남은 사과 수를 넘는다"

    # 라벨에 담긴 행동이 실제로 합법인가
    grids = v_grid.to(torch.float32).to(device)
    legal = q_net.index.legal_mask_from_grid(
        p_grid_f := labels["policy_grids"].to(torch.float32).to(device))
    top1 = p_act[:, 0].to(torch.long).to(device)
    assert bool(legal.gather(1, top1[:, None]).all()), "목표 1등이 불법 수다"
    print(f"  목표 분포 합 1.0, 1등 전부 합법, 가치 목표 0~{float(v_remain.max()):.0f} "
          f"(빔 최고 {score}점, 초기 사과 {occupied0}개)")

    recall = labels["topk_recall"]
    print(f"  beam/topk_recall = {recall:.3f} "
          f"(무작위 초기화 정책이라 낮은 게 정상. 학습되면 오른다)")

    # POLICY_TOPK > 0 경로 (배포 모드) 도 태운다
    saved = model_mod.POLICY_TOPK
    try:
        model_mod.POLICY_TOPK = 6
        narrow_path, narrow_score, _ = q_net.plan(grid, SMOKE_WIDTH)
        assert narrow_path, "정책으로 좁힌 빔이 수순을 못 만들었다"
        board = Board.from_seed((ROWS, COLS), 1234)
        for _, action in narrow_path:
            act = q_net.index  # 합법 판정은 엔진으로
            valid = {(a.top_left, a.bottom_right) for a in board.get_valid_actions()}
            key = ((int(act.r_lo[action]), int(act.c_lo[action])),
                   (int(act.r_hi[action]) - 1, int(act.c_hi[action]) - 1))
            assert key in valid, f"좁힌 빔이 불법 수를 계획했다: {key}"
            board.do_action(next(a for a in board.get_valid_actions()
                                 if (a.top_left, a.bottom_right) == key))
        print(f"  POLICY_TOPK=6 빔: {len(narrow_path)}수 {narrow_score}점, "
              f"수순 전체가 엔진에서 합법")
    finally:
        model_mod.POLICY_TOPK = saved


def main() -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "runs"))
    sys.path.insert(0, str(ROOT / "ai/models/DQN/V15"))
    import env as env_mod
    import model as model_mod
    import train as train_mod
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from ai.training import model_filename
    from agents.ai.model_loader import load_model

    model_mod.BEAM_WIDTH = SMOKE_WIDTH
    print(f"BEAM_WIDTH = {model_mod.BEAM_WIDTH} (스모크 전용), "
          f"MODE = {model_mod.MODE}, POLICY_TOPK = {model_mod.POLICY_TOPK}")

    venv = VecMonitor(DummyVecEnv([lambda: env_mod.make_env(ROWS, COLS) for _ in range(2)]))
    model = train_mod.GuidedBeamDQN(
        model_mod.POLICY_CLASS, venv,
        learning_rate=2e-4, buffer_size=4000,
        learning_starts=100, batch_size=32,
        tau=1.0, gamma=1.0, target_update_interval=10 ** 9,
        exploration_initial_eps=0.0, exploration_final_eps=0.0, exploration_fraction=1.0,
        train_freq=4, gradient_steps=2,
        max_grad_norm=10.0, device="cpu", verbose=0,
        policy_kwargs=model_mod.make_policy_kwargs(ROWS, COLS))

    train_mod.warm_start(model, train_mod.INIT_FROM)   # 없으면 안내만 하고 넘어간다

    check_labels(model_mod, model.policy.q_net, "cpu")

    start = time.time()
    model.learn(total_timesteps=STEPS)
    print(f"  학습 {STEPS} 스텝: {time.time() - start:.0f}s (CPU)")

    model.train(gradient_steps=3, batch_size=32)   # dump 없이 지표 확인
    logged = model.logger.name_to_value
    keys = ("train/policy_loss", "train/teacher_agreement", "train/topk_recall_train",
            "train/value_loss", "train/value_mae")
    print("  " + "  ".join(f"{k.split('/')[1]}={logged[k]:.3f}" for k in keys if k in logged))
    assert logged.get("train/policy_loss") is not None, "학습 루프가 한 번도 안 돌았다"
    assert logged.get("train/value_mae") is not None

    print(f"  버퍼: 가치 {model.labels.v_size:,}행, 정책 {model.labels.p_size:,}행 "
          f"(V14 는 {STEPS}스텝에 교사 전이 {STEPS}개 이하였다)")
    assert model.labels.v_size > STEPS, "가치 라벨이 스텝 수보다 적다 (죽은 빔을 안 쓰고 있다)"
    assert model.labels.p_size > 0, "정책 라벨이 하나도 없다"

    # 빔이 한 판에 한 번만 도는지 (캐시가 적중하는지)
    cache = len(model.policy.q_net._plan_cache)
    print(f"  계획 캐시 {cache}개 (판당 빔 1회면 스텝 수 근처여야 한다)")
    assert cache >= STEPS * 0.5, "계획이 자꾸 무효화되고 있다 (빔이 매 수 다시 돈다)"
    assert not model.policy.q_net.pending_labels, "라벨이 안 비워졌다 (메모리 누수)"

    version_dir = ROOT / "ai/models/DQN/V15"
    models_dir = version_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{model_filename('DQN', '15', STEPS)}.zip"
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
    print(f"== 스모크: V15 ({STEPS} 스텝, CPU) ==")
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    print("통과")
