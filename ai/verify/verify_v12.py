"""V12 계열 검증. 학습 없이 몇 초면 끝난다.

    python ai/verify/verify_v12.py --target v12a
    python ai/verify/verify_v12.py --target v12b

버전마다 env.py/model.py 를 같은 이름으로 import 하므로 **--target 하나씩 따로**.

가장 중요한 것은 **GPU 관측 인코딩이 `env._get_obs()` 와 원소 단위로 같은가**다.
V12 는 후보 수마다 애프터스테이트를 GPU 에서 만들어 평가하는데, 그 관측이 학습
때 본 것과 조금이라도 다르면 신경망이 분포 밖 입력을 받는다. **아무 에러도 안 나고
점수만 조용히 깎이는** 종류의 버그라 자동 대조가 필수다.

(2026-08-10 에 실제로 비슷한 사고가 있었다 — 기준선 측정 스크립트가 `dtype=np.int8`
을 빼먹어 다른 판에서 재고 있었는데, 합법수 판정은 엔진과 일치해서 몇 시간 동안
못 알아챘다.)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FAILED: list[str] = []


def check(name, condition, detail=""):
    print(f"  [{'OK  ' if condition else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        FAILED.append(name)


def load_version(rel: str):
    version_dir = ROOT / rel
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "runs"))
    sys.path.insert(0, str(version_dir))
    import env as env_mod
    import model as model_mod
    return env_mod, model_mod


def walk(env, seeds, depth):
    """여러 판을 무작위로 진행시키며 만나는 국면들을 내놓는다."""
    import numpy as np
    for seed in seeds:
        env.reset(options={"board_source": seed})
        for _ in range(depth):
            legal = np.flatnonzero(env.get_action_mask())
            if legal.size == 0:
                break
            yield env
            env.step(int(np.random.choice(legal)))


def main(target: str) -> None:
    import numpy as np
    import torch
    np.random.seed(0)
    torch.manual_seed(0)

    rel = f"ai/models/DQN/V{target[1:]}"
    env_mod, model_mod = load_version(rel)
    from game.board import Board

    rows, cols = 9, 18
    env = env_mod.make_env(rows, cols)
    policy = model_mod.POLICY_CLASS(
        env.observation_space, env.action_space, lambda _: 1e-4,
        **model_mod.make_policy_kwargs(rows, cols))
    q_net = policy.q_net
    index = q_net.index

    def grid_of(e):
        return torch.as_tensor(e.board.grid, dtype=torch.float32)[None]

    print("\n== 합법 판정: GPU vs game.Board ==")
    mismatch = compared = 0
    for e in walk(env, range(400, 408), 8):
        engine = e.get_action_mask()
        mine = index.legal_mask_from_grid(grid_of(e)).numpy()[0]
        compared += 1
        mismatch += int(not np.array_equal(engine, mine))
    check(f"합법수 마스크가 완전 일치 ({compared}개 국면)", mismatch == 0, f"불일치 {mismatch}")

    print("\n== 관측 인코딩: GPU vs env._get_obs()  ← 가장 중요 ==")
    worst = 0.0
    compared = 0
    for e in walk(env, range(500, 508), 8):
        expected = e._get_obs()
        actual = index.observation(grid_of(e)).numpy()[0]
        worst = max(worst, float(np.abs(expected - actual).max()))
        compared += 1
    check(f"모든 채널이 일치 ({compared}개 국면)", worst < 1e-6, f"최대 오차 {worst:.2e}")

    print("\n== 애프터스테이트 생성: GPU vs Board.do_action ==")
    mismatch = compared = 0
    for e in walk(env, range(600, 604), 5):
        legal = np.flatnonzero(e.get_action_mask())[:16]
        after = index.erase(grid_of(e).expand(len(legal), rows, cols),
                            torch.as_tensor(legal, dtype=torch.long)).numpy()
        for slot, action_index in enumerate(legal):
            probe = Board.from_board(e.board.grid)
            probe.do_action(e.index_to_action[int(action_index)])
            compared += 1
            mismatch += int(not np.array_equal(probe.grid, after[slot].astype(np.int8)))
    check(f"지운 판이 완전 일치 ({compared}개 수)", mismatch == 0, f"불일치 {mismatch}")

    print("\n== 애프터스테이트 관측: GPU vs 엔진으로 만든 관측 ==")
    worst = compared = 0
    for e in walk(env, range(700, 703), 4):
        legal = np.flatnonzero(e.get_action_mask())[:12]
        saved = e.board.grid.copy()
        gpu = index.observation(index.erase(
            grid_of(e).expand(len(legal), rows, cols),
            torch.as_tensor(legal, dtype=torch.long))).numpy()
        for slot, action_index in enumerate(legal):
            child = Board.from_board(saved)
            child.do_action(e.index_to_action[int(action_index)])
            e.board = child
            worst = max(worst, float(np.abs(e._get_obs() - gpu[slot]).max()))
            compared += 1
        e.board = Board.from_board(saved)
    check(f"애프터스테이트 관측이 일치 ({compared}개)", worst < 1e-6, f"최대 오차 {worst:.2e}")

    print("\n== 행동 선택 (q_net.forward) ==")
    env.reset(options={"board_source": 77})
    obs = torch.as_tensor(env._get_obs()[None])
    with torch.no_grad():
        q = q_net(obs)
    mask = torch.as_tensor(env.get_action_mask())[None]
    check("Q 모양이 (1, 7533)", tuple(q.shape) == (1, 7533), str(tuple(q.shape)))
    check("불법 수는 -1e8", bool((q[~mask] <= -1e7).all()))
    check("합법 수는 유한", bool(torch.isfinite(q[mask]).all()))
    check("argmax 가 합법", bool(mask[0, int(q.argmax())]))

    # forward 는 '잔여가 가장 적은 애프터스테이트' 를 골라야 한다
    legal = np.flatnonzero(env.get_action_mask())
    with torch.no_grad():
        after = index.erase(torch.as_tensor(env.board.grid, dtype=torch.float32)[None]
                            .expand(len(legal), rows, cols),
                            torch.as_tensor(legal, dtype=torch.long))
        leftover = q_net.expected_leftover(index.observation(after))
    check("argmax Q == argmin 예상잔여",
          int(legal[int(leftover.argmin())]) == int(q.argmax()),
          f"{int(legal[int(leftover.argmin())])} vs {int(q.argmax())}")

    occupied = int(np.count_nonzero(env.board.grid))
    with torch.no_grad():
        base = float(q_net.expected_leftover(obs).item())
    expected_init = occupied * 0.5 if target == "v12a" else rows * cols * 0.5
    check("초기 가치가 sigmoid(0)=0.5 지점에서 출발",
          abs(base - expected_init) < 1e-3, f"{base:.2f} (기대 {expected_init:.2f})")

    print("\n== 배치 처리 ==")
    boards = []
    for e in walk(env, range(800, 805), 3):
        boards.append(e._get_obs())
    batch = torch.as_tensor(np.stack(boards))
    with torch.no_grad():
        q_batch = q_net(batch)
        q_one = torch.cat([q_net(batch[i:i + 1]) for i in range(len(boards))])
    check(f"배치 {len(boards)}개와 낱개 처리 결과가 같다",
          bool(torch.allclose(q_batch, q_one, atol=1e-5)),
          f"최대 차이 {float((q_batch - q_one).abs().max()):.2e}")

    print("\n== 환경 계약 ==")
    env.reset(options={"board_source": 4242})
    total, info = 0.0, None
    while True:
        legal = np.flatnonzero(env.get_action_mask())
        if legal.size == 0:
            break
        _, reward, terminated, truncated, info = env.step(int(np.random.choice(legal)))
        total += reward
        if terminated or truncated:
            break
    check("보상 합 = 점수/162 (셰이핑 없음)", abs(total - info["score"] / 162) < 1e-6)
    check("종료 시 final_grid 를 낸다", "final_grid" in info)
    check("final_grid 가 실제 최종 판", np.array_equal(info["final_grid"], env.board.grid))
    check("잔여 사과 = 162 - 점수",
          int(np.count_nonzero(info["final_grid"])) == 162 - info["score"])

    print("\n== OutcomeReplayBuffer ==")
    import train as train_mod
    from gymnasium import spaces
    buf = train_mod.OutcomeReplayBuffer(
        200, spaces.Box(0.0, 1.0, (13, rows, cols), dtype=np.float32),
        spaces.Discrete(7533), device="cpu", n_envs=1)
    dummy = np.zeros((1, 13, rows, cols), dtype=np.float32)
    act = np.zeros((1, 1), dtype=np.int64)
    rew = np.zeros(1, dtype=np.float32)
    final = np.zeros((rows, cols), dtype=np.int8)
    final[0, 0], final[3, 5], final[8, 17] = 7, 2, 9
    for step in range(6):
        done = np.array([step == 5])
        buf.add(dummy, dummy, act, rew, done, [{"final_grid": final}] if done[0] else [{}])
    check("에피소드 전체 슬롯에 라벨이 채워진다", bool(buf.outcome_valid[:6, 0].all()))
    check("라벨 = 최종 판에 사과가 남아 있는가",
          bool((buf.outcome[:6, 0] == (final.reshape(-1) != 0).astype(np.uint8)).all()))
    check("라벨 합 = 잔여 사과 수 (스칼라 목표의 근거)",
          int(buf.outcome[0, 0].sum()) == 3)
    buf.add(dummy, dummy, act, rew, np.array([False]), [{}])
    check("진행 중 전이는 아직 무효", not bool(buf.outcome_valid[6, 0]))
    sample = buf.sample_outcome(4)
    check("sample_outcome 모양", sample is not None
          and sample[0].shape == (4, 13, rows, cols) and sample[1].shape == (4, rows * cols))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=("v12a", "v12b"))
    args = parser.parse_args()
    print(f"== {args.target} 검증 ==")
    main(args.target)
    print()
    if FAILED:
        print(f"실패 {len(FAILED)}개: {FAILED}")
        raise SystemExit(1)
    print("전부 통과")
