"""V13 검증. 학습 없이 몇 초면 끝난다.

    python ai/verify/verify_v13.py --target v13
    python ai/verify/verify_v13.py --target v13L

가장 중요한 것은 **GPU 롤아웃이 실제 게임과 같은 게임을 두는가**다. V13 은 매 수마다
후보 판을 GPU 에서 끝까지 두어 보고 그 점수로 수를 고른다. 롤아웃이 실제 규칙과
조금이라도 다르면 **아무 에러도 안 나고 점수만 조용히 깎인다.**

그래서 여기서는 GPU 롤아웃의 최종 점수를 `game.Board` 로 같은 정책을 돌린 결과와
**정확히 대조**한다. 관측 인코딩, 합법 판정, 판 지우기, 사과 세기도 전부 엔진과
원소 단위로 맞춰 본다.
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

    env_mod, model_mod = load_version(f"ai/models/DQN/{target.replace('v', 'V')}")
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

    print(f"\n== 게임 규칙 대조 (ROLLOUT_TOPK={model_mod.ROLLOUT_TOPK}) ==")
    mask_bad = obs_worst = apple_bad = erase_bad = 0
    positions = 0
    for e in walk(env, range(400, 408), 8):
        grid = grid_of(e)
        positions += 1
        mask_bad += int(not np.array_equal(
            e.get_action_mask(), index.legal_mask_from_grid(grid).numpy()[0]))
        obs_worst = max(obs_worst,
                        float(np.abs(e._get_obs() - index.observation(grid).numpy()[0]).max()))
        legal = np.flatnonzero(e.get_action_mask())[:12]
        counts = index.apple_counts(grid).numpy()[0]
        after = index.erase(grid.expand(len(legal), rows, cols),
                            torch.as_tensor(legal, dtype=torch.long)).numpy()
        saved = e.board.grid.copy()
        for slot, action_index in enumerate(legal):
            probe = Board.from_board(saved)
            _, removed = probe.do_action(e.index_to_action[int(action_index)])
            apple_bad += int(counts[action_index] != removed)
            erase_bad += int(not np.array_equal(probe.grid, after[slot].astype(np.int8)))
    check(f"합법 판정 == game.Board ({positions}개 국면)", mask_bad == 0, f"불일치 {mask_bad}")
    check("관측 인코딩 == env._get_obs()", obs_worst < 1e-6, f"최대 오차 {obs_worst:.2e}")
    check("사과 개수 == Board.do_action", apple_bad == 0, f"불일치 {apple_bad}")
    check("판 지우기 == Board.do_action", erase_bad == 0, f"불일치 {erase_bad}")

    print("\n== 롤아웃이 실제 게임과 같은 게임을 두는가  ← 가장 중요 ==")
    mismatch = compared = 0
    for e in walk(env, range(500, 504), 3):
        grid = grid_of(e)
        legal = np.flatnonzero(e.get_action_mask())[:6]
        actions = torch.as_tensor(legal, dtype=torch.long)
        with torch.no_grad():
            gpu = q_net.rollout_returns(
                grid.expand(len(legal), rows, cols).contiguous(),
                torch.zeros(len(legal), dtype=torch.long), actions, 0.0).numpy()

        saved = e.board.grid.copy()
        for slot, action_index in enumerate(legal):
            # 같은 정책(로짓 argmax)을 game.Board 로 직접 돌려 본다
            board = Board.from_board(saved)
            _, total = board.do_action(e.index_to_action[int(action_index)])
            while board.get_valid_actions():
                e.board = board
                with torch.no_grad():
                    logits = q_net.policy_logits(torch.as_tensor(e._get_obs()[None]))
                pick = int(logits.argmax())
                ok, removed = board.do_action(e.index_to_action[pick])
                assert ok, "정책이 불법 수를 골랐다"
                total += removed
            compared += 1
            mismatch += int(total != int(gpu[slot]))
        e.board = Board.from_board(saved)
    check(f"GPU 롤아웃 점수 == 엔진으로 돌린 점수 ({compared}개)", mismatch == 0,
          f"불일치 {mismatch}")

    print("\n== 행동 선택 ==")
    env.reset(options={"board_source": 77})
    obs = torch.as_tensor(env._get_obs()[None])
    mask_np = env.get_action_mask()
    with torch.no_grad():
        q = q_net(obs)
        logits = q_net.policy_logits(obs)
    mask = torch.as_tensor(mask_np)[None]
    check("모양이 (1, 7533)", tuple(q.shape) == (1, 7533))
    check("불법 수는 -1e8", bool((q[~mask] <= -1e7).all()))
    check("argmax 가 합법", bool(mask_np[int(q.argmax())]))

    if model_mod.ROLLOUT_TOPK <= 0:
        check("경량 모드: forward == 정책 로짓", bool(torch.equal(q, logits)))
    else:
        # 무거운 모드: 후보 중 롤아웃 점수가 최고인 수를 골라야 한다
        legal = np.flatnonzero(mask_np)
        top_k = min(model_mod.ROLLOUT_TOPK, legal.size)
        candidates = logits.topk(top_k, dim=1).indices[0]
        with torch.no_grad():
            returns = q_net.rollout_returns(
                torch.as_tensor(env.board.grid, dtype=torch.float32)[None]
                .expand(top_k, rows, cols).contiguous(),
                torch.zeros(top_k, dtype=torch.long), candidates, 0.0)
        check("argmax == 롤아웃 점수 최고 후보",
              int(candidates[int(returns.argmax())]) == int(q.argmax()),
              f"{int(candidates[int(returns.argmax())])} vs {int(q.argmax())}")
        check("후보에서 빠진 합법수는 후보보다 낮게 매겨진다",
              float(q[0][candidates].min()) > model_mod.PRUNED_FILL)
        check("롤아웃 점수가 0~162 범위", bool(((returns >= 0) & (returns <= 162)).all()),
              f"[{float(returns.min()):.0f}, {float(returns.max()):.0f}]")

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

    if target == "v13":
        print("\n== ImprovedActionBuffer ==")
        import train as train_mod
        from gymnasium import spaces
        buf = train_mod.ImprovedActionBuffer(
            200, spaces.Box(0.0, 1.0, (13, rows, cols), dtype=np.float32),
            spaces.Discrete(7533), device="cpu", n_envs=1)
        dummy = np.zeros((1, 13, rows, cols), dtype=np.float32)
        rew = np.zeros(1, dtype=np.float32)
        # 3수 에피소드: 점수 4 -> 9 -> 15, 두 번째 수만 무작위 탐험
        for step, (score, greedy) in enumerate([(4, True), (9, False), (15, True)]):
            buf._greedy_flags = np.array([greedy])
            buf.add(dummy, dummy, np.array([[step]]), rew, np.array([step == 2]),
                    [{"score": score}])
        check("전 슬롯에 결말 라벨", bool(buf.labelled[:3, 0].all()))
        check("remain = 이 상태에서 앞으로 얻은 점수",
              list(buf.remain[:3, 0]) == [15.0, 11.0, 6.0], str(list(buf.remain[:3, 0])))
        check("무작위로 둔 수는 교사가 아니다", list(buf.greedy[:3, 0]) == [True, False, True])
        sample = buf.sample_improved(8)
        check("sample_improved 는 greedy 인 것만 낸다",
              sample is not None and set(np.asarray(sample[1]).tolist()) <= {0, 2},
              str(sorted(set(np.asarray(sample[1]).tolist()))))
        check("다음 에피소드는 0점에서 다시 센다", buf._prev_score[0] == 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=("v13", "v13L"))
    args = parser.parse_args()
    print(f"== {args.target} 검증 ==")
    main(args.target)
    print()
    if FAILED:
        print(f"실패 {len(FAILED)}개: {FAILED}")
        raise SystemExit(1)
    print("전부 통과")
