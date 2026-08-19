"""V14 검증. 학습 없이 몇십 초면 끝난다.

    python ai/verify/verify_v14.py --target v14     # 빔 모드 (학습 + 무거운 배포)
    python ai/verify/verify_v14.py --target v14L    # 1회 forward
    python ai/verify/verify_v14.py --target v14A    # 깊이 1

가장 중요한 것 둘:

1. **빔이 실제 게임과 같은 게임을 두는가.** 빔이 계획한 수순을 `game.Board` 로
   그대로 재생해서 점수가 일치하는지 본다. 어긋나면 아무 에러 없이 점수만 깎인다.
2. **V12b 워밍스타트가 실제로 붙는가.** 인코더와 가치 헤드의 키가 V12b 와 맞아야
   1스텝째부터 강한 교사를 쓸 수 있다. 조용히 실패하면 학습이 몇 배 느려진다.
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

    folder = "V" + target[1:]
    env_mod, model_mod = load_version(f"ai/models/DQN/{folder}")
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

    print(f"\n== 게임 규칙 대조 (MODE={model_mod.MODE}) ==")
    mask_bad = apple_bad = erase_bad = positions = 0
    obs_worst = 0.0
    for e in walk(env, range(400, 406), 8):
        grid = grid_of(e)
        positions += 1
        mask_bad += int(not np.array_equal(
            e.get_action_mask(), index.legal_mask_from_grid(grid).numpy()[0]))
        obs_worst = max(obs_worst,
                        float(np.abs(e._get_obs() - index.observation(grid).numpy()[0]).max()))
        legal = np.flatnonzero(e.get_action_mask())[:10]
        counts = index.apple_counts(grid).numpy()[0]
        after = index.erase(grid.expand(len(legal), rows, cols),
                            torch.as_tensor(legal, dtype=torch.long)).numpy()
        saved = e.board.grid.copy()
        for slot, a in enumerate(legal):
            probe = Board.from_board(saved)
            _, removed = probe.do_action(e.index_to_action[int(a)])
            apple_bad += int(counts[a] != removed)
            erase_bad += int(not np.array_equal(probe.grid, after[slot].astype(np.int8)))
    check(f"합법 판정 == game.Board ({positions}개 국면)", mask_bad == 0)
    check("관측 인코딩 == env._get_obs()", obs_worst < 1e-6, f"최대 오차 {obs_worst:.2e}")
    check("사과 개수 == Board.do_action", apple_bad == 0)
    check("판 지우기 == Board.do_action", erase_bad == 0)

    print("\n== 빔 계획이 실제 게임과 같은가  ← 가장 중요 ==")
    mismatch = compared = 0
    for seed in range(500, 504):
        env.reset(options={"board_source": seed})
        grid = torch.as_tensor(env.board.grid, dtype=torch.float32)
        with torch.no_grad():
            path, claimed = q_net.plan(grid, width=8)
        board = Board.from_board(env.board.grid)
        replayed = 0
        ok_line = True
        for planned_board, action in path:
            if not np.array_equal(board.grid, planned_board.numpy().astype(np.int8)):
                ok_line = False
                break
            valid, removed = board.do_action(env.index_to_action[int(action)])
            if not valid:
                ok_line = False
                break
            replayed += removed
        compared += 1
        mismatch += int(not ok_line or replayed != claimed)
    check(f"빔 수순을 game.Board 로 재생한 점수 == 빔이 주장한 점수 ({compared}판)",
          mismatch == 0, f"불일치 {mismatch}")

    print("\n== 배포 모드 ==")
    env.reset(options={"board_source": 77})
    obs = torch.as_tensor(env._get_obs()[None])
    mask_np = env.get_action_mask()
    with torch.no_grad():
        q = q_net(obs)
        logits = q_net.policy_logits(obs)
    check("모양이 (1, 7533)", tuple(q.shape) == (1, 7533))
    check("argmax 가 합법", bool(mask_np[int(q.argmax())]))
    check("불법 수는 -1e8", bool((q[0][~torch.as_tensor(mask_np)] <= -1e7).all()))
    if model_mod.MODE == "policy":
        check("policy 모드: forward == 정책 로짓", bool(torch.equal(q, logits)))
    elif model_mod.MODE == "afterstate":
        legal = np.flatnonzero(mask_np)
        with torch.no_grad():
            after = index.erase(torch.as_tensor(env.board.grid, dtype=torch.float32)[None]
                                .expand(len(legal), rows, cols).contiguous(),
                                torch.as_tensor(legal, dtype=torch.long))
            leftover = q_net.expected_leftover(index.observation(after))
        check("afterstate 모드: argmax == 예상 잔여 최소",
              int(legal[int(leftover.argmin())]) == int(q.argmax()))
    else:
        with torch.no_grad():
            first = q_net.plan(torch.as_tensor(env.board.grid, dtype=torch.float32),
                               model_mod.BEAM_WIDTH)[0][0][1]
        check("beam 모드: argmax == 계획의 첫 수", int(first) == int(q.argmax()))
        # 계획을 따라가면 캐시가 계속 적중해야 한다 (판당 빔 1회)
        env.reset(options={"board_source": 77})
        q_net._plan_cache = {}
        obs, steps = env._get_obs(), 0
        while steps < 200:
            m = env.get_action_mask()
            if not m.any():
                break
            with torch.no_grad():
                a = int(q_net(torch.as_tensor(obs[None])).argmax())
            if not m[a]:
                break
            obs, _, t_, tr, info = env.step(a)
            steps += 1
            if t_ or tr:
                break
        check("계획을 끝까지 따라간다 (한 판에 빔 1회)",
              steps > 30 and len(q_net._plan_cache) >= steps,
              f"{steps}수, 캐시 {len(q_net._plan_cache)}개, {info['score']}점")

    if target == "v14":
        print("\n== V12b 워밍스타트 ==")
        import train as train_mod
        path = train_mod.INIT_FROM
        if not path.exists():
            print(f"  [건너뜀] {path.name} 없음 (학습 머신에서 확인 필요)")
        else:
            from stable_baselines3.common.save_util import load_from_zip_file
            _, params, _ = load_from_zip_file(path, device="cpu")
            info = policy.load_state_dict(params["policy"], strict=False)
            check("V12b 의 인코더/가치 헤드가 붙는다", len(info.unexpected_keys) == 0,
                  f"안 붙은 키 {len(info.unexpected_keys)}개")
            check("정책 헤드만 새로 배운다",
                  all("policy_net" in k for k in info.missing_keys),
                  f"새 키 {len(info.missing_keys)}개")

        print("\n== BeamTeacherBuffer ==")
        from gymnasium import spaces
        buf = train_mod.BeamTeacherBuffer(
            200, spaces.Box(0.0, 1.0, (13, rows, cols), dtype=np.float32),
            spaces.Discrete(7533), device="cpu", n_envs=1)
        dummy = np.zeros((1, 13, rows, cols), dtype=np.float32)
        rew = np.zeros(1, dtype=np.float32)
        for step, score in enumerate((4, 9, 15)):
            buf._teacher_flags = np.array([True])
            buf.add(dummy, dummy, np.array([[step]]), rew, np.array([step == 2]),
                    [{"score": score}])
        check("remain = 이 상태에서 앞으로 얻은 점수",
              list(buf.remain[:3, 0]) == [15.0, 11.0, 6.0], str(list(buf.remain[:3, 0])))
        check("전 슬롯에 결말 라벨", bool(buf.labelled[:3, 0].all()))
        check("sample_teacher 동작", buf.sample_teacher(4) is not None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=("v14", "v14L", "v14A"))
    args = parser.parse_args()
    print(f"== {args.target} 검증 ==")
    main(args.target)
    print()
    if FAILED:
        print(f"실패 {len(FAILED)}개: {FAILED}")
        raise SystemExit(1)
    print("전부 통과")
