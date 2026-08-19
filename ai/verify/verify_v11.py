"""V11 계열 + V10c 로드 수정의 단위 검증. 학습을 돌리지 않고 핵심 로직만 본다.

    python ai/verify/verify_v11.py --target loader
    python ai/verify/verify_v11.py --target v11a
    python ai/verify/verify_v11.py --target v11b
    python ai/verify/verify_v11.py --target v11c

버전마다 env.py/model.py 를 같은 이름으로 import 하므로 한 프로세스에서 여러
버전을 볼 수 없다. **--target 하나씩 따로 실행해야 한다.**

여기서 보는 것은 학습이 시작되기 전에 틀릴 수 있는 것들이다 — 마스킹이 게임
엔진과 일치하는가, 그룹 환경이 같은 판을 반복하는가, 보상식이 의도대로인가,
GPU 로 만든 라벨이 정확한가. 실제 학습 경로는 smoke_v11.py 가 본다.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # ai/verify/ -> 저장소 루트

FAILED = []


def check(name, condition, detail=""):
    mark = "OK  " if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        FAILED.append(name)


# ══════════════════════════════════════════════════════════════════════════
def target_loader():
    """V10c 크래시 재현 + custom_objects 로 고쳐지는지."""
    import types
    import cloudpickle
    from stable_baselines3.common.save_util import data_to_json, json_to_data

    print("\n== V10c 로드 수정 ==")

    # 학습 상황 재현: 버전 폴더가 sys.path 에 있어 env.py 가 최상위 모듈 'env' 로 잡힌다
    fake = types.ModuleType("env")
    exec("class AppleGameEnv:\n    def __init__(self):\n        self.board = None\n", fake.__dict__)
    sys.modules["env"] = fake
    instance = fake.AppleGameEnv()  # type: ignore[attr-defined]

    payload = data_to_json({"_search_env": instance, "gamma": 0.997})
    check("cloudpickle 이 env 모듈을 참조로 저장한다",
          b"env" in cloudpickle.dumps(instance))

    del sys.modules["env"]  # measure.py 상황: 'env' 라는 모듈이 없다

    try:
        json_to_data(payload)
        crashed = False
    except ModuleNotFoundError as exc:
        crashed = "env" in str(exc)
    except Exception:
        crashed = False
    check("custom_objects 없이는 ModuleNotFoundError 로 죽는다", crashed)

    restored = json_to_data(payload, custom_objects={"_search_env": None})
    check("custom_objects 를 주면 로드가 성공한다",
          restored["_search_env"] is None and restored["gamma"] == 0.997)

    # 실제 파일이 선언한 값도 확인
    sys.path.insert(0, str(ROOT))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_v10c_env", ROOT / "ai/models/QRDQN/V10c/env.py")
    module = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    spec.loader.exec_module(module)                          # type: ignore[union-attr]
    declared = getattr(module, "LOAD_CUSTOM_OBJECTS", None)
    check("V10c/env.py 가 LOAD_CUSTOM_OBJECTS 를 선언한다",
          isinstance(declared, dict) and "_search_env" in declared, str(declared))

    # 로더가 그것을 실제로 집어넣는지
    sys.path.insert(0, str(ROOT / "runs"))
    from agents.ai.model_loader import _training_only_objects
    check("model_loader 가 그 선언을 custom_objects 로 넘긴다",
          _training_only_objects(module) == declared)


# ══════════════════════════════════════════════════════════════════════════
def _load_version(rel: str):
    version_dir = ROOT / rel
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(version_dir))
    import env as env_mod
    import model as model_mod
    return env_mod, model_mod, version_dir


def _make_policy(env_mod, model_mod, rows=9, cols=18):
    env = env_mod.make_env(rows, cols)
    return env, model_mod.POLICY_CLASS(
        env.observation_space, env.action_space, lambda _: 1e-4,
        **model_mod.make_policy_kwargs(rows, cols))


def _mask_matches_engine(env, q_net, torch, steps=6, trials=4):
    """신경망 안 마스킹이 game.Board 판정과 일치하는가."""
    import numpy as np
    ok = True
    for t in range(trials):
        obs, _ = env.reset(options={"board_source": 500 + t})
        for _ in range(steps):
            engine = env.get_action_mask()
            with torch.no_grad():
                net = q_net.legal_mask(torch.as_tensor(obs[None])).numpy()[0]
            if not np.array_equal(engine, net):
                ok = False
            legal = np.flatnonzero(engine)
            if legal.size == 0:
                break
            obs, _, term, trunc, _ = env.step(int(np.random.choice(legal)))
            if term or trunc:
                break
    return ok


# ══════════════════════════════════════════════════════════════════════════
def target_v11a():
    import numpy as np
    import torch

    env_mod, model_mod, _ = _load_version("ai/models/DQN/V11a")
    import train as train_mod

    print("\n== V11a: 그룹 환경 ==")
    genv = env_mod.make_train_env(9, 18)
    seen, groups, slots = [], [], []
    for _ in range(env_mod.GROUP_SIZE * 2 + 1):
        genv.reset()
        seen.append(genv.board.grid.copy())
        groups.append(genv._group_index)
        slots.append(genv._group_slot)

    k = env_mod.GROUP_SIZE
    check(f"같은 판을 {k}번 반복한다",
          all(np.array_equal(seen[0], b) for b in seen[:k])
          and not np.array_equal(seen[0], seen[k]))
    check("그룹 번호가 K판마다 하나씩 증가한다",
          groups[:k] == [0] * k and groups[k:2 * k] == [1] * k)
    check("group_slot 이 1..K 로 돈다", slots[:k] == list(range(1, k + 1)))
    check("info 에 group_index 가 실린다", "group_index" in genv._get_info())

    plain = env_mod.make_env(9, 18)
    plain.reset(options={"board_source": 1234})
    a = plain.board.grid.copy()
    plain.reset(options={"board_source": 1234})
    check("make_env(평가용)은 그룹 로직을 타지 않는다",
          np.array_equal(a, plain.board.grid) and not hasattr(plain, "_group_index"))

    print("\n== V11a: 듀얼링 신경망 ==")
    env, policy = _make_policy(env_mod, model_mod)
    q_net = policy.q_net
    obs, _ = env.reset(options={"board_source": 7})
    obs_t = torch.as_tensor(obs[None])

    with torch.no_grad():
        q = q_net(obs_t)
        mask = q_net.legal_mask(obs_t)
        feats = q_net._split(q_net.extract_features(obs_t, q_net.features_extractor))
        value = q_net.value_head(feats[0], feats[1])
        adv = q_net.q_net(feats[0], feats[1])

    check("Q 모양이 (1, 7533)", tuple(q.shape) == (1, 7533), str(tuple(q.shape)))
    check("불법 수는 -1e8 로 눌린다", bool((q[~mask] <= -1e7).all()))
    legal_q = q[mask]
    legal_adv = adv[mask]
    check("합법수에 대한 Q 평균 = V(s)  (듀얼링 항등식)",
          bool(torch.allclose(legal_q.mean(), value.squeeze(), atol=1e-4)),
          f"{legal_q.mean().item():.6f} vs {value.item():.6f}")
    check("Q 의 형제 서열 = A 의 형제 서열",
          bool(torch.equal(legal_q.argsort(), legal_adv.argsort())))
    check("V(s) 는 0 에서 시작한다 (마지막 층 0 초기화)",
          abs(value.item()) < 1e-6, f"{value.item():.2e}")
    check("신경망 마스킹이 game.Board 와 일치한다", _mask_matches_engine(env, q_net, torch))

    print("\n== V11a: 그룹 엘리트 버퍼 ==")
    buf = train_mod.GroupEliteBuffer(n_envs=1, capacity=10_000, min_gap=2.0)
    obs_dummy = np.zeros((13, 9, 18), dtype=np.float32)
    # 그룹 0: 점수 100/120/105  -> 최고 120, 평균 108.3, gap 11.7 -> 채택
    for score, action in ((100, 11), (120, 22), (105, 33)):
        buf.record_step(0, obs_dummy, action)
        buf.end_episode(0, score, group_index=0)
    # 그룹 1: 점수 110/110      -> gap 0 -> 기각
    for score, action in ((110, 44), (110, 55)):
        buf.record_step(0, obs_dummy, action)
        buf.end_episode(0, score, group_index=1)
    buf.end_episode(0, 0.0, group_index=2)   # 그룹 1 마감 유발

    check("그룹 안 최고 롤아웃만 채택한다", buf.actions == [22], str(buf.actions))
    check("차이가 없는 그룹(gap<min_gap)은 기각한다", buf.groups_accepted == 1,
          f"closed={buf.groups_closed}, accepted={buf.groups_accepted}")
    stats = buf.stats()
    check("group_spread 를 기록한다", stats["elite/group_spread"] > 0, f"{stats}")

    print("\n== V11a: large-margin 손실 ==")
    model = train_mod.GroupEliteDQN.__new__(train_mod.GroupEliteDQN)
    q = torch.tensor([[0.0, 1.0, 0.5]])
    expert = torch.tensor([1])
    margin = torch.full_like(q, train_mod.ELITE_MARGIN)
    margin.scatter_(1, expert[:, None], 0.0)
    loss = ((q + margin).max(dim=1).values - q.gather(1, expert[:, None]).squeeze(1)).mean()
    check("교사 행동이 충분히 앞서면 손실 0", abs(loss.item()) < 1e-6, f"{loss.item():.6f}")

    q2 = torch.tensor([[0.0, 1.0, 1.02]])       # 다른 수가 0.02 앞선다 (margin 0.05 미만)
    margin2 = torch.full_like(q2, train_mod.ELITE_MARGIN)
    margin2.scatter_(1, expert[:, None], 0.0)
    loss2 = ((q2 + margin2).max(dim=1).values - q2.gather(1, expert[:, None]).squeeze(1)).mean()
    check("교사가 뒤지면 양수 손실", loss2.item() > 0, f"{loss2.item():.4f}")
    check("손실이 유계다 (CE 처럼 발산하지 않는다)",
          loss2.item() <= train_mod.ELITE_MARGIN + 0.05 + 1e-6, f"{loss2.item():.4f}")


# ══════════════════════════════════════════════════════════════════════════
def target_v11b():
    import numpy as np

    env_mod, model_mod, _ = _load_version("ai/models/MaskablePPO/V11b")

    print("\n== V11b: 그룹 상대 보상 ==")
    env = env_mod.make_train_env(9, 18)
    scale = env_mod.GROUP_SCORE_SCALE

    episodes = []
    for _ in range(env_mod.GROUP_SIZE):
        obs, info = env.reset()
        rewards, mid_nonzero = [], 0
        while True:
            mask = env.get_action_mask()
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            obs, reward, term, trunc, info = env.step(int(np.random.choice(legal)))
            rewards.append(reward)
            if not (term or trunc) and reward != 0.0:
                mid_nonzero += 1
            if term or trunc:
                break
        episodes.append((env.score, rewards, mid_nonzero, info))

    check("중간 수의 보상은 전부 0", all(e[2] == 0 for e in episodes))
    check("에피소드 보상 합 = 종료 보상",
          all(abs(sum(e[1]) - e[1][-1]) < 1e-9 for e in episodes))
    check("첫 롤아웃은 전역 평균 기준 (보상 0)", abs(episodes[0][1][-1]) < 1e-9,
          f"{episodes[0][1][-1]:.4f}")

    scores = [e[0] for e in episodes]
    ok = True
    for j in range(1, len(episodes)):
        expected = (scores[j] - float(np.mean(scores[:j]))) / scale
        if abs(episodes[j][1][-1] - expected) > 1e-6:
            ok = False
    check("j번째 보상 = (점수 - 이전 롤아웃 평균)/scale", ok)
    check("같은 판을 반복했다 (그룹 번호 0 유지)",
          all(e[3]["group_index"] == 0 for e in episodes))
    check("점수는 판마다 다르다 (신호가 존재)", len(set(scores)) > 1, str(scores))
    check("info 에 group_baseline 이 실린다", "group_baseline" in episodes[-1][3])

    plain = env_mod.make_env(9, 18)
    plain.reset(options={"board_source": 1234})
    total, steps = 0.0, 0
    while True:
        legal = np.flatnonzero(plain.get_action_mask())
        if legal.size == 0:
            break
        _, reward, term, trunc, info = plain.step(int(np.random.choice(legal)))
        total += reward
        steps += 1
        if term or trunc:
            break
    check("make_env(평가용) 보상 합 = 점수/162 (셰이핑 없음)",
          abs(total - info["score"] / 162) < 1e-6, f"{total:.4f} vs {info['score']/162:.4f}")

    print("\n== V11b: 신경망 (V9c 와 동일해야 한다) ==")
    import subprocess
    diff = subprocess.run(
        ["git", "diff", "--no-index", "--numstat",
         "ai/models/MaskablePPO/V9c/model.py", "ai/models/MaskablePPO/V11b/model.py"],
        cwd=ROOT, capture_output=True, text=True)
    body_same = True
    v9c = (ROOT / "ai/models/MaskablePPO/V9c/model.py").read_text(encoding="utf-8")
    v11b = (ROOT / "ai/models/MaskablePPO/V11b/model.py").read_text(encoding="utf-8")
    body_same = v9c.split('"""', 2)[2] == v11b.split('"""', 2)[2]
    check("model.py 가 V9c 와 코드는 완전히 동일 (독스트링만 다름)", body_same)

    env2, policy = _make_policy(env_mod, model_mod)
    import torch
    obs, _ = env2.reset(options={"board_source": 3})
    with torch.no_grad():
        dist = policy.get_distribution(torch.as_tensor(obs[None]),
                                       action_masks=env2.get_action_mask()[None])
        probs = dist.distribution.probs
    legal = torch.as_tensor(env2.get_action_mask())[None]
    check("불법 수의 확률이 0", float(probs[~legal].sum()) < 1e-6)
    check("합법수 확률 합 = 1", abs(float(probs.sum()) - 1.0) < 1e-4)


# ══════════════════════════════════════════════════════════════════════════
def target_v11c():
    import numpy as np
    import torch

    env_mod, model_mod, _ = _load_version("ai/models/DQN/V11c")
    import train as train_mod
    from game.board import Board

    print("\n== V11c: 환경 ==")
    env = env_mod.make_env(9, 18)
    env.reset(options={"board_source": 42})
    total, info = 0.0, None
    while True:
        legal = np.flatnonzero(env.get_action_mask())
        if legal.size == 0:
            break
        _, reward, term, trunc, info = env.step(int(np.random.choice(legal)))
        total += reward
        if term or trunc:
            break
    check("보상 합 = 점수/162 (셰이핑 제거 확인)",
          abs(total - info["score"] / 162) < 1e-6, f"{total:.4f}")
    check("종료 시 final_grid 를 info 로 낸다", "final_grid" in info)
    check("final_grid 가 실제 최종 판", np.array_equal(info["final_grid"], env.board.grid))
    check("종료 전에는 final_grid 를 싣지 않는다",
          "final_grid" not in env_mod.make_env(9, 18).reset(options={"board_source": 1})[1])

    print("\n== V11c: 신경망 ==")
    env2, policy = _make_policy(env_mod, model_mod)
    q_net = policy.q_net
    obs, _ = env2.reset(options={"board_source": 11})
    obs_t = torch.as_tensor(obs[None])

    with torch.no_grad():
        q, look, survive, mask = q_net.heads(obs_t)
    check("heads() 모양", tuple(q.shape) == (1, 7533) and tuple(look.shape) == (1, 7533)
          and tuple(survive.shape) == (1, 9, 18))
    check("look_weight 는 0 에서 시작", abs(float(q_net.look_weight.item())) < 1e-9)
    check("look 은 0 에서 시작 (전부 0 초기화)", float(look.abs().max()) < 1e-6)
    check("신경망 마스킹이 game.Board 와 일치한다", _mask_matches_engine(env2, q_net, torch))

    print("\n== V11c: grid_from_obs ==")
    # 위 마스킹 검사가 env2 를 진행시켜 놓았으므로 관측을 지금 다시 만든다
    ok = True
    for trial in range(3):
        env2.reset(options={"board_source": 300 + trial})
        for _ in range(3):
            fresh = torch.as_tensor(env2._get_obs()[None])
            rebuilt = model_mod.grid_from_obs(fresh)[0].numpy()
            if not np.array_equal(rebuilt.astype(np.int8), env2.board.grid):
                ok = False
            legal = np.flatnonzero(env2.get_action_mask())
            if legal.size == 0:
                break
            env2.step(int(np.random.choice(legal)))
    check("관측에서 숫자판을 정확히 복원한다", ok)

    print("\n== V11c: afterstate_legal_counts vs 게임 엔진 ==")
    mismatches, compared = 0, 0
    for trial in range(3):
        env2.reset(options={"board_source": 900 + trial})
        for depth in range(4):
            board = env2.board
            engine_actions = board.get_valid_actions()
            if not engine_actions:
                break
            obs_t = torch.as_tensor(env2._get_obs()[None])
            grid = model_mod.grid_from_obs(obs_t)

            legal_idx = np.flatnonzero(env2.get_action_mask())
            pick = legal_idx[:12]
            counts = q_net.index.afterstate_legal_counts(
                grid, torch.as_tensor(pick, dtype=torch.long)[None], chunk=8)[0].numpy()

            for slot, action_index in enumerate(pick):
                probe = Board.from_board(board.grid)
                valid, _ = probe.do_action(env2.index_to_action[int(action_index)])
                truth = len(probe.get_valid_actions()) if valid else -1
                compared += 1
                if int(counts[slot]) != truth:
                    mismatches += 1
            env2.step(int(np.random.choice(legal_idx)))

    check(f"GPU 라벨이 게임 엔진과 완전 일치 ({compared}개 비교)", mismatches == 0,
          f"불일치 {mismatches}개")

    print("\n== V11c: OutcomeReplayBuffer ==")
    from gymnasium import spaces
    buf = train_mod.OutcomeReplayBuffer(
        200, spaces.Box(0.0, 1.0, (13, 9, 18), dtype=np.float32), spaces.Discrete(7533),
        device="cpu", n_envs=1)
    dummy = np.zeros((1, 13, 9, 18), dtype=np.float32)
    act = np.zeros((1, 1), dtype=np.int64)
    rew = np.zeros(1, dtype=np.float32)

    final = np.zeros((9, 18), dtype=np.int8)
    final[0, 0] = 7
    final[3, 5] = 2
    for step in range(5):
        done = np.array([step == 4])
        infos = [{"final_grid": final}] if done[0] else [{}]
        buf.add(dummy, dummy, act, rew, done, infos)

    check("에피소드 전체 슬롯에 라벨이 채워진다", bool(buf.outcome_valid[:5, 0].all()))
    expected = (final.reshape(-1) != 0).astype(np.uint8)
    check("라벨 = (최종 판에서 사과가 남아 있는가)",
          bool((buf.outcome[:5, 0] == expected).all()))
    check("라벨이 붙은 사과 개수 = 2", int(buf.outcome[0, 0].sum()) == 2)

    buf.add(dummy, dummy, act, rew, np.array([False]), [{}])
    check("진행 중 전이는 아직 유효하지 않다", not bool(buf.outcome_valid[5, 0]))

    sampled = buf.sample_outcome(4)
    check("sample_outcome 이 (관측, 라벨) 을 낸다",
          sampled is not None and sampled[0].shape == (4, 13, 9, 18)
          and sampled[1].shape == (4, 162))

    print("\n== V11c: 보조 손실 경로 ==")
    grid_t = model_mod.grid_from_obs(torch.as_tensor(env2._get_obs()[None]))
    mask_t = q_net.legal_mask(torch.as_tensor(env2._get_obs()[None]))
    picks = torch.multinomial(mask_t.float(), train_mod.LOOK_SAMPLES, replacement=True)
    counts = q_net.index.afterstate_legal_counts(grid_t, picks, chunk=64)
    target = torch.log1p(counts) / model_mod.LOOK_LOG_SCALE
    check("look 목표가 [0, 1] 범위로 정규화된다",
          bool(((target >= 0) & (target <= 1.05)).all()), f"max={float(target.max()):.3f}")
    check("multinomial 이 합법수만 뽑는다", bool(mask_t[0][picks[0]].all()))


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True,
                        choices=("loader", "v11a", "v11b", "v11c"))
    args = parser.parse_args()

    import numpy as np
    np.random.seed(0)
    import torch
    torch.manual_seed(0)

    {"loader": target_loader, "v11a": target_v11a,
     "v11b": target_v11b, "v11c": target_v11c}[args.target]()

    print()
    if FAILED:
        print(f"실패 {len(FAILED)}개: {FAILED}")
        raise SystemExit(1)
    print("전부 통과")
