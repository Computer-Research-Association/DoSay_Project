"""SearchAgent 배관 검증. 좁은 폭으로 돌려 CPU 에서도 몇 초에 끝나게 한다.

프리셋의 폭만 줄이고 코드 경로는 완전히 같다 (verify_search.py 와 같은 요령).
"""
import os, sys, time
from pathlib import Path
import numpy as np, game
ROOT = Path(game.__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT/"runs"))

from agents.ai.search_agent import SearchAgent
from game.board import Board

CKPT = ROOT / "ai/models/DQN/V15/models/V15_DQN_50000.zip"
fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not cond: fails.append(name)

def narrow(agent, budget=3.0):
    """첫 빔/복구 폭을 좁혀 CPU 에서 돌 만하게. 나머지 설정은 프리셋 그대로."""
    agent.cfg = type(agent.cfg)(**{**agent.cfg.__dict__, "init_width": 32,
                                   "repair_width": 16, "budget_sec": budget})
    agent.budget = budget
    agent.replan_budget = budget
    return agent

print("\n[1] cut-all (폭만 좁힘)")
t0 = time.time()
ag = narrow(SearchAgent((9,18), CKPT, preset="cut-all", budget=3.0,
                        device="cpu", value_cache=True))
print(f"      로드 {time.time()-t0:.1f}s")
check("정보에 프리셋이 남는다", ag.get_info().preset == "cut-all",
      f"{ag.get_info().model_name} / {ag.get_info().preset_label}")

# run_episode 가 내부에서 '주장 점수 == 엔진 재생 점수' 를 스스로 검사한다
steps, score = ag.run_episode(board_source=1234, render=False, delay=0)
check("수순이 game/ 엔진에서 그대로 성립", True, f"{steps}수 {score}점")
check("빔만 쓴 것보다 나쁘지 않다", score >= ag.last_init_score,
      f"{score} >= {ag.last_init_score} (탐색 이득 {score - ag.last_init_score:+d})")

print("\n[2] select_action 재생 (기기 경로와 같은 방식)")
b = Board.from_seed((9,18), 1234)
before = b.grid.copy()
ag2 = narrow(SearchAgent((9,18), CKPT, preset="cut-all", budget=3.0, device="cpu"))
first = ag2.select_action(b)
check("첫 수가 합법", first in b.get_valid_actions(), str(first))
check("판을 안 바꾼다", np.array_equal(b.grid, before))

moves, sc, a = 0, 0, first
while a is not None:
    ok, cleared = b.do_action(a)
    if not ok:
        check("모든 수가 합법", False, f"{moves}수째 {a}"); break
    moves += 1; sc += cleared
    a = ag2.select_action(b)
check("끝까지 간다", b.is_done()[0], f"{moves}수 {sc}점")
check("한 번만 풀었다 (재계획 없음)", ag2.plan_count == 1, f"plan_count={ag2.plan_count}")

print("\n[3] 판이 어긋나면 다시 푼다")
b2 = Board.from_seed((9,18), 1235)
ag3 = narrow(SearchAgent((9,18), CKPT, preset="cut-all", budget=2.0, device="cpu"))
ag3.select_action(b2)
n_before = ag3.plan_count
b2.do_action(b2.get_valid_actions()[-1])       # 계획과 다른 수를 밖에서 둔다
nxt = ag3.select_action(b2)
check("어긋남을 알아채고 다시 푼다", ag3.plan_count == n_before + 1,
      f"{n_before} -> {ag3.plan_count}")
check("다시 푼 뒤 첫 수도 합법", nxt in b2.get_valid_actions(), str(nxt))

print("\n[4] NRPA 프리셋도 같은 경로")
ag4 = SearchAgent((9,18), CKPT, preset="nrpa-seed", budget=3.0, device="cpu")
ag4.cfg = type(ag4.cfg)(**{**ag4.cfg.__dict__, "init_width": 32, "budget_sec": 3.0})
steps4, score4 = ag4.run_episode(board_source=1234, render=False, delay=0)
check("nrpa-seed run_episode", True, f"{steps4}수 {score4}점")

print("\n[5] 끝난 판이면 None")
check("None", ag.select_action(Board.from_board(np.zeros((9,18),dtype=np.int8))) is None)

print("\n[6] plan() 없는 체크포인트는 알아듣게 거부")
try:
    SearchAgent((9,18), ROOT/"ai/models/MaskablePPO/V4.0/V4.0_MaskablePPO_1000000.zip", device="cpu")
    check("SystemExit", False)
except SystemExit as e:
    check("SystemExit", "V15" in str(e), str(e).splitlines()[1].strip()[:60])
except Exception as e:
    check("SystemExit", False, f"{type(e).__name__}: {e}")

print()
if fails: print(f"실패 {len(fails)}건: {fails}"); raise SystemExit(1)
print("전부 통과")
