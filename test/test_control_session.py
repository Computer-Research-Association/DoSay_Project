"""기기 없이 조작 루프를 검증한다.

화면을 흉내내는 가짜 객체를 두고 DeviceSession 을 그대로 돌린다. 검증하는 것은
셋이다: (1) 판이 끝까지 진행되는가, (2) 화면 좌표 변환이 맞는가,
(3) 드래그가 누락됐을 때 화면 대조가 그것을 잡아내고 복구하는가.

    python test/test_control_session.py
"""

import os
import sys
from pathlib import Path

import numpy as np

import game

ROOT_DIR = Path(game.__file__).resolve().parent.parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR / "runs"))

from control import session as session_mod            # noqa: E402
from control.controller import DryRunProcessor        # noqa: E402
from control.device import DeviceDisconnected, device_call  # noqa: E402
from control.session import DeviceSession, Orientation, SessionConfig  # noqa: E402
from control.vision import GridGeometry               # noqa: E402
from game.action import Action                        # noqa: E402
from game.board import Board                          # noqa: E402

ROWS, COLS = 9, 18
PITCH, ORIGIN = 60, 100

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(name)


def fake_geometry(rows: int, cols: int) -> GridGeometry:
    centers = np.array([[(ORIGIN + c * PITCH, ORIGIN + r * PITCH) for c in range(cols)]
                        for r in range(rows)], dtype=np.float32)
    return GridGeometry(centers=centers, pitch_x=PITCH, pitch_y=PITCH,
                        radius=PITCH * 0.4, rows=rows, cols=cols)


class FakeScreen:
    """화면의 진실. 드래그를 받으면 반영하되 drop_moves 에 든 순번은 무시한다."""

    def __init__(self, screen_grid, drop_moves=()):
        self.board = Board.from_board(screen_grid)
        self.drop_moves = set(drop_moves)
        self.received = 0

    def apply(self, screen_action: Action) -> None:
        self.received += 1
        if self.received in self.drop_moves:
            return                       # 드래그가 먹지 않은 상황
        ok, _ = self.board.do_action(screen_action)
        if not ok:
            raise AssertionError(f"화면 좌표가 틀렸다: {screen_action} 이 화면에서 불법")


class FakeController(DryRunProcessor):
    def __init__(self, geom, screen: FakeScreen, **kwargs):
        super().__init__(geom, **kwargs)
        self.screen = screen

    def move(self, action: Action) -> None:
        super().move(action)             # 명령 기록 + 픽셀 좌표 계산
        self.screen.apply(action)


class FirstValidAgent:
    """가장 단순한 에이전트. 루프 자체를 재는 것이 목적이라 판정은 아무래도 좋다."""

    def select_action(self, board: Board) -> Action | None:
        actions = board.get_valid_actions()
        return actions[0] if actions else None


def install_fake_vision(screen: FakeScreen) -> None:
    """vision 을 화면 흉내로 갈아끼운다. capture 가 넘긴 것을 resync_board 가 그대로 읽는다."""
    session_mod.vision.capture = lambda device: device
    session_mod.vision.resync_board = lambda img, geom, known: img.board.grid.copy()


def build(screen: FakeScreen, orientation: Orientation, config: SessionConfig,
          geom: GridGeometry) -> tuple[DeviceSession, FakeController]:
    controller = FakeController(geom, screen)
    sess = DeviceSession(screen, geom, templates=None, controller=controller,  # type: ignore[arg-type]
                         orientation=orientation, config=config)
    sess.known_values = screen.board.grid.copy()
    sess.board = Board.from_board(orientation.to_agent(screen.board.grid))
    return sess, controller


# ── 1. 방향 판별 ────────────────────────────────────────────────────────────
print("\n[1] Orientation")

check("가로 화면은 그대로", not Orientation.detect(fake_geometry(9, 18), (ROWS, COLS)).transposed)
check("세로 화면은 전치", Orientation.detect(fake_geometry(18, 9), (ROWS, COLS)).transposed)

try:
    Orientation.detect(fake_geometry(10, 10), (ROWS, COLS))
    check("엉뚱한 격자는 거부", False)
except RuntimeError:
    check("엉뚱한 격자는 거부", True)

# 전치 좌표가 실제로 같은 사과를 가리키는가
screen_grid = Board.from_seed((18, 9), 1234).grid
flip = Orientation(transposed=True)
agent_grid = flip.to_agent(screen_grid)
a = Board.from_board(agent_grid).get_valid_actions()[0]
s = flip.to_screen_action(a)
(ar1, ac1), (ar2, ac2) = a.top_left, a.bottom_right
(sr1, sc1), (sr2, sc2) = s.top_left, s.bottom_right
check("전치 액션이 같은 사과를 가리킨다",
      np.array_equal(agent_grid[ar1:ar2 + 1, ac1:ac2 + 1],
                     screen_grid[sr1:sr2 + 1, sc1:sc2 + 1].T),
      f"agent {a.top_left}~{a.bottom_right} -> screen {s.top_left}~{s.bottom_right}")

# ── 2. 대조 없이 한 판 ──────────────────────────────────────────────────────
print("\n[2] 화면 대조 없이 한 판 (resync-every 0)")

geom = fake_geometry(ROWS, COLS)
screen = FakeScreen(Board.from_seed((ROWS, COLS), 1234).grid)
install_fake_vision(screen)
sess, ctrl = build(screen, Orientation(False), SessionConfig(move_delay=0, resync_every=0), geom)
result = sess.play(FirstValidAgent(), agent_name="first-valid")

check("수를 뒀다", result.moves > 0, f"{result.moves}수")
check("점수 = 162 - 잔여", result.score == 162 - result.remaining,
      f"{result.score} vs {162 - result.remaining}")
check("드래그 수 = 보낸 드래그", len(ctrl.commands) == result.drags)
check("누락이 없으면 수 = 드래그", result.moves == result.drags,
      f"{result.moves} vs {result.drags}")
check("화면과 내부 판이 같다", np.array_equal(screen.board.grid, sess.board.grid))
check("더 둘 수 없다", not sess.board.get_valid_actions())

# ── 3. 매 수 대조, 드래그 정상 ──────────────────────────────────────────────
print("\n[3] 매 수 대조, 드래그 정상 (resync-every 1)")

screen = FakeScreen(Board.from_seed((ROWS, COLS), 1234).grid)
install_fake_vision(screen)
sess, ctrl = build(screen, Orientation(False),
                   SessionConfig(move_delay=0, resync_every=1, resync_retry_delay=0,
                                 final_settle_delay=0), geom)
clean = sess.play(FirstValidAgent(), agent_name="first-valid")

check("대조를 했다", clean.resync_count > 0, f"{clean.resync_count}회")
check("어긋난 적 없다", clean.resync_mismatch == 0)
check("대조를 넣어도 점수가 같다", clean.score == result.score, f"{clean.score} vs {result.score}")

# ── 4. 드래그 누락 복구 ─────────────────────────────────────────────────────
print("\n[4] 드래그가 누락돼도 복구하는가 (3, 7번째 수를 화면이 무시)")

screen = FakeScreen(Board.from_seed((ROWS, COLS), 1234).grid, drop_moves=(3, 7))
install_fake_vision(screen)
sess, ctrl = build(screen, Orientation(False),
                   SessionConfig(move_delay=0, resync_every=1, resync_retry_delay=0,
                                 final_settle_delay=0), geom)
dropped = sess.play(FirstValidAgent(), agent_name="first-valid")

check("누락을 잡아냈다", dropped.resync_mismatch >= 2, f"{dropped.resync_mismatch}회 불일치")
check("누락된 드래그는 수로 세지 않는다", dropped.moves < dropped.drags,
      f"수 {dropped.moves} < 드래그 {dropped.drags}")
check("끝까지 갔다", not sess.board.get_valid_actions())
check("화면과 내부 판이 같다", np.array_equal(screen.board.grid, sess.board.grid))
check("점수 = 162 - 잔여", dropped.score == 162 - dropped.remaining,
      f"{dropped.score} vs {162 - dropped.remaining}")

# ── 5. 세로 화면으로 한 판 ──────────────────────────────────────────────────
print("\n[5] 세로 화면(18x9)으로 한 판")

geom_p = fake_geometry(18, 9)
screen = FakeScreen(Board.from_seed((18, 9), 1234).grid)
install_fake_vision(screen)
sess, ctrl = build(screen, Orientation(True),
                   SessionConfig(move_delay=0, resync_every=1, resync_retry_delay=0,
                                 final_settle_delay=0), geom_p)
portrait = sess.play(FirstValidAgent(), agent_name="first-valid")

check("수를 뒀다", portrait.moves > 0, f"{portrait.moves}수")
check("어긋난 적 없다", portrait.resync_mismatch == 0)
check("화면과 내부 판이 전치 관계", np.array_equal(screen.board.grid.T, sess.board.grid))

# 6. 드래그가 하나도 먹지 않을 때 (dry-run 과 '케이블은 붙었는데 터치가 안 먹는' 경우)
print("\n[6] 드래그가 하나도 먹지 않을 때")

all_dropped = tuple(range(1, 500))

screen = FakeScreen(Board.from_seed((ROWS, COLS), 1234).grid, drop_moves=all_dropped)
install_fake_vision(screen)
sess, ctrl = build(screen, Orientation(False),
                   SessionConfig(move_delay=0, resync_every=0, verify_on_finish=False), geom)
dry = sess.play(FirstValidAgent(), agent_name="first-valid")
check("최종 검증을 끄면 끝난다 (dry-run)", True, f"{dry.drags}번 드래그")
# dry-run 은 화면을 안 건드릴 뿐 내부 판은 정상 진행한다. '이렇게 두면 몇 점'
# 을 보여주는 것이 목적이므로 정상 실행과 같은 수/점수가 나와야 한다.
check("dry-run 도 예상 수/점수를 그대로 낸다",
      (dry.moves, dry.score) == (result.moves, result.score),
      f"{dry.moves}수 {dry.score}점 vs {result.moves}수 {result.score}점")

screen = FakeScreen(Board.from_seed((ROWS, COLS), 1234).grid, drop_moves=all_dropped)
install_fake_vision(screen)
sess, ctrl = build(screen, Orientation(False),
                   SessionConfig(move_delay=0, resync_every=1, resync_retry_delay=0,
                                 final_settle_delay=0, max_resync_failures=3), geom)
stuck = sess.play(FirstValidAgent(), agent_name="first-valid")
check("터치가 안 먹으면 중단한다", "연속 어긋남" in stuck.stopped_reason, stuck.stopped_reason)
check("무한히 돌지 않는다", stuck.drags < 50, f"{stuck.drags}번 드래그 후 중단")

# 7. 도중에 연결이 끊기면 조용히 넘어가지 않고 멈춘다
print("\n[7] 플레이 도중 연결이 끊기면")


class DroppingController(FakeController):
    """5번째 드래그에서 케이블이 빠진 상황."""

    def move(self, action):
        if len(self.commands) >= 5:
            raise DeviceDisconnected("테스트: 케이블이 빠졌다")
        super().move(action)


screen = FakeScreen(Board.from_seed((ROWS, COLS), 1234).grid)
install_fake_vision(screen)
sess = DeviceSession(screen, geom, templates=None,  # type: ignore[arg-type]
                     controller=DroppingController(geom, screen),
                     orientation=Orientation(False),
                     config=SessionConfig(move_delay=0, resync_every=0))
sess.known_values = screen.board.grid.copy()
sess.board = Board.from_board(screen.board.grid)

try:
    sess.play(FirstValidAgent(), agent_name="first-valid")
    check("DeviceDisconnected 로 멈춘다", False, "예외 없이 끝났다")
except DeviceDisconnected as e:
    check("DeviceDisconnected 로 멈춘다", True, str(e))

check("끊기기 전까지 둔 수는 판에 남는다", sess.moves == 5, f"{sess.moves}수")

print("\n[8] device_call 이 연결 오류를 갈아끼운다")

for exc in (RuntimeError("ERROR: 'FAIL' device not found"),
            ConnectionResetError("연결이 재설정됨"),
            ValueError("빈 응답")):
    try:
        device_call(lambda: (_ for _ in ()).throw(exc))
        check(f"{type(exc).__name__} 을 바꾼다", False)
    except DeviceDisconnected:
        check(f"{type(exc).__name__} 을 바꾼다", True)

check("정상 호출은 그대로 통과", device_call(lambda: 42) == 42)

print("\n[9] --drag-ms 가 motion 모드에도 먹는다")

slow = DryRunProcessor(geom, mode="motion", duration_ms=800, motion_steps=8)
fast = DryRunProcessor(geom, mode="motion", duration_ms=80, motion_steps=8)
check("걸음 간격 = 드래그 시간 / 걸음 수", slow.motion_step_sleep == 0.1,
      f"{slow.motion_step_sleep}")
check("드래그 시간을 줄이면 간격도 준다", fast.motion_step_sleep == 0.01,
      f"{fast.motion_step_sleep}")

act = Board.from_board(Board.from_seed((ROWS, COLS), 1234).grid).get_valid_actions()[0]
slow.move(act)
cmd = slow.commands[0]
check("명령에 그 간격이 들어간다", "sleep 0.1000" in cmd)
check("DOWN -> MOVE*8 -> UP 구조", cmd.count("MOVE") == 8 and "DOWN" in cmd and "UP" in cmd,
      f"MOVE {cmd.count('MOVE')}개")

swipe = DryRunProcessor(geom, mode="swipe", duration_ms=800)
swipe.move(act)
check("swipe 는 duration 을 그대로 붙인다", swipe.commands[0].endswith(" 800"),
      swipe.commands[0])

print()
if failures:
    print(f"실패 {len(failures)}건: {', '.join(failures)}")
    raise SystemExit(1)
print("전부 통과")
