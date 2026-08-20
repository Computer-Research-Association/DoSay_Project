"""기기 위에서 한 판을 진행한다.

내부에 판 하나(game.Board)를 들고 있으면서 에이전트에게 한 수씩 물어보고,
그 수를 화면에 드래그한 뒤 내부 판을 갱신한다. 사과게임은 클리어해도 사과가
이동하지 않으므로 매 수 화면을 다시 읽을 필요가 없다. 대신 N수마다 화면과
대조해서(resync) 드래그가 누락됐으면 화면 기준으로 맞춘다.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from game.action import Action
from game.board import Board

from . import vision
from .controller import PROCESSOR
from .vision import GridGeometry


class ActionSelector(Protocol):
    """세션이 에이전트에게 요구하는 전부. agents.Agent 가 이것을 만족한다."""

    def select_action(self, board: Board) -> Action | None: ...


@dataclass(frozen=True)
class Orientation:
    """화면 격자와 에이전트 판의 방향을 맞춘다.

    폰을 세로로 들면 9x18 판이 화면에는 18x9 로 잡힌다. 반면 에이전트의 행동
    공간은 학습할 때의 (rows, cols) 에 묶여 있어서 바꿀 수가 없다. 그래서 판을
    전치해 에이전트에게 주고, 돌아온 Action 을 화면 좌표로 되돌린다.
    """

    transposed: bool

    @classmethod
    def detect(cls, geom: GridGeometry, agent_shape: tuple[int, int]) -> "Orientation":
        rows, cols = agent_shape
        if (geom.rows, geom.cols) == (rows, cols):
            return cls(transposed=False)
        if (geom.rows, geom.cols) == (cols, rows):
            return cls(transposed=True)
        raise RuntimeError(
            f"화면 격자가 {geom.rows}x{geom.cols} 로 잡혔습니다. "
            f"에이전트는 {rows}x{cols} 판에서만 동작합니다.\n"
            "게임 시작 화면(사과가 전부 있는 상태)인지, 화면에 사과 말고 "
            "주황색 UI 가 함께 잡히지는 않았는지 확인해 주세요.")

    def to_agent(self, screen_grid: NDArray[np.int8]) -> NDArray[np.int8]:
        return screen_grid.T.copy() if self.transposed else screen_grid.copy()

    def to_screen_action(self, action: Action) -> Action:
        """에이전트 좌표의 액션을 화면 좌표로. agent[r][c] == screen[c][r] 이므로 뒤집으면 된다."""
        if not self.transposed:
            return action
        (r1, c1), (r2, c2) = action.top_left, action.bottom_right
        return Action((c1, r1), (c2, r2))

    def describe(self) -> str:
        return "전치 (화면 세로)" if self.transposed else "그대로"


@dataclass(frozen=True)
class SessionConfig:
    move_delay: float = 0.35          # 드래그 후 대기(초). 클리어 애니메이션 시간.
    resync_every: int = 8             # N번째 드래그마다 화면 대조. 0이면 안 함.
    resync_retry_delay: float = 0.3   # 불일치 시 재확인까지 기다리는 시간
    max_resync_failures: int = 3      # 연속 불일치가 이만큼이면 중단
    final_settle_delay: float = 0.6   # 마지막 대조 전 대기
    # 끝난 것 같을 때 화면으로 한 번 더 확인할지. 주기적 대조와 따로 두는 이유는
    # dry-run 때문이다. 드래그를 보내지 않으면 화면이 시작 상태 그대로라, 최종
    # 대조가 판을 통째로 되살려 놓아 루프가 끝나지 않는다.
    verify_on_finish: bool = True


@dataclass
class PlayResult:
    """한 판의 결과. format_dataclass_box 로 그대로 출력한다.

    moves 와 score 는 누적하지 않고 판에서 직접 계산한다. 드래그가 누락되면
    화면 대조가 판을 되돌리는데, 누적해 두면 그 되돌림이 숫자에 반영되지 않아
    실제보다 높은 점수가 찍힌다. (실제로 그 버그를 한 번 냈다.)
    """

    agent: str                = field(metadata={"label": "Agent"})
    moves: int                = field(metadata={"label": "Moves"})
    drags: int                = field(metadata={"label": "Drags sent"})
    score: int                = field(metadata={"label": "Score"})
    remaining: int            = field(metadata={"label": "Remaining apples"})
    resync_count: int         = field(metadata={"label": "Resyncs"})
    resync_mismatch: int      = field(metadata={"label": "Resync mismatches"})
    elapsed_sec: float        = field(metadata={"label": "Elapsed (s)"})
    sec_per_move: float       = field(metadata={"label": "Sec/move"})
    stopped_reason: str       = field(metadata={"label": "Stopped by"})


class DeviceSession:
    def __init__(self, device, geom: GridGeometry, templates: vision.Templates,
                 controller: PROCESSOR, orientation: Orientation,
                 config: SessionConfig = SessionConfig()) -> None:
        self.device = device
        self.geom = geom
        self.templates = templates
        self.controller = controller
        self.orientation = orientation
        self.config = config

        self.board: Board
        self.known_values: NDArray[np.int8]   # 화면 좌표. 사과 값은 판 내내 안 바뀐다.
        self.resync_count = 0
        self.resync_mismatch = 0
        self._consecutive_mismatch = 0
        self._initial_sum = 0
        self._initial_apples = 0

    # ── 판 잡기 ──────────────────────────────────────────────────────────

    def read_initial_board(self, img: NDArray[np.uint8]) -> tuple[Board, float]:
        """시작 화면에서 숫자까지 읽어 판을 세운다. (판, 최소 신뢰도) 반환."""
        screen_grid, confidence = vision.read_board(img, self.geom, self.templates)
        self.known_values = screen_grid.copy()
        self.board = Board.from_board(self.orientation.to_agent(screen_grid))
        self._set_baseline()
        return self.board, confidence

    def _set_baseline(self) -> None:
        """score/moves 를 재는 기준점. 판을 잡은 순간의 합과 사과 수."""
        self._initial_sum = int(self.board.grid.sum())
        self._initial_apples = int(np.count_nonzero(self.board.grid))

    # ── 화면 대조 ────────────────────────────────────────────────────────

    def resync(self) -> bool:
        """화면과 내부 판을 대조. 같으면 True.

        다르면 (클리어 애니메이션이 아직 도는 중일 수 있으므로) 한 번 더 보고,
        그래도 다르면 화면 기준으로 내부 판을 새로 세우고 False 를 돌려준다.
        숫자를 다시 분류하지 않으므로 비용은 screencap 이 거의 전부다.
        """
        self.resync_count += 1

        if self._screen_matches():
            self._consecutive_mismatch = 0
            return True

        time.sleep(self.config.resync_retry_delay)
        if self._screen_matches():
            self._consecutive_mismatch = 0
            return True

        screen_grid = self._read_screen_grid()
        self.board = Board.from_board(self.orientation.to_agent(screen_grid))
        self.resync_mismatch += 1
        self._consecutive_mismatch += 1
        return False

    def _read_screen_grid(self) -> NDArray[np.int8]:
        return vision.resync_board(vision.capture(self.device), self.geom, self.known_values)

    def _screen_matches(self) -> bool:
        agent_grid = self.orientation.to_agent(self._read_screen_grid())
        return bool(np.array_equal(agent_grid, self.board.grid))

    # ── 한 판 ────────────────────────────────────────────────────────────

    def play(self, agent: ActionSelector, agent_name: str = "?",
             on_move: Callable[[int, Action, int, int], None] | None = None) -> PlayResult:
        start = time.time()
        self._set_baseline()
        drags = 0
        reason = "판에 둘 수 있는 수가 없음"

        while True:
            if self._should_resync(drags) and not self.resync():
                if self._consecutive_mismatch >= self.config.max_resync_failures:
                    reason = (f"화면 대조가 {self._consecutive_mismatch}회 연속 어긋남 "
                              "(드래그가 먹지 않는 것으로 보임)")
                    break

            action = agent.select_action(self.board)

            if action is None:
                if not self.config.verify_on_finish:
                    break
                # 정말 끝인지 화면으로 한 번 더 확인한다. 드래그가 하나 누락됐을 뿐인데
                # 끝난 것으로 보고 나가면 남은 점수를 그냥 버리게 된다.
                time.sleep(self.config.final_settle_delay)
                if self.resync() or not self.board.get_valid_actions():
                    break
                if self._consecutive_mismatch >= self.config.max_resync_failures:
                    reason = (f"화면 대조가 {self._consecutive_mismatch}회 연속 어긋남 "
                              "(드래그가 먹지 않는 것으로 보임)")
                    break
                continue

            if action not in self.board.get_valid_actions():
                raise RuntimeError(
                    f"에이전트가 불법 수를 선택했습니다: {action}\n"
                    f"(합법수 {len(self.board.get_valid_actions())}개, {drags}번째 드래그)\n"
                    "그대로 두면 화면과 내부 판이 갈라져 엉뚱한 곳을 드래그하게 됩니다.")

            self.controller.move(self.orientation.to_screen_action(action))
            _, removed = self.board.do_action(action)
            drags += 1

            if on_move is not None:
                on_move(drags, action, removed, self.score)
            if self.config.move_delay > 0:
                time.sleep(self.config.move_delay)

        elapsed = time.time() - start
        moves = self.moves
        return PlayResult(
            agent=agent_name,
            moves=moves,
            drags=drags,
            score=self.score,
            remaining=int(np.count_nonzero(self.board.grid)),
            resync_count=self.resync_count,
            resync_mismatch=self.resync_mismatch,
            elapsed_sec=round(elapsed, 1),
            sec_per_move=round(elapsed / moves, 3) if moves else 0.0,
            stopped_reason=reason,
        )

    # 둘 다 누적하지 않고 판에서 직접 계산한다. 화면 대조가 판을 되돌려도 저절로 맞는다.

    @property
    def score(self) -> int:
        """지운 사과 수. 판을 잡은 시점이 기준이다."""
        return self._initial_apples - int(np.count_nonzero(self.board.grid))

    @property
    def moves(self) -> int:
        """판이 실제로 진행된 수. 한 수는 언제나 합 10 을 가져간다 (game-analysis.md 1절)."""
        return (self._initial_sum - int(self.board.grid.sum())) // 10

    def _should_resync(self, drags: int) -> bool:
        every = self.config.resync_every
        return every > 0 and drags > 0 and drags % every == 0
