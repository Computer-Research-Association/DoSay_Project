import copy
import importlib
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from types import ModuleType
from typing import Callable, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from algorithm.models.utils import HeuristicRegistry
from game.action import Action
from game.board import Board
from agents.dtos import AlgoInfo  # type: ignore


class AlgoExecutor(ABC):
    def __init__(self, grid_shape: Tuple[int, int], model_path: Path):
        super().__init__()
        self.grid_shape = grid_shape
        self.model_path = model_path
        self.model_info: AlgoInfo
        self.cls: ModuleType
        self.board: Board
        self._load_model()

    def reset(self, board_source: int | list[int] | NDArray[np.int8]):
        if isinstance(board_source, (np.ndarray, list)):
            _board = np.asarray(board_source, dtype=np.int8)
            self.board = Board.from_board(_board)
            self._score = _board.size - int(np.count_nonzero(_board))
        elif board_source is None or isinstance(board_source, int):
            self.board = Board.from_seed(self.grid_shape, board_source)
            self._score = 0
        else:
            raise TypeError(f"board_source must be ndarray, list, or int, got {type(board_source).__name__}")

        return self.board, self._score

    @abstractmethod
    def select_action(self, board: Board) -> Action | None:
        """주어진 판에서 둘 수 하나. 합법수가 없으면 None.

        판을 바꾸지 않는다. 수를 두는 것은 부르는 쪽의 몫이다.
        실 기기처럼 판의 주인이 밖에 있을 때 쓴다.
        """

    def do_step(self) -> int:
        """내부 판에 한 수 둔다. 지운 사과 수를 돌려준다. 더 둘 수 없으면 0.

        합법수는 반드시 사과를 1개 이상 지우므로, 0 은 '끝났다' 는 뜻으로만 나온다.
        부르는 쪽은 그 성질에 기대어 루프를 끊는다.
        """
        if getattr(self, "board", None) is None:
            raise RuntimeError("Executor reset needed: call reset() before do_step().")

        action = self.select_action(self.board)
        if action is None:
            return 0

        ok, cleared = self.board.do_action(action)
        if not ok:
            # 조용히 0 을 돌려주면 부르는 쪽 루프가 영원히 돈다. 판은 그대로인데
            # 끝나지도 않았으니 같은 수를 무한히 다시 고르게 된다.
            raise RuntimeError(
                f"{type(self).__name__} 이 불법 수를 냈습니다: {action}\n"
                f"  (합법수 {len(self.board.get_valid_actions())}개)")
        return cleared

    def _load_model(self):
        # 파일명 규격: <종류>_<버전>.py  예) Greedy_V04b.py, Anneal_V1.py
        # stem 을 쓰는 이유: name 은 '.py' 를 달고 있어서 버전이 'V1.py' 가 된다.
        parts = self.model_path.stem.split("_")
        if len(parts) != 2:
            raise ValueError(
                f"모델 파일명 규격을 인식할 수 없습니다: '{self.model_path.name}'\n"
                f"  기대 형식: <종류>_<버전>.py  예) Greedy_V04b.py")
        model_type, model_version = parts

        # 'V' 를 떼서 AI 쪽과 규칙을 맞춘다. AI 는 체크포인트 V15_DQN_50000 에서
        # 버전을 '15' 로 뽑고, 화면에 찍는 쪽(measure.py)이 'V' 를 붙인다.
        # 여기서 'V1' 을 그대로 두면 'Anneal / VV1' 이 된다.
        if model_version[:1] in ("V", "v"):
            model_version = model_version[1:]

        source = (self.model_path.relative_to(Path.cwd())
                  if self.model_path.is_absolute() and self.model_path.is_relative_to(Path.cwd())
                  else self.model_path)
        self.model_info = AlgoInfo("Algorithm", model_type, model_version, source)
        self.cls = importlib.import_module(f"algorithm.models.version.{self.model_path.stem}")


class GreedyExecutor(AlgoExecutor):
    """한 수 앞만 본다. 각 후보를 두어 본 판을 버전 파일의 휴리스틱으로 채점한다."""

    def select_action(self, board: Board) -> Action | None:
        actions = board.get_valid_actions()
        if not actions:
            return None

        registry = cast(HeuristicRegistry, self.cls.registry)

        def score_of(action: Action) -> float:
            child_board = copy.deepcopy(board)
            child_board.do_action(action)
            return registry.evaluate(child_board)

        return max(actions, key=score_of)


class BeamSearchExecutor(AlgoExecutor):
    """아직 없다. Beam_*.py 들은 휴리스틱 가중치만 정의해 둔 상태다."""

    def select_action(self, board: Board) -> Action | None:
        raise NotImplementedError(
            "빔서치는 아직 구현되지 않았습니다.\n"
            f"  {__file__} 의 BeamSearchExecutor.select_action 을 채워 주세요.\n"
            "  algorithm/models/version/Beam_*.py 는 휴리스틱 가중치만 갖고 있고,\n"
            "  그것을 어떻게 펼칠지(빔 폭, 깊이)를 정하는 코드가 없습니다.\n"
            "  (조용히 다른 방식으로 두면 'Beam' 이라는 이름표를 달고 다른 성능이 나옵니다)")


# ── 어닐링 ───────────────────────────────────────────────────────────────────

AnnealBackend = Callable[[NDArray[np.int8], int, int], tuple[int, list]]


def _without(grid: NDArray[np.int8], action: Action) -> NDArray[np.int8]:
    """그 수를 둔 뒤의 판. 계획이 예상하는 다음 판을 들고 있으려고 쓴다."""
    (r1, c1), (r2, c2) = action.top_left, action.bottom_right
    after = grid.copy()
    after[r1:r2 + 1, c1:c2 + 1] = 0
    return after


def _resolve_anneal_backend() -> tuple[str, AnnealBackend]:
    """쓸 수 있는 어닐링 구현을 고른다.

    Cython(bench/anneal_cy.pyx)이 빌드돼 있으면 그것을 쓰고, 없으면 numba 이식본을
    쓴다. 둘은 같은 알고리즘이라 점수가 통계적으로 같아야 한다.

    ImportError 만 잡는다. 예전 코드는 except Exception 으로 감싸 놓아서, 빌드된
    Cython 이 실행 중에 터져도 조용히 다른 구현으로 넘어갔다. 어떤 구현으로 낸
    점수인지 모르게 되는 것이 성능 문제보다 나쁘다.
    """
    bench_dir = Path(__file__).resolve().parents[2] / "bench"
    if str(bench_dir) not in sys.path:
        sys.path.append(str(bench_dir))   # reset 마다 넣으면 경로가 무한히 자란다

    try:
        from anneal_cy import anneal_cy  # type: ignore

        def cython_backend(grid, iters, seed):
            return anneal_cy(np.ascontiguousarray(grid, dtype=np.int8), iters, seed)

        return "cython", cython_backend
    except ImportError:
        pass

    from algorithm.models.anneal import anneal_once

    def numba_backend(grid, iters, seed):
        return anneal_once(grid, iters=iters, rng_seed=seed)

    return "numba", numba_backend


class AnnealExecutor(AlgoExecutor):
    """수순 전체를 풀어 두고 한 수씩 재생한다.

    어닐링은 '다음 한 수' 를 묻는 알고리즘이 아니다. 수순 하나를 놓고 일부를 부순 뒤
    다시 이어 붙이기를 반복해 더 나은 수순을 찾는다. 그래서 do_step 이 매번 탐색하는
    대신, 판을 처음 잡을 때 한 번 풀고 그 결과를 재생한다.

    판에 무작위성이 없어서 미리 푸는 것(open-loop)과 매 수 다시 푸는 것(closed-loop)의
    결과가 같다. 다만 판이 밖에서 바뀌면(실 기기 인식 오차 같은) 수순이 어긋나므로,
    select_action 이 그것을 알아채고 다시 푼다.

    버전 파일에서 읽는 값
        ITERS     : 어닐링 반복 수
        INSTANCES : 서로 다른 난수로 몇 번 풀어 그중 최고를 쓸지 (max-of-N)
    """

    def __init__(self, grid_shape: Tuple[int, int], model_path: Path):
        super().__init__(grid_shape, model_path)
        self.iters = int(getattr(self.cls, "ITERS", 8000))
        self.instances = int(getattr(self.cls, "INSTANCES", 16))
        self.backend_name, self._backend = _resolve_anneal_backend()

        self.model_info.iterations = self.iters
        self.model_info.instances = self.instances
        self.model_info.backend = self.backend_name

        self._plan: list[Action] = []
        self._ptr = 0
        self._expected = np.zeros(grid_shape, dtype=np.int8)   # 계획이 예상하는 다음 판

    def _solve(self, grid: NDArray[np.int8]) -> list[Action]:
        """서로 다른 난수로 instances 번 풀고 가장 좋은 수순을 고른다."""
        best_score, best_seq = -1, []
        for j in range(self.instances):
            score, seq = self._backend(grid, self.iters, j * 100 + 1)
            if score > best_score:
                best_score, best_seq = score, seq
        return [Action(top_left=(r1, c1), bottom_right=(r2, c2)) for r1, c1, r2, c2 in best_seq]

    def reset(self, board_source):
        board, score = super().reset(board_source)
        self._replan(self.board)
        return board, score

    def _replan(self, board: Board) -> None:
        self._plan = self._solve(board.grid)
        self._ptr = 0
        self._expected = board.grid.copy()

    def select_action(self, board: Board) -> Action | None:
        """수순의 다음 수. 판이 계획과 어긋났으면 그 판에서 다시 푼다.

        어긋남을 **합법성이 아니라 판 자체로** 판단한다. 다음 계획 수가 어쩌다 아직
        합법일 수는 있지만, 그 수순은 다른 판을 전제로 최적화된 것이라 그대로 이어
        두면 조용히 나쁜 수순을 따라가게 된다.
        """
        valid = board.get_valid_actions()
        if not valid:
            return None

        if self._peek() is None or not np.array_equal(board.grid, self._expected):
            self._replan(board)
            if self._peek() is None:
                # 어닐링이 합법수가 있는 판에서 수순을 못 냈다. 있을 수 없는 일이라
                # 조용히 넘기지 않는다 (백엔드가 판 크기를 잘못 다루는 경우 등).
                raise RuntimeError(
                    f"어닐링이 수순을 내지 못했습니다 (합법수 {len(valid)}개, "
                    f"backend={self.backend_name}).")

        action = self._peek()
        if action not in valid:
            raise RuntimeError(
                f"어닐링이 낸 수가 판에서 불법입니다: {action}\n"
                f"  (판은 계획과 같은데 수가 안 맞습니다 - 좌표 변환을 의심할 것)")

        self._ptr += 1
        self._expected = _without(self._expected, action)
        return action

    def _peek(self) -> Action | None:
        return self._plan[self._ptr] if self._ptr < len(self._plan) else None
