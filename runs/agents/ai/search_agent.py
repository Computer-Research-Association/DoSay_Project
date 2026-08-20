"""추론 시점 탐색(국소탐색 / NRPA)으로 두는 에이전트.

`AIAgent` 와 다른 점은 **한 수씩 묻지 않는다**는 것이다. 국소탐색은 수순 하나를
통째로 놓고 일부를 부순 뒤 다시 짓기를 반복해 더 나은 수순을 찾는다. 그래서 판을
받으면 한 번에 끝까지 풀고, 그 수순을 하나씩 재생한다 (알고리즘 트랙의
AnnealExecutor 와 같은 모양이다).

판에 무작위성이 없으므로 미리 푸는 것(open-loop)과 매 수 다시 푸는 것(closed-loop)의
결과가 같다 — 문서(search-history.md 2.1)가 빔에 대해 적어 둔 것과 같은 이유다.
다만 판이 밖에서 바뀌면(실 기기 인식 오차) 수순이 어긋나므로 그때는 다시 푼다.

    python runs/play_device.py --preset cut-all --budget 36 --fp16 --value-cache

프리셋 표는 runs/agents/ai/presets.py 에 있고 runs/search_bench.py 와 공유한다.
"""

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from agents.base import Agent
from agents.dtos import SearchInfo
from ai.search import localsearch, nrpa
from game.action import Action
from game.board import Board

from .model_loader import load_model
from .presets import actions_to_moves, build, preset_labels
from .utils import ignore_logs

ENGINE_MODULES = {"ls": localsearch, "nrpa": nrpa}
ENGINE_NAMES = {"ls": "국소탐색", "nrpa": "NRPA"}


def _without(grid: np.ndarray, action: Action) -> np.ndarray:
    """그 수를 둔 뒤의 판. 계획이 예상하는 다음 판을 들고 있으려고 쓴다."""
    (r1, c1), (r2, c2) = action.top_left, action.bottom_right
    after = grid.copy()
    after[r1:r2 + 1, c1:c2 + 1] = 0
    return after


class SearchAgent(Agent):
    def __init__(self, grid_shape: Tuple[int, int], model_path: Path,
                 preset: str = "cut-all", budget: float = 36.0,
                 device: str = "cpu", fp16: bool = False, value_cache: bool = False,
                 prefilter: int = 0, eval_chunk: int = 0, seed: int = 0,
                 replan_budget: float | None = None) -> None:
        super().__init__(grid_shape, model_path)

        ignore_logs()
        self.grid_shape = grid_shape
        self.device = device
        self.seed = seed

        model, _env, model_info, _ = load_model(model_path, grid_shape, device=device)
        # q_net 은 DQN 계열에만 있고, plan() 은 그중 V15 계열에만 있다.
        # 둘 다 getattr 로 본다 - MaskablePPO 를 주면 q_net 접근에서 AttributeError 가
        # 나서 "왜 안 되는지" 대신 "무엇이 없는지" 만 보이게 된다.
        self.net = getattr(model.policy, "q_net", None)
        if self.net is None or not hasattr(self.net, "plan"):
            missing = "q_net" if self.net is None else "plan()"
            raise SystemExit(
                f"탐색은 V15 계열 체크포인트에서만 됩니다.\n"
                f"  {model_path.name} 에는 {missing} 이 없습니다.\n"
                "  ai/models/DQN/V15*/models/ 아래를 골라 주세요.")

        # 속도 손잡이. search_bench.py 와 같은 순서로 건다.
        if fp16:
            if device == "cpu":
                raise SystemExit("--fp16 은 cuda/mps 에서만 의미가 있습니다.")
            self.net.autocast_dtype = torch.float16
        self.net.prefilter_mult = prefilter
        if eval_chunk:
            self.net.eval_chunk = eval_chunk

        self.preset = preset
        self.engine, self.cfg, self.label = build(preset, budget, self.net,
                                                  value_cache=value_cache)
        self.module = ENGINE_MODULES[self.engine]
        self.budget = float(self.cfg.budget_sec)
        # 판이 어긋나 다시 풀 때의 예산. 기본은 처음과 같다. 실기기에서 예산이
        # 크면(36초) 판 도중에 그만큼 멈추므로, 짧게 줄이고 싶을 때 쓴다.
        self.replan_budget = self.budget if replan_budget is None else float(replan_budget)

        self.info = SearchInfo(
            model_type="AI + Search",
            model_name=f"{ENGINE_NAMES[self.engine]}/{preset}",
            agent_version=model_info.agent_version,
            source_path=model_info.source_path,
            engine=ENGINE_NAMES[self.engine],
            preset=preset,
            preset_label=preset_labels().get(preset, preset),
            budget_sec=self.budget,
            device=device,
            knobs=(f"fp16 {'O' if fp16 else 'X'}, "
                   f"2단평가 {prefilter or 'X'}, "
                   f"값캐시 {'O' if value_cache else 'X'}, "
                   f"eval_chunk {self.net.eval_chunk}"),
            total_train_steps=model_info.total_train_steps,
        )

        self._plan: list[Action] = []
        self._ptr = 0
        self._expected = np.zeros(grid_shape, dtype=np.int8)   # 계획이 예상하는 다음 판
        self._last_score = 0
        self._last_init_score = 0
        self.plan_count = 0

    def get_info(self) -> SearchInfo:
        return self.info

    # 마지막으로 푼 결과. 부르는 쪽이 "몇 점짜리 수순을 들고 시작하는지" 를 보여줄 때 쓴다.

    @property
    def plan_length(self) -> int:
        return len(self._plan)

    @property
    def last_score(self) -> int:
        """탐색이 낸 수순의 점수 (엔진 재생 전 주장값)."""
        return self._last_score

    @property
    def last_init_score(self) -> int:
        """국소탐색 전, 첫 빔만 썼을 때의 점수. 탐색이 얼마나 벌었는지 보려고."""
        return self._last_init_score

    # ── 판 하나를 통째로 풀기 ────────────────────────────────────────────

    def solve(self, grid: np.ndarray, budget: float | None = None):
        """판 하나를 풀어 수순을 돌려준다. (수순, 탐색결과)"""
        tensor = torch.as_tensor(np.asarray(grid), dtype=torch.float32, device=self.device)
        overrides = {"seed": self.seed}
        if budget is not None:
            overrides["budget_sec"] = budget
        cfg = type(self.cfg)(**{**self.cfg.__dict__, **overrides})

        result = self.module.solve(self.net, tensor, cfg)
        self.plan_count += 1
        self._last_score = result.score
        self._last_init_score = result.init_score
        moves = actions_to_moves(self.net.index, result.actions)
        return [Action((r1, c1), (r2, c2)) for r1, c1, r2, c2 in moves], result

    def plan_for(self, board: Board, budget: float | None = None) -> None:
        self._plan, _ = self.solve(board.grid, budget)
        self._ptr = 0
        self._expected = board.grid.copy()

    # ── 한 수씩 ──────────────────────────────────────────────────────────

    def select_action(self, board: Board) -> Action | None:
        """수순의 다음 수. 판이 계획과 어긋났으면 그 판에서 다시 푼다.

        어긋남을 **합법성이 아니라 판 자체로** 판단한다. 다음 계획 수가 어쩌다
        아직 합법일 수는 있지만, 그 수순은 다른 판을 전제로 최적화된 것이라
        그대로 이어 두면 조용히 나쁜 수순을 따라가게 된다.

        _ptr 을 여기서 올리고 다음에 보게 될 판을 미리 계산해 둔다. 부르는 쪽이
        돌려받은 수를 두지 않으면 다음 호출에서 판이 안 맞는 것이 드러난다.
        """
        valid = board.get_valid_actions()
        if not valid:
            return None

        if self._peek() is None or not np.array_equal(board.grid, self._expected):
            self.plan_for(board, self.replan_budget if self._plan else None)
            if self._peek() is None:
                raise RuntimeError(
                    f"탐색이 수순을 내지 못했습니다 (합법수 {len(valid)}개, "
                    f"preset={self.preset}).")

        action = self._peek()
        if action not in valid:
            raise RuntimeError(
                f"탐색이 낸 수가 판에서 불법입니다: {action}\n"
                f"  (판은 계획과 같은데 수가 안 맞습니다 - 좌표 변환을 의심할 것)")

        self._ptr += 1
        self._expected = _without(self._expected, action)
        return action

    def _peek(self) -> Action | None:
        return self._plan[self._ptr] if self._ptr < len(self._plan) else None

    # ── 한 판 ────────────────────────────────────────────────────────────

    def run_episode(self, board_source, render, delay):
        board = (Board.from_seed(self.grid_shape, board_source)
                 if board_source is None or isinstance(board_source, int)
                 else Board.from_board(np.asarray(board_source, dtype=np.int8)))

        self.plan_for(board)
        steps, score = 0, 0
        for action in self._plan:
            ok, cleared = board.do_action(action)
            if not ok:
                # 탐색이 낸 수순은 엔진에서 그대로 성립해야 한다. 안 되면 좌표
                # 변환이나 합법성 규칙이 어긋난 것이므로 조용히 넘기면 안 된다.
                raise RuntimeError(
                    f"탐색이 낸 {steps}번째 수가 엔진에서 불법입니다: {action}")
            steps += 1
            score += cleared
            if render:
                print(f"\n[step {steps:>3}] score={score}")
                if delay > 0: time.sleep(delay)

        if score != self._last_score:
            raise RuntimeError(
                f"탐색이 주장한 점수와 엔진 재생 점수가 다릅니다: "
                f"{self._last_score} vs {score}")
        return steps, score
