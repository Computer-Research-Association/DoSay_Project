"""학습된 가치함수로 하는 빔 탐색.

V6.0 학습 로그가 방향을 정해 주었다. explained_variance 가 0.975 — 가치망은
남은 점수를 거의 정확히 맞힌다. 반면 정책은 엔트로피가 0.79(유효 2.2수)까지
떨어져 사실상 결정적인데도 105점에서 멈춘다. 좋은 평가를 갖고도 행동으로
옮기지 못하는 상태다.

탐색은 그 간극을 직접 메운다. 정확한 V 가 있으면 수를 실제로 두어 보고
고르면 되기 때문이다. 손으로 짠 1수 앞 탐색(선택지 개수만 보는)이 이미 118점을
내므로, 0.975 짜리 V 로 더 깊게 보면 그 위를 노릴 수 있다.

경로 점수는

    (경로에서 먹은 사과 합)/162  +  V(잎) + Φ(잎)      단, 끝난 판이면 둘 다 0

이다. Φ 는 환경이 퍼텐셜 셰이핑(potential())을 쓸 때만 붙는 항이다. 셰이핑을
쓰면 V 가 학습하는 값이 V_shaped(s) = V_true(s) - Φ(s) 라, Φ 를 되더해야 참값이
된다. 셰이핑이 없는 버전에서는 Φ=0 이라 식이 그대로 (먹은 합 + V) 가 된다.

Φ 를 되더하는 것이 왜 중요한가: V7.0 은 explained_variance 0.983 인데도 탐색이
+2점밖에 못 냈고, log1p(남은 합법수) 라는 조잡한 대용값을 V 자리에 넣으면 +13점이
나왔다. EV 0.983 의 잔차가 약 1.7점인데 형제 상태 간 가치 차이가 딱 그 정도라,
V 혼자서는 형제를 일관되게 서열 매기지 못한다. Φ 는 절대값은 틀려도 서열이
일관돼서, V 가 잡음일 때 탐색이 최소한 그 대용값 수준으로는 동작하게 받쳐 준다.
"""

from dataclasses import dataclass

import numpy as np
import torch
from stable_baselines3.common.utils import obs_as_tensor

from game.board import Board


@dataclass(frozen=True)
class SearchConfig:
    top_k: int = 0        # 루트에서 시도할 후보 수 (0이면 탐색 안 함)
    depth: int = 1        # 앞을 몇 수 보는가
    beam_width: int = 1   # 각 깊이에서 남길 상태 수
    beam_top_k: int = 0   # 2수째부터 각 상태가 펼칠 후보 수 (0이면 top_k)

    @property
    def enabled(self) -> bool:
        return self.top_k > 0

    def expand_width(self, depth: int) -> int:
        return self.top_k if depth == 0 else (self.beam_top_k or self.top_k)

    def describe(self) -> str:
        if not self.enabled:
            return "off"
        if self.depth <= 1:
            return f"depth 1, top-{self.top_k}"
        return f"depth {self.depth}, beam {self.beam_width}, top-{self.top_k}/{self.beam_top_k or self.top_k}"


@dataclass
class _Node:
    """탐색 트리의 한 상태. first_action 은 루트에서 처음 둔 수."""
    board: Board
    reward: float          # 루트부터 여기까지 먹은 사과 / 전체 칸수
    first_action: int


class ValueBeamSearch:
    def __init__(self, model, env, config: SearchConfig) -> None:
        self.model = model
        self.env = env          # 버전 폴더의 env (관측 인코딩을 여기서 빌린다)
        self.config = config

    # ── 환경 빌려쓰기 ────────────────────────────────────────────────────
    # env 의 board 를 잠깐 갈아끼워 그 상태의 관측/마스크를 얻는다.
    # 탐색이 끝나면 원래 board 를 반드시 되돌려 놓는다.

    def _encode(self, board: Board) -> tuple[np.ndarray, np.ndarray]:
        self.env.board = board
        return self.env._get_obs(), self.env.get_action_mask()

    def _potential(self, board: Board) -> float:
        """셰이핑을 쓰는 환경이면 Φ(s), 아니면 0."""
        potential = getattr(self.env, "potential", None)
        if potential is None:
            return 0.0
        self.env.board = board
        return potential()

    def _values(self, observations: list[np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            tensor = obs_as_tensor(np.stack(observations), self.model.device)
            return self.model.policy.predict_values(tensor).squeeze(-1).cpu().numpy()

    def _policy_ranking(self, observations, masks, width: int) -> list[list[int]]:
        """각 상태에서 정책 확률 상위 width 개 행동 인덱스."""
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(
                obs_as_tensor(np.stack(observations), self.model.device),
                action_masks=np.stack(masks),
            )
            probs = distribution.distribution.probs  # type: ignore[union-attr]

        ranking = []
        for row, mask in zip(probs, masks):
            k = min(width, int(mask.sum()))
            ranking.append(torch.topk(row, k).indices.tolist())
        return ranking

    # ── 탐색 ─────────────────────────────────────────────────────────────

    def choose(self, root_board: Board) -> int:
        saved = root_board.grid.copy()
        try:
            return self._search(root_board)
        finally:
            self.env.board = Board.from_board(saved)  # 원래 판 복구

    def _search(self, root_board: Board) -> int:
        cell_count = self.env.total_cell_count
        frontier = [_Node(root_board, 0.0, -1)]
        best_action, best_score = -1, -np.inf

        for depth in range(self.config.depth):
            observations, masks = zip(*(self._encode(node.board) for node in frontier))
            ranking = self._policy_ranking(observations, masks, self.config.expand_width(depth))

            children: list[_Node] = []
            for node, actions in zip(frontier, ranking):
                for index in actions:
                    board = Board.from_board(node.board.grid)
                    _, removed = board.do_action(self.env.index_to_action[index])
                    children.append(_Node(
                        board=board,
                        reward=node.reward + removed / cell_count,
                        first_action=index if node.first_action < 0 else node.first_action,
                    ))
            if not children:
                break

            child_obs, _ = zip(*(self._encode(child.board) for child in children))
            values = self._values(list(child_obs))
            potentials = np.array([self._potential(child.board) for child in children])
            terminal = np.array([not child.board.get_valid_actions() for child in children])
            values[terminal] = 0.0       # 끝난 판의 남은 가치는 0
            potentials[terminal] = 0.0   # Φ(종료)=0 이지만 명시해 둔다

            scores = np.array([child.reward for child in children]) + values + potentials
            top = int(np.argmax(scores))
            if scores[top] > best_score:
                best_score, best_action = float(scores[top]), children[top].first_action

            # 다음 깊이로 넘길 상태 — 이미 끝난 판은 더 펼칠 것이 없다
            alive = [i for i in np.argsort(-scores) if not terminal[i]]
            frontier = [children[i] for i in alive[:self.config.beam_width]]
            if not frontier:
                break

        return best_action
