"""
PUCT MCTS (AlphaZero) — 단일 플레이어, 중간보상(제거 칸) 처리.
- value net v(s) = s에서 도달 가능한 '추가 제거 칸수' / 162 (∈[0,1]).
- 한 수 a(즉시 k칸 제거)의 가치 = k/162 + v(s').  → 백업 시 경로의 보상 누적.
- leaf 평가 = net value (랜덤 rollout 없음).
- 개선정책 π = 루트 방문수 분포.
"""
import numpy as np
from dl.az.game import legal_moves, apply_move, TOTAL


class Node:
    __slots__ = ("grid", "moves", "expanded", "P", "N", "W", "reward", "children", "value")

    def __init__(self, grid):
        self.grid = grid
        self.moves = legal_moves(grid)
        self.expanded = False

    def expand(self, net, device, noise=False, alpha=0.3, eps=0.25):
        p, v = net.infer(self.grid, self.moves, device)
        M = len(self.moves)
        if M and noise:
            d = np.random.dirichlet([alpha] * M)
            p = (1 - eps) * p + eps * d
        self.P = p
        self.N = np.zeros(M)
        self.W = np.zeros(M)
        self.reward = np.array(
            [int(np.count_nonzero(self.grid[r1:r2 + 1, c1:c2 + 1]))
             for (r1, c1, r2, c2) in self.moves], dtype=np.float64)
        self.children = [None] * M
        self.value = v if M else 0.0
        self.expanded = True
        return self.value


def _select(node, c_puct):
    sumN = node.N.sum()
    # 미방문 수의 Q = 부모 value (FPU) → value가 ~0.5여도 탐색이 퍼짐
    Q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), node.value)
    U = c_puct * node.P * np.sqrt(1.0 + sumN) / (1.0 + node.N)
    return int(np.argmax(Q + U))


def run_mcts(root_grid, net, device, n_sims, c_puct=0.8, add_noise=True):
    root = Node(root_grid.copy())
    root.expand(net, device, noise=add_noise)
    for _ in range(n_sims):
        node = root
        path = []
        while True:
            if len(node.moves) == 0:              # 종료 상태
                leaf_value = 0.0
                break
            i = _select(node, c_puct)
            path.append((node, i))
            if node.children[i] is None:
                cg = node.grid.copy()
                apply_move(cg, node.moves[i])
                child = Node(cg)
                node.children[i] = child
                leaf_value = child.expand(net, device)   # 새 leaf 확장 → 평가
                break
            node = node.children[i]
        G = leaf_value                             # 백업: 경로 보상 누적
        for (nd, i) in reversed(path):
            G = nd.reward[i] / TOTAL + G
            nd.W[i] += G
            nd.N[i] += 1
    pi = root.N / max(root.N.sum(), 1e-9)
    return root, pi
