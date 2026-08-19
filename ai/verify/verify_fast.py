"""속도 손잡이가 **결과를 바꾸지 않는지** 대조한다 (2026-08-18).

    python ai/verify/verify_fast.py                 # CPU, 몇 초
    python ai/verify/verify_fast.py --device cuda

`verify_search.py` 는 "수순이 엔진에서 합법인가" 를 본다. 이 파일은 그 앞 단계 —
**최적화 전후의 계산이 같은가** 를 본다. 둘은 잡는 버그가 다르다.

검사하는 것.

    1. legal_mask 가 옛 구현 및 `game.Board` 와 같은가             (완전 일치여야 함)
    2. observation 이 옛 구현과 같은가 (마스크 물려주기 포함)      (완전 일치여야 함)
    3. 비트팩 중복 제거 == np.unique(axis=0)                     (집합·대표 모두 일치)
    4. legal_mask_chunked == legal_mask_from_grid                (완전 일치여야 함)

**옛 구현을 여기에 참조본으로 들고 있는 이유**: model.py 에 죽은 코드를 남기지
않으면서도 "무엇과 비교해 같다고 말하는가" 를 영구히 박아 두기 위해서다.
`RectIndex` 의 규칙이 바뀌면 이 파일도 같이 바뀌어야 하고, 그때 사람이 그 사실을
반드시 마주치게 된다.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ai.envs.action_sets import get_all_action
from ai.models.DQN.V15.model import (DENSITY_LOG_SCALE, N_DIGITS, RectIndex,
                                     TARGET_SUM, VALID_ACTION_SCALE, prefix_sum)
from ai.training import resolve_device
from game.board import Board

ROWS, COLS = 9, 18


# ──────────────────────────────────────────────────────────────────────────
# 참조본 — 2026-08-18 이전의 구현 그대로. 고치지 말 것.
# ──────────────────────────────────────────────────────────────────────────
def legal_mask_reference(index: RectIndex, grids: torch.Tensor) -> torch.Tensor:
    """옛 구현: 2차원 prefix 만으로 다섯 영역을 잰다 (사각형당 20회 읽기)."""
    prefix = prefix_sum(grids)
    area = index._area
    total = area(prefix, index.r_lo, index.c_lo, index.r_hi, index.c_hi)
    top = area(prefix, index.r_lo, index.c_lo, index.r_lo + 1, index.c_hi)
    bottom = area(prefix, index.r_hi - 1, index.c_lo, index.r_hi, index.c_hi)
    left = area(prefix, index.r_lo, index.c_lo, index.r_hi, index.c_lo + 1)
    right = area(prefix, index.r_lo, index.c_hi - 1, index.r_hi, index.c_hi)
    return (((total - TARGET_SUM).abs() < 0.5)
            & (top > 0) & (bottom > 0) & (left > 0) & (right > 0))


def observation_reference(index: RectIndex, grids: torch.Tensor) -> torch.Tensor:
    """옛 구현: 원핫을 (M, 10, R, C) 비교 텐서로 만들고, 밀도에 where 를 썼다."""
    m, rows, cols = grids.shape[0], index.rows, index.cols
    weight = index.legal_mask_from_grid(grids).to(grids.dtype)

    diff = grids.new_zeros(m, (rows + 1) * (cols + 1))
    diff.index_add_(1, index.corner_tl, weight)
    diff.index_add_(1, index.corner_tr, -weight)
    diff.index_add_(1, index.corner_bl, -weight)
    diff.index_add_(1, index.corner_br, weight)
    density = diff.view(m, rows + 1, cols + 1).cumsum(1).cumsum(2)[:, :rows, :cols]

    digits = torch.arange(N_DIGITS, device=grids.device, dtype=grids.dtype)
    obs = grids.new_zeros(m, N_DIGITS + 3, rows, cols)
    obs[:, :N_DIGITS] = (grids.round()[:, None] == digits[None, :, None, None]).to(grids.dtype)

    peak = density.amax(dim=(1, 2), keepdim=True)
    obs[:, N_DIGITS] = torch.where(peak > 0, density / peak.clamp(min=1.0),
                                   torch.zeros_like(density))
    obs[:, N_DIGITS + 1] = (torch.log1p(density) / DENSITY_LOG_SCALE).clamp(max=1.0)
    obs[:, N_DIGITS + 2] = (weight.sum(dim=1) / VALID_ACTION_SCALE).clamp(max=1.0)[:, None, None]
    return obs


def dedup_reference(children: torch.Tensor) -> torch.Tensor:
    """옛 구현: CPU 로 내려서 np.unique(axis=0). 대표는 그룹 안 최소 원본 인덱스."""
    _, uniq = np.unique(children.reshape(children.shape[0], -1).cpu().numpy(),
                        axis=0, return_index=True)
    return torch.as_tensor(np.sort(uniq), device=children.device)


# ──────────────────────────────────────────────────────────────────────────
# 판 만들기 — 빈 판부터 거의 다 지운 판까지 골고루
# ──────────────────────────────────────────────────────────────────────────
def sample_children(index: RectIndex, device: str, seed: int = 1234,
                    width: int = 48, cap: int = 600) -> torch.Tensor:
    """빔 한 스텝을 흉내낸다 — **뿌리 하나**에서 갈라진 자식들.

    깊이 2까지 펼치면 서로 다른 수순이 같은 판에 도달하는 중복이 자연히 생긴다.
    `dedup_indices` 가 노리는 것이 정확히 그 중복이고, 그 함수는 "뿌리가 같다" 는
    전제 위에서만 성립하므로 대조도 그 조건에서 해야 한다.
    """
    grid = torch.as_tensor(Board.from_seed((ROWS, COLS), seed).grid,
                           dtype=torch.float32, device=device)
    acts = index.legal_mask_from_grid(grid[None])[0].nonzero(as_tuple=True)[0][:width]
    level1 = index.erase(grid[None].expand(acts.numel(), -1, -1), acts)
    state_idx, action_idx = index.legal_mask_from_grid(level1).nonzero(as_tuple=True)
    level2 = index.erase(level1[state_idx[:cap]], action_idx[:cap])
    return torch.cat([level1, level2])


def sample_grids(device: str, n_boards: int = 24) -> torch.Tensor:
    """실제 대국에서 나오는 분포로 판을 모은다 (무작위 합법수를 두어 가며)."""
    grids, rng = [], np.random.default_rng(0)
    for i in range(n_boards):
        board = Board.from_seed((ROWS, COLS), 1234 + i)
        grids.append(board.grid.copy())
        steps = int(rng.integers(0, 55))
        for _ in range(steps):
            acts = board.get_valid_actions()
            if not acts:
                break
            board.do_action(acts[int(rng.integers(len(acts)))])
            grids.append(board.grid.copy())
    return torch.as_tensor(np.stack(grids), dtype=torch.float32, device=device)


def report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'O' if ok else 'X'}  {name}{('   ' + detail) if detail else ''}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--boards", type=int, default=24)
    args = parser.parse_args()

    device = resolve_device(args.device)
    index = RectIndex(get_all_action(ROWS, COLS), ROWS, COLS).to(device)
    grids = sample_grids(device, args.boards)
    print(f"장치 {device}   판 {grids.shape[0]}개   행동 {int(index.r_lo.numel())}개\n")

    ok = True

    # 1. legal_mask 12회 읽기 == 20회 읽기
    want = legal_mask_reference(index, grids)
    got = index.legal_mask_from_grid(grids)
    same = bool(torch.equal(want, got))
    ok &= report("legal_mask  12회 읽기 == 20회 읽기", same,
                 f"불일치 {int((want != got).sum())}개, 합법수 평균 "
                 f"{float(want.sum(1).float().mean()):.1f}개")

    # 1b. 엔진(game.Board)과도 대조한다 — 참조본 자체가 틀렸을 수 있다
    engine_ok = True
    for i in range(min(8, grids.shape[0])):
        board = Board.from_board(grids[i].cpu().numpy().astype(np.int8))
        want_set = {(a.top_left, a.bottom_right) for a in board.get_valid_actions()}
        idx = got[i].nonzero(as_tuple=True)[0].cpu().tolist()
        got_set = {((int(index.r_lo[a]), int(index.c_lo[a])),
                    (int(index.r_hi[a]) - 1, int(index.c_hi[a]) - 1)) for a in idx}
        engine_ok &= (want_set == got_set)
    ok &= report("legal_mask  == game.Board.get_valid_actions()", engine_ok)

    # 2. observation — 마스크를 미리 넘겨도, 구현을 바꿔도 같은가
    same = bool(torch.equal(index.observation(grids), index.observation(grids, got)))
    ok &= report("observation(grids, mask) == observation(grids)", same)

    want_obs = observation_reference(index, grids)
    got_obs = index.observation(grids, got)
    same = bool(torch.equal(want_obs, got_obs))
    ok &= report("observation == 옛 구현 (원핫 scatter · where 제거)", same,
                 f"최대 오차 {float((want_obs - got_obs).abs().max()):.3g}")

    # 3. 비트팩 중복 제거 == np.unique(axis=0)
    #    뿌리가 같은 자식들에서만 대조한다 (dedup_indices 의 전제. docstring 참고)
    dedup_ok, detail = True, ""
    for seed in (1234, 1235, 1236):
        kids = sample_children(index, device, seed)
        want_sel = dedup_reference(kids)
        got_sel = torch.sort(index.dedup_indices(kids)).values
        same = bool(torch.equal(want_sel.cpu(), got_sel.cpu()))
        dedup_ok &= same
        detail = f"{kids.shape[0]}개 -> {got_sel.numel()}개 (중복 {kids.shape[0]-got_sel.numel()}개)"
    ok &= report("비트팩 중복 제거 == np.unique(axis=0)", dedup_ok, detail)

    # 4. 청크로 잘라 계산해도 같은가 (VRAM 봉우리를 자르는 경로)
    same = bool(torch.equal(got, index.legal_mask_chunked(grids, chunk=7)))
    ok &= report("legal_mask_chunked == legal_mask_from_grid", same)

    print()
    if not ok:
        print("불일치가 있다. 속도 손잡이를 켜면 안 된다.")
        return 1
    print("전부 통과. 최적화가 계산 결과를 바꾸지 않는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
