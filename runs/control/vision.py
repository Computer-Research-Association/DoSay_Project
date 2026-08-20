"""화면 인식: ADB screencap 1회 -> 격자 기하 검출 -> 셀 단위 숫자 분류(템플릿 매칭).

비용을 아끼는 지점이 둘이다.

  1. 격자 기하(중심 좌표, 피치)는 시작할 때 한 번만 컨투어로 찾는다. 사과게임은
     클리어해도 사과가 이동하지 않으므로 좌표가 판 내내 그대로다.
  2. 숫자도 한 번만 읽는다. 값이 바뀌지 않고 사라지기만 하므로
     (초기 인식값) x (지금 사과가 있는가) 만으로 현재 판이 완전히 결정된다.
     그래서 매 수 재확인은 주황 커버리지 검사(resync_board)로 끝난다.
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .device import device_call

# 주황 사과 HSV 범위 (스크린샷 기준 검증됨)
ORANGE_LO = (0, 80, 120)
ORANGE_HI = (25, 255, 255)
GLYPH_SIZE = (24, 32)          # (w, h) 숫자 정규화 크기
MIN_APPLE_AREA_RATIO = 1e-4    # 화면 대비 최소 사과 면적 비율
WHITE_THRESH = 200             # 흰 숫자 이진화 임계값

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


@dataclass
class GridGeometry:
    centers: NDArray[np.float32]   # (rows, cols, 2) -> (x, y) 픽셀 중심 좌표
    pitch_x: float                 # 셀 간 가로 간격
    pitch_y: float                 # 셀 간 세로 간격
    radius: float                  # 사과 반지름(대략)
    rows: int
    cols: int


def capture(device) -> NDArray[np.uint8]:
    """ADB screencap 1회 -> BGR 이미지"""
    raw = device_call(device.screencap)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("screencap 디코딩 실패")
    return img


def _orange_mask(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, ORANGE_LO, ORANGE_HI)


def detect_grid(img: NDArray[np.uint8]) -> GridGeometry:
    """시작 화면에서 사과 격자 기하를 검출한다. (모든 사과가 있을 때 1회 호출)"""
    mask = _orange_mask(img)
    min_area = img.shape[0] * img.shape[1] * MIN_APPLE_AREA_RATIO
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (0.6 < w / h < 1.7):   # 원형(가로세로 비슷)만 사과로 인정
            continue
        cells.append((x + w / 2.0, y + h / 2.0, (w + h) / 4.0))

    if len(cells) < 20:
        raise RuntimeError(
            f"사과 검출 실패 (검출 수 {len(cells)}개). 게임 시작 화면인지 확인해 주세요.")

    # y 기준 행 클러스터링
    cells.sort(key=lambda t: t[1])
    radius = float(np.median([c[2] for c in cells]))
    row_gap = radius   # 반지름 이상 벌어지면 다른 행
    rows: list[list] = [[cells[0]]]
    for c in cells[1:]:
        if c[1] - rows[-1][-1][1] > row_gap:
            rows.append([c])
        else:
            rows[-1].append(c)

    n_cols = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == n_cols]   # 불완전 행 제거(UI 오검출 방지)
    n_rows = len(rows)
    for r in rows:
        r.sort(key=lambda t: t[0])

    centers = np.array([[(c[0], c[1]) for c in r] for r in rows], dtype=np.float32)
    pitch_x = float(np.median(np.diff(centers[:, :, 0], axis=1)))
    pitch_y = float(np.median(np.diff(centers[:, :, 1], axis=0)))

    return GridGeometry(centers=centers, pitch_x=pitch_x, pitch_y=pitch_y,
                        radius=radius, rows=n_rows, cols=n_cols)


class Templates:
    """숫자 템플릿을 정규화 벡터(zero-mean, unit-norm)로 미리 바꿔 두고,
    분류를 행렬-벡터 곱 1회로 처리한다 (피어슨 상관 = TM_CCOEFF_NORMED 와 동치)."""

    def __init__(self, mat: NDArray[np.float32], labels: NDArray[np.int_]):
        self.mat = mat          # (N, GLYPH_W*GLYPH_H) 정규화된 템플릿
        self.labels = labels    # (N,) 각 행의 숫자


def _normalize_vec(a: NDArray) -> NDArray[np.float32]:
    v = a.astype(np.float32).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def load_templates(template_dir: str = TEMPLATE_DIR) -> Templates:
    mats, labels = [], []
    for d in range(1, 10):
        p = os.path.join(template_dir, f"{d}.png")
        t = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if t is None:
            raise FileNotFoundError(f"템플릿 없음: {p}")
        mats.append(_normalize_vec(t))
        labels.append(d)
    return Templates(np.stack(mats), np.array(labels))


def _extract_glyph(cell_bgr: NDArray[np.uint8]) -> NDArray[np.uint8] | None:
    """셀 이미지에서 흰 숫자 글리프를 잘라 정규화. 숫자가 없으면 None."""
    g = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, WHITE_THRESH, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) >= 8]
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return cv2.resize(th[y0:y1, x0:x1], GLYPH_SIZE, interpolation=cv2.INTER_AREA)


def _classify(glyph: NDArray[np.uint8], templates: Templates) -> tuple[int, float]:
    scores = templates.mat @ _normalize_vec(glyph)
    i = int(np.argmax(scores))
    return int(templates.labels[i]), float(scores[i])


def read_board(img: NDArray[np.uint8], geom: GridGeometry, templates: Templates,
               min_confidence: float = 0.55) -> tuple[NDArray[np.int8], float]:
    """스크린샷 1장에서 전체 판을 읽는다.

    반환: (board[r,c] in 0..9, 최소 매칭 신뢰도). 사과가 없는 칸은 0.
    """
    omask = _orange_mask(img)
    half = int(geom.radius * 1.05)
    board = np.zeros((geom.rows, geom.cols), dtype=np.int8)
    min_conf = 1.0

    for r in range(geom.rows):
        for c in range(geom.cols):
            cx, cy = geom.centers[r, c]
            x0, y0 = int(cx - half), int(cy - half)
            x1, y1 = int(cx + half), int(cy + half)

            patch = omask[y0:y1, x0:x1]        # 사과 존재 여부: 중심 주변 주황 비율
            if patch.size == 0 or patch.mean() < 255 * 0.25:
                board[r, c] = 0
                continue

            glyph = _extract_glyph(img[y0:y1, x0:x1])
            if glyph is None:
                board[r, c] = 0
                continue

            d, s = _classify(glyph, templates)
            if s < min_confidence:
                raise RuntimeError(
                    f"숫자 인식 신뢰도가 낮습니다 ({r},{c}) score={s:.2f}.\n"
                    f"기기 해상도나 폰트가 다르면 그 기기 스크린샷으로 "
                    f"{TEMPLATE_DIR} 의 템플릿을 다시 만들어야 합니다.")
            board[r, c] = d
            min_conf = min(min_conf, s)

    return board, min_conf


def presence_grid(img: NDArray[np.uint8], geom: GridGeometry,
                  coverage: float = 0.25) -> NDArray[np.bool_]:
    """각 셀에 사과(주황)가 있는지 (rows, cols) bool 배열로.

    적분 영상으로 전 셀을 한 번에 검사한다 (셀 수와 무관하게 수 ms).
    """
    omask = _orange_mask(img)
    ii = cv2.integral((omask // 255).astype(np.uint8))   # (H+1, W+1) int32
    H, W = omask.shape

    half = int(geom.radius * 0.7)
    cx = geom.centers[:, :, 0].astype(np.int32)
    cy = geom.centers[:, :, 1].astype(np.int32)
    x0 = np.clip(cx - half, 0, W - 1); x1 = np.clip(cx + half, 1, W)
    y0 = np.clip(cy - half, 0, H - 1); y1 = np.clip(cy + half, 1, H)

    sums = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    area = (x1 - x0) * (y1 - y0)
    return (sums / np.maximum(area, 1)) > coverage


def resync_board(img: NDArray[np.uint8], geom: GridGeometry,
                 known_values: NDArray[np.int8]) -> NDArray[np.int8]:
    """숫자 분류 없이 현재 판을 복원한다.

    사과 값은 바뀌지 않고 사라지기만 하므로 (초기 인식값) x (존재 여부) 가 곧 현재 판이다.
    """
    return np.where(presence_grid(img, geom), known_values, 0).astype(np.int8)


def format_board(grid: NDArray[np.int8]) -> str:
    return "\n".join(" ".join("." if v == 0 else str(v) for v in line) for line in grid)
