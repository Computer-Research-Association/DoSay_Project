from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import pygame
from game.action import Action

Color = Tuple[int, int, int]


THEME = {
    "BG": (205, 217, 187), 

    "BOARD_FILL": (214, 225, 197),  # 보드(사과 영역) 채움
    "BOARD_EDGE": (168, 184, 146),  # 보드 테두리

    "APPLE_TOP": (234, 112, 80),    # 사과 위쪽
    "APPLE_BOTTOM": (222, 94, 64),  # 사과 아래쪽
    "APPLE_EDGE": (198, 78, 52),    # 사과 테두리(구분용)
    "APPLE_TEXT": (252, 250, 246),  # 사과 값

    "PILL_BG": (245, 247, 238),     # 정보 배경
    "PILL_LABEL": (120, 130, 108),  # 라벨
    "PILL_VALUE": (64, 72, 54),     # 라벨 값

    "HIGHLIGHT": (255, 255, 255, 160),       # 마지막 수 강조 외각선
    "HIGHLIGHT_FILL": (255, 255, 255, 90),   # 강조 배경

    "CELL_SIZE": 46,       # 셀 한 변(px)
    "APPLE_PAD": 4,        # 셀 안쪽 사과 여백(px)
    "MARGIN": 22,          # 창 바깥 여백(px)
    "HEADER_H": 66,        # 상단 정보 영역 높이(px)
    "BOARD_PAD": 9,        # 보드 테두리와 사과 사이 여백(px)
    "BOARD_RADIUS": 14,    # 보드 모서리 둥글기(px)
    "PILL_RADIUS": 16      # 알약 둥글기(px)
}
THEME["MARGIN"]

class GUI:
    def __init__(self, rows: int, cols: int, *, title: str = "Apple Game") -> None:
        self.rows = rows
        self.cols = cols
        self._alive = True

        
        self._panel_x = THEME["MARGIN"]
        self._panel_y = THEME["HEADER_H"]
        self._panel_w = cols * THEME["CELL_SIZE"] + THEME["BOARD_PAD"] * 2
        self._panel_h = rows * THEME["CELL_SIZE"] + THEME["BOARD_PAD"] * 2
        self._board_x = self._panel_x + THEME["BOARD_PAD"]
        self._board_y = self._panel_y + THEME["BOARD_PAD"]
        self.width = self._panel_w + THEME["MARGIN"] * 2
        self.height = self._panel_y + self._panel_h + THEME["MARGIN"]

        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((self.width, self.height))

        self._font_label = self._load_font(17, bold=False)
        self._font_value = self._load_font(23, bold=True)
        self._font_apple = self._load_font(int(THEME["CELL_SIZE"] * 0.46), bold=True)

        self._max_digits = len(str(max(1, rows * cols)))
        digit_w = max(self._font_value.size(str(d))[0] for d in range(10))
        self._value_field_w = digit_w * self._max_digits

        self._apple_tiles: dict[int, pygame.Surface] = { v: self._build_apple_tile(v) for v in range(1, 10) }
        self._value_cache: dict[str, tuple[str, pygame.Surface]] = {}

    @staticmethod
    def _load_font(size: int, *, bold: bool) -> pygame.font.Font:
        path = pygame.font.match_font("segoeui,arial,dejavusans", bold=bold)
        if path:
            return pygame.font.Font(path, size)
        return pygame.font.SysFont("arial", size, bold=bold)

    def _build_apple_tile(self, value: int) -> pygame.Surface:
        
        size = THEME["CELL_SIZE"]
        s = 4
        big = size * s
        surf = pygame.Surface((big, big), pygame.SRCALPHA)

        cx = cy = big // 2
        radius = (size // 2 - THEME["APPLE_PAD"]) * s

        for dy in range(-radius, radius + 1):
            frac = (dy + radius) / (2 * radius)
            color = (
                round(THEME["APPLE_TOP"][0] + (THEME["APPLE_BOTTOM"][0] - THEME["APPLE_TOP"][0]) * frac),
                round(THEME["APPLE_TOP"][1] + (THEME["APPLE_BOTTOM"][1] - THEME["APPLE_TOP"][1]) * frac),
                round(THEME["APPLE_TOP"][2] + (THEME["APPLE_BOTTOM"][2] - THEME["APPLE_TOP"][2]) * frac),
            )
            half = int((radius * radius - dy * dy) ** 0.5)
            pygame.draw.line(surf, color, (cx - half, cy + dy), (cx + half, cy + dy))

        pygame.draw.circle(surf, THEME["APPLE_EDGE"], (cx, cy), radius, width=2 * s)

        surf = pygame.transform.smoothscale(surf, (size, size))
        text = self._font_apple.render(str(value), True, THEME["APPLE_TEXT"])
        surf.blit(text, text.get_rect(center=(size // 2, size // 2)))
        return surf

    def is_alive(self) -> bool:
        return self._alive

    def render(self, grid: Sequence[Sequence[int]] | Iterable[Iterable[int]], *, score: int = 0, step: int = 0, remaining: int = 0, action: Optional[Action] = None) -> bool:
        """한 프레임을 그린다. 창이 닫혔으면 False 를 반환.

        grid      : 2D int 격자 (ndarray / list 등 순회 가능하면 됨)
        score     : 점수(지운 사과 수 등)
        step      : 진행한 수(move) 번호
        remaining : 남은 사과 수 (GUI 는 계산하지 않고 받은 값을 그대로 표시)
        action    : 이번에 둘 수(Action) — 해당 사각형을 강조. 없으면 None
        """
        if not self._alive:
            return False

        self._pump_events()
        if not self._alive:
            return False

        self.screen.fill(THEME["BG"])
        self._draw_board_panel()
        self._draw_hud(score, step, remaining)
        self._draw_apples(grid)
        if action is not None:
            self._draw_highlight(action)

        pygame.display.flip()
        return True

    def close(self) -> None:
        if pygame.get_init():
            pygame.quit()
        self._alive = False

    def _pump_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._alive = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._alive = False

    def _draw_board_panel(self) -> None:
        
        rect = pygame.Rect(self._panel_x, self._panel_y, self._panel_w, self._panel_h)
        pygame.draw.rect(self.screen, THEME["BOARD_FILL"], rect, border_radius=THEME["BOARD_RADIUS"])
        pygame.draw.rect(self.screen, THEME["BOARD_EDGE"], rect, width=2, border_radius=THEME["BOARD_RADIUS"])

    def _draw_hud(self, score: int, step: int, remaining: int) -> None:
        
        cy = THEME["HEADER_H"] // 2

        x = THEME["MARGIN"]
        for label, value in [("STEP", step), ("APPLES", remaining)]:
            x = self._draw_pill(x, cy, label, str(value), anchor="left") + 12
        self._draw_pill(self.width - THEME["MARGIN"], cy, "SCORE", str(score), anchor="right")

    def _draw_pill(self, x: int, cy: int, label: str, value: str, *, anchor: str) -> int:
        
        pad_x, gap, h = 16, 10, 38
        lab = self._font_label.render(label.title(), True, THEME["PILL_LABEL"])
        w = pad_x * 2 + lab.get_width() + gap + self._value_field_w

        left = x if anchor == "left" else x - w
        rect = pygame.Rect(left, cy - h // 2, w, h)
        pygame.draw.rect(self.screen, THEME["PILL_BG"], rect, border_radius=THEME["PILL_RADIUS"])

        tx = left + pad_x
        self.screen.blit(lab, (tx, cy - lab.get_height() // 2))

        val = self._value_text(label, value)
        field_right = tx + lab.get_width() + gap + self._value_field_w
        self.screen.blit(val, (field_right - val.get_width(), cy - val.get_height() // 2))  # 우측 정렬
        return rect.right

    def _value_text(self, key: str, value: str) -> pygame.Surface:
        cached = self._value_cache.get(key)
        if cached is None or cached[0] != value:
            surf = self._font_value.render(value, True, THEME["PILL_VALUE"])
            self._value_cache[key] = (value, surf)
            return surf
        return cached[1]

    def _draw_apples(self, grid: Iterable[Iterable[int]]) -> None:
        
        x0, y0 = self._board_x, self._board_y
        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                v = int(value)
                if v <= 0:
                    continue  # 0(빈 칸)은 그리지 않음
                tile = self._apple_tiles.get(v)
                if tile is not None:
                    self.screen.blit(tile, (x0 + c * THEME["CELL_SIZE"], y0 + r * THEME["CELL_SIZE"]))

    def _draw_highlight(self, action: Action) -> None:
        
        (r1, c1), (r2, c2) = action.top_left, action.bottom_right
        x = self._board_x + c1 * THEME["CELL_SIZE"] - 2
        y = self._board_y + r1 * THEME["CELL_SIZE"] - 2
        w = (c2 - c1 + 1) * THEME["CELL_SIZE"] + 4
        h = (r2 - r1 + 1) * THEME["CELL_SIZE"] + 4

        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(overlay, THEME["HIGHLIGHT_FILL"], overlay.get_rect(), border_radius=10)
        pygame.draw.rect(overlay, THEME["HIGHLIGHT"], overlay.get_rect(), width=3, border_radius=10)
        self.screen.blit(overlay, (x, y))