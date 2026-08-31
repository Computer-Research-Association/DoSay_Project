import pygame
import sys
import os
import json

# ---------------------------------------------------------
# 판 크기
# ---------------------------------------------------------
ROWS = 9
COLS = 18

# 사진 보드 (배포판 어닐링 161/162, 꿀판)
INITIAL_BOARD = [
    [2, 8, 3, 1, 5, 6, 7, 3, 3, 1, 6, 9, 1, 3, 2, 2, 6, 9],
    [3, 7, 7, 1, 8, 5, 1, 2, 6, 4, 6, 4, 5, 7, 5, 4, 3, 5],
    [1, 1, 7, 3, 9, 2, 8, 4, 9, 1, 4, 1, 4, 1, 9, 2, 9, 1],
    [9, 2, 8, 3, 5, 2, 4, 5, 5, 5, 7, 5, 5, 2, 6, 1, 5, 5],
    [7, 4, 5, 6, 2, 5, 5, 1, 6, 2, 8, 2, 7, 2, 4, 1, 6, 1],
    [3, 1, 1, 2, 4, 4, 8, 3, 1, 3, 2, 8, 3, 3, 6, 4, 6, 9],
    [8, 5, 9, 3, 4, 5, 2, 3, 2, 4, 1, 6, 2, 8, 7, 3, 6, 2],
    [7, 3, 6, 5, 7, 3, 2, 8, 1, 1, 2, 2, 1, 4, 5, 2, 1, 8],
    [5, 1, 4, 5, 6, 9, 5, 5, 7, 9, 8, 3, 7, 5, 8, 3, 3, 4],
]

# 시드로 판 생성: `python run.py --seed=409834` 또는 DOSAY_SEED 환경변수. 없으면 위 기본판.
_seed = None
for _a in sys.argv[1:]:
    if _a.startswith("--seed="):
        _seed = int(_a.split("=", 1)[1])
if _seed is None and os.environ.get("DOSAY_SEED"):
    _seed = int(os.environ["DOSAY_SEED"])
if _seed is not None:
    from models.board import make_board
    INITIAL_BOARD = [[int(v) for v in row] for row in make_board(_seed)]
    print(f"[run] seed {_seed} 판 로드")

TIME_LIMIT_SECONDS = 1200

# ---------------------------------------------------------
# 화면/셀 크기
# ---------------------------------------------------------
CELL_SIZE = 40
MARGIN_LEFT = 40
MARGIN_TOP = 80
CIRCLE_RADIUS = 17

WINDOW_WIDTH = MARGIN_LEFT * 2 + COLS * CELL_SIZE
WINDOW_HEIGHT = MARGIN_TOP + ROWS * CELL_SIZE + 40

BG_COLOR = (245, 245, 245)
CIRCLE_COLOR = (220, 60, 60)
TEXT_COLOR = (255, 255, 255)
SELECT_BOX_COLOR = (60, 120, 220)         # 드래그/AI = 파랑
SELECT_BOX_VALID_COLOR = (60, 180, 90)    # 유효 수 = 초록
SCORE_COLOR = (30, 30, 30)


def cell_to_pixel(row, col):
    return MARGIN_LEFT + col * CELL_SIZE, MARGIN_TOP + row * CELL_SIZE


def pixel_to_cell(x, y):
    col = min(max((x - MARGIN_LEFT) // CELL_SIZE, 0), COLS - 1)
    row = min(max((y - MARGIN_TOP) // CELL_SIZE, 0), ROWS - 1)
    return row, col


class AppleGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("사과게임 — 초록: 가능한 수 / 파랑: AI 추천")
        self.font = pygame.font.SysFont("applegothic", 18)
        self.big_font = pygame.font.SysFont("applegothic", 26, bold=True)
        self.clock = pygame.time.Clock()

        self.dragging = False
        self.drag_start = None
        self.drag_current = None
        self.show_hints = False   # 초록(가능한 수)은 기본 꺼둠, H로 토글
        self.solution = self._load_solution()   # 162 해답 수순
        self.guide_idx = 0

        self.reset_board()
        self.score = 0
        self.start_ticks = pygame.time.get_ticks()
        self.game_over = False

    # ---------- 보드 상태 ----------
    def _load_solution(self):
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution_moves.json")
            with open(path) as f:
                return [tuple(m) for m in json.load(f)]
        except Exception:
            return []

    def _move_done(self, mv):
        r1, c1, r2, c2 = mv
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if not self.removed[r][c]:
                    return False
        return True

    def advance_guide(self):
        """다 지워진 해답 수는 건너뛰고 다음 안내할 수로."""
        while self.guide_idx < len(self.solution) and self._move_done(self.solution[self.guide_idx]):
            self.guide_idx += 1

    def reset_board(self):
        self.board = [row[:] for row in INITIAL_BOARD]
        self.removed = [[False] * COLS for _ in range(ROWS)]
        self.guide_idx = 0
        self.recompute()

    def restart(self):
        self.reset_board()
        self.score = 0
        self.start_ticks = pygame.time.get_ticks()
        self.game_over = False
        self.dragging = False
        self.drag_start = None
        self.drag_current = None

    def _prefix(self):
        g = [[0 if self.removed[r][c] else self.board[r][c]
              for c in range(COLS)] for r in range(ROWS)]
        P = [[0] * (COLS + 1) for _ in range(ROWS + 1)]
        for r in range(ROWS):
            for c in range(COLS):
                P[r + 1][c + 1] = g[r][c] + P[r][c + 1] + P[r + 1][c] - P[r][c]
        return P

    def find_valid_moves(self, P=None):
        """합=10인 모든 사각형 (면적 오름차순)."""
        if P is None:
            P = self._prefix()
        moves = []
        for r1 in range(ROWS):
            for r2 in range(r1, ROWS):
                for c1 in range(COLS):
                    for c2 in range(c1, COLS):
                        s = P[r2 + 1][c2 + 1] - P[r1][c2 + 1] - P[r2 + 1][c1] + P[r1][c1]
                        if s == 10:
                            moves.append((r1, c1, r2, c2))
                        elif s > 10:
                            break
        moves.sort(key=lambda m: (m[2] - m[0] + 1) * (m[3] - m[1] + 1))
        return moves

    def _count_valid_after(self, mv):
        """mv를 뒀다고 가정했을 때 남는 유효 수 개수 (action-max 평가용)."""
        r1, c1, r2, c2 = mv
        changed = []
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if not self.removed[r][c]:
                    self.removed[r][c] = True
                    changed.append((r, c))
        n = len(self.find_valid_moves())
        for (r, c) in changed:
            self.removed[r][c] = False
        return n

    def recompute(self):
        """판이 바뀔 때마다: 유효 수 목록 갱신 + 해답 가이드 전진."""
        self.hint_moves = self.find_valid_moves()
        self.advance_guide()

    # ---------- 입력 ----------
    def handle_mouse_down(self, pos):
        if self.game_over:
            return
        self.dragging = True
        self.drag_start = pixel_to_cell(*pos)
        self.drag_current = self.drag_start

    def handle_mouse_motion(self, pos):
        if self.dragging:
            self.drag_current = pixel_to_cell(*pos)

    def handle_mouse_up(self, pos):
        if not self.dragging:
            return
        self.dragging = False
        self.drag_current = pixel_to_cell(*pos)
        r1, r2, c1, c2 = self.get_selection_bounds()   # (행min,행max,열min,열max)
        total, count = self.selection_sum(r1, r2, c1, c2)
        if total == 10 and count > 0:
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if not self.removed[r][c]:
                        self.removed[r][c] = True
                        self.score += 1
            self.recompute()
        self.drag_start = None
        self.drag_current = None

    def get_selection_bounds(self):
        r1, c1 = self.drag_start
        r2, c2 = self.drag_current
        return min(r1, r2), max(r1, r2), min(c1, c2), max(c1, c2)

    def selection_sum(self, r1, r2, c1, c2):
        total = count = 0
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if not self.removed[r][c]:
                    total += self.board[r][c]
                    count += 1
        return total, count

    def remaining_seconds(self):
        elapsed = (pygame.time.get_ticks() - self.start_ticks) / 1000
        return max(0, TIME_LIMIT_SECONDS - int(elapsed))

    # ---------- 그리기 ----------
    def draw_board(self):
        for r in range(ROWS):
            for c in range(COLS):
                if self.removed[r][c]:
                    continue
                x, y = cell_to_pixel(r, c)
                cx, cy = x + CELL_SIZE // 2, y + CELL_SIZE // 2
                pygame.draw.circle(self.screen, CIRCLE_COLOR, (cx, cy), CIRCLE_RADIUS)
                t = self.font.render(str(self.board[r][c]), True, TEXT_COLOR)
                self.screen.blit(t, t.get_rect(center=(cx, cy)))

    def _box(self, mv, color, width, inset=2):
        r1, c1, r2, c2 = mv
        x1, y1 = cell_to_pixel(r1, c1)
        rect = pygame.Rect(x1 + inset, y1 + inset,
                           (c2 - c1 + 1) * CELL_SIZE - 2 * inset,
                           (r2 - r1 + 1) * CELL_SIZE - 2 * inset)
        pygame.draw.rect(self.screen, color, rect, width=width)

    def draw_overlays(self):
        if self.show_hints:                              # 초록: 가능한 수 (최대 14개)
            for mv in self.hint_moves[:14]:
                self._box(mv, SELECT_BOX_VALID_COLOR, 2)
        if 0 <= self.guide_idx < len(self.solution):     # 보라: 다음 둘 수 (162 해답)
            self._box(self.solution[self.guide_idx], (150, 60, 220), 4, inset=0)

    def draw_selection(self):
        if not self.dragging or self.drag_start is None:
            return
        r1, r2, c1, c2 = self.get_selection_bounds()
        total, _ = self.selection_sum(r1, r2, c1, c2)
        x1, y1 = cell_to_pixel(r1, c1)
        rect = pygame.Rect(x1, y1, (c2 - c1 + 1) * CELL_SIZE, (r2 - r1 + 1) * CELL_SIZE)
        color = SELECT_BOX_VALID_COLOR if total == 10 else SELECT_BOX_COLOR
        pygame.draw.rect(self.screen, color, rect, width=3)

    def draw_hud(self):
        self.screen.blit(self.big_font.render(f"점수: {self.score}", True, SCORE_COLOR), (MARGIN_LEFT, 14))
        n = len(self.hint_moves)
        msg = f"가능한 수: {n}" if n else "막힘 (더 못 지움)"
        col = SELECT_BOX_VALID_COLOR if n else (200, 30, 30)
        self.screen.blit(self.big_font.render(msg, True, col), (WINDOW_WIDTH // 2 - 100, 14))
        self.screen.blit(self.big_font.render(f"시간: {self.remaining_seconds()}", True, SCORE_COLOR),
                         (WINDOW_WIDTH - 150, 14))
        done = self.guide_idx >= len(self.solution)
        gtxt = "해답 완주! (162)" if done else f"보라=다음 둘 수(해답)  {self.guide_idx + 1}/{len(self.solution)}수"
        self.screen.blit(self.font.render(gtxt + "   (H:가능한수 토글, R:재시작)", True,
                                          (150, 60, 220) if not done else (60, 180, 90)), (MARGIN_LEFT, 48))
        if self.game_over:
            over = self.big_font.render("게임 종료 - R키로 재시작", True, (200, 30, 30))
            self.screen.blit(over, over.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 18)))

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_down(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_mouse_motion(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_up(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.restart()
                    elif event.key == pygame.K_h:
                        self.show_hints = not self.show_hints

            if self.remaining_seconds() == 0:
                self.game_over = True

            self.screen.fill(BG_COLOR)
            self.draw_board()
            self.draw_overlays()
            self.draw_selection()
            self.draw_hud()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    AppleGame().run()
