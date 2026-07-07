from dataclasses import dataclass

@dataclass(frozen=True)
class Action:
    # (row, col)
    top_left: tuple[int, int]
    bottom_right: tuple[int, int]
def get_cleared_cells(self, board) -> list[tuple[int, int]]:
    """이 action으로 지워지는(0이 아닌) 칸들의 좌표 리스트 반환"""
    (r1, c1), (r2, c2) = self.top_left, self.bottom_right
    area = board[r1:r2+1, c1:c2+1]
    return [(r1 + i, c1 + j)
            for i in range(area.shape[0])
            for j in range(area.shape[1])
            if area[i, j] != 0]