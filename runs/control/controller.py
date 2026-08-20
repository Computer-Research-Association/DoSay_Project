"""사각형 드래그 조작.

    usage: input touchscreen swipe <x1> <y1> <x2> <y2> [duration(ms)]
           input motionevent DOWN|MOVE|UP <x> <y>

드래그가 scrcpy 에서 실시간으로 보이도록 두 가지 방식을 둔다.

  swipe  : duration(기본 300ms)을 주면 커서가 이동하는 것이 그대로 보인다. shell 1회라 빠르다.
  motion : DOWN -> MOVE*n -> UP 을 한 shell 명령으로 체이닝한다.
           swipe 를 무시하는(눌린 상태 유지가 필요한) 게임에 대비한 대안.
"""

from game.action import Action

from .device import device_call
from .vision import GridGeometry


class PROCESSOR:
    """격자 좌표(Action)를 화면 픽셀 드래그로 옮긴다."""

    def __init__(self, device, geom: GridGeometry,
                 mode: str = "swipe",
                 duration_ms: int = 300,
                 margin_ratio: float = 0.45,
                 motion_steps: int = 8):
        """
        device       : ppadb device
        geom         : vision.detect_grid() 결과
        mode         : "swipe" | "motion"
        duration_ms  : 드래그 한 번에 걸리는 시간. 두 모드에서 같은 뜻이다.
                       (크게 줄수록 scrcpy 에서 눈에 잘 보인다)
        margin_ratio : 선택 박스를 사과 중심에서 피치의 몇 배만큼 바깥으로 잡을지
        motion_steps : motion 모드에서 DOWN 과 UP 사이에 넣을 MOVE 개수
        """
        self.device = device
        self.geom = geom
        self.mode = mode
        self.duration = duration_ms
        self.margin_ratio = margin_ratio
        self.motion_steps = max(2, motion_steps)

    @property
    def motion_step_sleep(self) -> float:
        """motion 모드에서 MOVE 사이 간격.

        duration_ms 를 걸음 수로 나눈다. 이렇게 해야 --drag-ms 가 swipe 든
        motion 이든 '드래그에 걸리는 시간' 이라는 같은 뜻이 된다.
        (예전에는 0.03초 고정이라 motion 모드에서 --drag-ms 가 무시됐다.)
        """
        return self.duration / 1000.0 / self.motion_steps

    def rect_pixels(self, action: Action) -> tuple[int, int, int, int]:
        """액션이 덮는 셀들의 바깥쪽 픽셀 사각형.

        셀 경계에 딱 맞추면 반올림 오차 한 픽셀에 선택이 실패하므로
        피치의 margin_ratio 배만큼 바깥으로 벌려 잡는다.
        """
        (r1, c1), (r2, c2) = action.top_left, action.bottom_right
        g = self.geom
        x1, y1 = g.centers[r1, c1]
        x2, y2 = g.centers[r2, c2]
        sx = int(x1 - g.pitch_x * self.margin_ratio)
        sy = int(y1 - g.pitch_y * self.margin_ratio)
        ex = int(x2 + g.pitch_x * self.margin_ratio)
        ey = int(y2 + g.pitch_y * self.margin_ratio)
        return sx, sy, ex, ey

    def move(self, action: Action) -> None:
        sx, sy, ex, ey = self.rect_pixels(action)
        if self.mode == "motion":
            self._drag_motion(sx, sy, ex, ey)
        else:
            self._drag_swipe(sx, sy, ex, ey)

    def _drag_swipe(self, sx, sy, ex, ey) -> None:
        self._send(f"input touchscreen swipe {sx} {sy} {ex} {ey} {self.duration}")

    def _drag_motion(self, sx, sy, ex, ey) -> None:
        # 한 번의 shell 호출로 DOWN -> MOVE*n -> UP 체이닝 (라운드트립 1회)
        n = self.motion_steps
        gap = f"{self.motion_step_sleep:.4f}"
        parts = [f"input motionevent DOWN {sx} {sy}"]
        for i in range(1, n + 1):
            t = i / n
            mx = int(sx + (ex - sx) * t)
            my = int(sy + (ey - sy) * t)
            parts.append(f"sleep {gap}")
            parts.append(f"input motionevent MOVE {mx} {my}")
        parts.append(f"input motionevent UP {ex} {ey}")
        self._send(" && ".join(parts))

    def _send(self, cmd: str) -> None:
        device_call(self.device.shell, cmd)


class DryRunProcessor(PROCESSOR):
    """실제로 기기를 건드리지 않고 보낼 명령만 모은다.

    기기 없이 판정 루프만 확인하거나, 좌표가 맞는지 눈으로 보고 싶을 때 쓴다.
    """

    def __init__(self, geom: GridGeometry, **kwargs):
        super().__init__(device=None, geom=geom, **kwargs)
        self.commands: list[str] = []

    def _send(self, cmd: str) -> None:
        self.commands.append(cmd)
