"""실 기기 자동 플레이.

에이전트를 골라서 그대로 폰 화면을 조작한다.

    python runs/play_device.py                       # 골라서 플레이 (프리셋도 물어본다)
    python runs/play_device.py --dry-run             # 드래그 없이 인식/판정만 확인
    python runs/play_device.py --drag-ms 500         # 드래그를 더 천천히 (눈으로 보기 좋게)
    python runs/play_device.py --drag-mode motion    # swipe 를 무시하는 경우의 대안

search_bench.py 에서 재던 설정을 그대로 붙여 쓸 수 있다. --preset 을 주면 묻지 않는다.

    python runs/play_device.py --checkpoint ai/models/DQN/V15/models/V15_DQN_50000.zip \
        --preset cut-all --budget 36 --fp16 --value-cache --device cuda

프리셋 표는 runs/agents/ai/presets.py 한 곳에 있고 search_bench.py 와 공유한다.
V15 계열 체크포인트에서만 탐색이 되고, 그 외에는 정책 1회 forward 로 둔다.

시작 순서
  1. 폰을 C타입 케이블로 유선 연결 (개발자 옵션 > USB 디버깅)
  2. scrcpy 실행 - 드래그가 실시간으로 보인다
  3. 이 스크립트를 먼저 실행. 기기를 잡고 모델 로딩이 끝나면 대기 상태가 된다
  4. 그때 게임을 시작하고 Enter

순서에 이유가 둘 있다.

  기기 연결이 맨 앞이다. 케이블이 안 꽂혀 있으면 모델을 읽기 전에 알아야 한다.
  30초 걸려 다 읽고 나서 'no devices' 를 보는 것은 낭비다.

  그 다음이 무거운 초기화(torch import, 모델 로드)이고, 이것도 게임 시작 '전'에
  끝낸다. 게임이 시작된 뒤에 모델을 읽으면 그 수십 초 동안 판이 그냥 놀고 있게 된다.

플레이 도중 연결이 끊기면 DeviceDisconnected 로 즉시 멈춘다 (종료 코드 2).
화면을 못 읽는 채로 계속하면 내부 판만 혼자 진행해서, 재연결되는 순간 엉뚱한
곳을 긁게 된다.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import game
from agents.ai.agent import AIAgent
from agents.ai.presets import preset_labels, preset_names
from agents.ai.search_agent import SearchAgent
from agents.selector import agent_class_for, select_agent
from agents.utils import format_dataclass_box, prompt_int, prompt_yes_no, select_from
from ai.training import resolve_device as resolve_torch_device
from control import vision
from control.controller import PROCESSOR, DryRunProcessor
from control.device import DeviceDisconnected, resolve_device
from control.session import DeviceSession, Orientation, SessionConfig
from game.action import Action

ROOT_DIR = Path(game.__file__).resolve().parent.parent
ROWS, COLS = 9, 18

os.chdir(ROOT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="실 기기 자동 플레이")

    parser.add_argument("--checkpoint", type=Path,
                        help="체크포인트(.zip) 또는 알고리즘 모델(.py). 생략하면 물어본다.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="AI 추론 장치 (기본 auto)")
    parser.add_argument("--serial", help="adb 기기 시리얼. 여러 대 붙어 있을 때만 필요하다.")

    parser.add_argument("--drag-mode", choices=("swipe", "motion"), default="swipe",
                        help="swipe 는 shell 1회로 빠르다. 게임이 swipe 를 무시하면 motion.")
    parser.add_argument("--drag-ms", type=int, default=300,
                        help="드래그 하나에 걸리는 시간(ms). 두 모드에서 같은 뜻이다. "
                             "클수록 화면에서 잘 보인다.")
    parser.add_argument("--motion-steps", type=int, default=8,
                        help="motion 모드에서 DOWN 과 UP 사이에 넣을 MOVE 개수")
    parser.add_argument("--drag-margin", type=float, default=0.45,
                        help="선택 박스를 셀 피치의 몇 배만큼 바깥으로 잡을지")
    parser.add_argument("--move-delay", type=float, default=0.35,
                        help="드래그와 드래그 사이 대기(초). 클리어 애니메이션이 끝날 시간을 준다.")
    parser.add_argument("--resync-every", type=int, default=8,
                        help="N수마다 화면과 대조한다. 1이면 매 수 검증. 0이면 안 한다.")

    # ── 추론 시점 탐색 (V15 계열 체크포인트에서만) ──────────────────────
    # --preset 을 주면 그대로 쓰고, 안 주면 시작할 때 물어본다.
    # 이름과 뜻은 runs/search_bench.py 와 같은 표(agents/ai/presets.py)에서 온다.
    search = parser.add_argument_group("탐색 (search_bench.py 와 같은 프리셋)")
    search.add_argument("--preset", help="국소탐색/NRPA 프리셋. 생략하면 물어본다. "
                                         "예) cut-all")
    search.add_argument("--no-search", action="store_true",
                        help="탐색을 쓰지 않고 정책 1회 forward 로 둔다 (빠르다)")
    search.add_argument("--budget", type=float, default=36.0,
                        help="판당 탐색 예산(초). 첫 빔 시간이 여기 포함된다 (기본 36)")
    search.add_argument("--replan-budget", type=float,
                        help="판이 어긋나 다시 풀 때의 예산(초). 기본은 --budget 과 같다. "
                             "판 도중에 오래 멈추는 게 싫으면 짧게 준다")
    search.add_argument("--fp16", action="store_true",
                        help="인코더를 반정밀도로. cuda/mps 에서만 의미가 있다")
    search.add_argument("--value-cache", action="store_true",
                        help="같은 판을 두 번 평가하지 않는다. 결과는 완전히 같다")
    search.add_argument("--prefilter", type=int, default=0, metavar="N",
                        help="2단 평가. 0 이면 끔 (기본)")
    search.add_argument("--eval-chunk", type=int, default=0, metavar="N",
                        help="한 번에 평가할 자식 수. 0 이면 모델 기본값")
    search.add_argument("--search-seed", type=int, default=0, help="탐색의 난수 씨앗")

    parser.add_argument("--dry-run", action="store_true",
                        help="드래그를 보내지 않고 인식/판정/좌표만 확인한다. 화면 대조는 꺼진다.")
    parser.add_argument("--quiet", action="store_true", help="매 수 로그를 찍지 않는다")

    return parser.parse_args()


def supports_search(source: Path) -> bool:
    """탐색은 plan() 을 가진 V15 계열에서만 된다.

    체크포인트를 통째로 읽지 않고 파일 이름만 본다. 여기서 모델을 두 번 읽으면
    게임 시작 전 대기 시간이 그만큼 길어진다. 실제 확인은 SearchAgent 가 한다.
    """
    return source.suffix == ".zip" and source.parent.parent.name.startswith("V15")


def choose_search(args: argparse.Namespace, source: Path) -> dict | None:
    """탐색 설정을 정한다. 안 쓰면 None.

    --preset 을 줬으면 묻지 않고 그대로 쓴다 (search_bench.py 와 같은 명령을
    그대로 붙여 넣을 수 있어야 한다). 안 줬으면 시작할 때 물어본다.
    """
    if args.no_search or not supports_search(source):
        return None

    device = resolve_torch_device(args.device)
    knobs = dict(device=device, fp16=args.fp16, value_cache=args.value_cache,
                 prefilter=args.prefilter, eval_chunk=args.eval_chunk,
                 seed=args.search_seed, replan_budget=args.replan_budget)

    if args.preset:
        return dict(preset=args.preset, budget=args.budget, **knobs)

    if not prompt_yes_no("\n추론 시점 탐색을 쓸까요? (안 쓰면 정책 1회 forward)", default=True):
        return None

    names = preset_names()
    labels = preset_labels()
    options = [f"{name:<14} {labels[name]}" + ("   <- 문서 기준 최고" if name == "cut-all" else "")
               for name in names]
    preset = names[select_from("Select Search Preset", options)]

    budget = float(prompt_int("판당 탐색 예산(초)", default=int(args.budget)))

    # fp16 은 CPU 에서 의미가 없으므로 물어보지도 않는다
    fp16 = args.fp16 or (device != "cpu" and
                         prompt_yes_no("fp16 인코더를 쓸까요? (cuda/mps 에서 빠름)", default=True))
    value_cache = args.value_cache or prompt_yes_no(
        "값 캐시를 쓸까요? (같은 판 재평가 생략, 결과 동일)", default=True)

    return dict(preset=preset, budget=budget,
                **{**knobs, "fp16": fp16, "value_cache": value_cache})


def build_agent(args: argparse.Namespace):
    """무거운 초기화. 기기를 잡기 전에 여기서 다 끝낸다."""
    if args.checkpoint is not None:
        source = args.checkpoint if args.checkpoint.is_absolute() else ROOT_DIR / args.checkpoint
        if not source.exists():
            raise SystemExit(f"대상을 찾을 수 없습니다: {source}")
        agent_cls = agent_class_for(source)
    else:
        agent_cls, source = select_agent(agents_dir=ROOT_DIR)

    if issubclass(agent_cls, AIAgent):
        search = choose_search(args, source)
        if search is not None:
            return SearchAgent((ROWS, COLS), source, **search)
        return AIAgent((ROWS, COLS), source, device=resolve_torch_device(args.device))

    return agent_cls((ROWS, COLS), source)


def agent_label(agent) -> str:
    info = agent.get_info()
    name = getattr(info, "model_name", None) or getattr(info, "algorithm_name", "?")
    return f"{name} / V{info.agent_version}"


def make_move_printer(session: DeviceSession, show_pixels: bool):
    def on_move(moves: int, action: Action, removed: int, score: int) -> None:
        (r1, c1), (r2, c2) = action.top_left, action.bottom_right
        line = f"[{moves:3d}] ({r1},{c1})~({r2},{c2})  +{removed}  score={score}"
        if show_pixels:
            sx, sy, ex, ey = session.controller.rect_pixels(
                session.orientation.to_screen_action(action))
            line += f"   drag ({sx},{sy})->({ex},{ey})"
        print(line, flush=True)
    return on_move


def main() -> int:
    args = parse_args()

    # 1) 기기 연결을 가장 먼저. 케이블이 안 꽂혀 있으면 모델을 읽기 전에 알아야
    #    한다. 30초 걸려 모델을 다 읽고 나서 'no devices' 를 보는 것은 낭비다.
    device = resolve_device(args.serial)
    templates = vision.load_templates()
    print(f"기기 연결: {device.serial}\n")

    # 2) 에이전트 (느린 단계). 게임이 시작되기 전에 끝내야 판이 놀지 않는다.
    agent = build_agent(args)
    print(agent.get_info_formatted())

    # 3) 게임 화면 대기
    if args.dry_run:
        print("[dry-run] 드래그를 보내지 않습니다. 화면 대조도 꺼집니다.")
    input("\n게임 시작 화면(사과가 전부 있는 상태)을 띄우고 Enter > ")

    # 4) 격자와 판
    img = vision.capture(device)
    geom = vision.detect_grid(img)
    orientation = Orientation.detect(geom, (ROWS, COLS))
    print(f"격자 검출: {geom.rows}행 x {geom.cols}열  "
          f"(pitch {geom.pitch_x:.0f}x{geom.pitch_y:.0f}px, 방향 {orientation.describe()})")

    controller = (
        DryRunProcessor(geom, mode=args.drag_mode, duration_ms=args.drag_ms,
                        margin_ratio=args.drag_margin, motion_steps=args.motion_steps)
        if args.dry_run else
        PROCESSOR(device, geom, mode=args.drag_mode, duration_ms=args.drag_ms,
                  margin_ratio=args.drag_margin, motion_steps=args.motion_steps)
    )
    config = SessionConfig(
        move_delay=0.0 if args.dry_run else args.move_delay,
        # dry-run 은 드래그를 보내지 않아 화면이 시작 상태 그대로다. 대조하면
        # 매번 어긋난 것으로 나오고, 최종 검증은 판을 통째로 되살려 버린다.
        resync_every=0 if args.dry_run else args.resync_every,
        verify_on_finish=not args.dry_run,
    )
    session = DeviceSession(device, geom, templates, controller, orientation, config)

    board, confidence = session.read_initial_board(img)
    print(f"판 인식 완료 (최소 신뢰도 {confidence:.3f})\n")
    print(vision.format_board(board.grid))
    print()

    # 5) 탐색이면 여기서 판 전체를 먼저 푼다. select_action 이 첫 호출에서 알아서
    #    풀기는 하지만, 그러면 아무 설명 없이 수십 초 멈춘 것처럼 보인다.
    if isinstance(agent, SearchAgent):
        print(f"탐색 중... (예산 {agent.budget:.0f}초, preset={agent.preset})", flush=True)
        started = time.time()
        agent.plan_for(board)
        print(f"수순 {agent.plan_length}수 완성 "
              f"(예상 {agent.last_score}점, 빔만 썼을 때 {agent.last_init_score}점, "
              f"{time.time() - started:.1f}초)\n", flush=True)

    # 6) 플레이
    on_move = None if args.quiet else make_move_printer(session, show_pixels=args.dry_run)
    result = session.play(agent, agent_name=agent_label(agent), on_move=on_move)

    print()
    print(format_dataclass_box("Play Result", result))
    print()
    print(vision.format_board(session.board.grid))

    if args.dry_run:
        print(f"\n[dry-run] 보내지 않은 드래그 명령 {len(controller.commands)}개")  # type: ignore[union-attr]
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DeviceDisconnected as e:
        # RuntimeError 의 하위라 반드시 아래 절보다 먼저 와야 한다.
        print(f"\n[연결 끊김] {e}", file=sys.stderr)
        raise SystemExit(2)
    except RuntimeError as e:
        print(f"\n[중단] {e}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 멈췄습니다.", file=sys.stderr)
        raise SystemExit(130)
