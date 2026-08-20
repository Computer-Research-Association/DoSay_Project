"""adb 기기 연결.

기기가 안 잡히는 원인은 늘 몇 가지로 정해져 있는데(케이블, USB 디버깅, adb 서버),
ppadb 가 내는 예외는 그중 무엇인지 알려주지 않는다. 여기서 원인별로 갈라 준다.
"""

from typing import Any

ADB_HOST = "127.0.0.1"
ADB_PORT = 5037


def resolve_device(serial: str | None = None) -> Any:
    """연결된 기기 하나. serial 을 주면 그 기기, 없으면 첫 번째.

    반환 타입은 ppadb 의 Device 지만, ppadb 를 못 찾는 환경에서도 이 모듈을
    import 할 수 있어야 해서(예: --dry-run) 지연 import 하고 Any 로 둔다.
    """
    try:
        from ppadb.client import Client
    except ImportError as e:
        raise RuntimeError(
            "ppadb 가 없습니다. 다음으로 설치해 주세요:\n"
            "    pip install pure-python-adb"
        ) from e

    client = Client(host=ADB_HOST, port=ADB_PORT)
    try:
        devices = client.devices()
    except RuntimeError as e:
        raise RuntimeError(
            f"adb 서버에 연결하지 못했습니다 ({ADB_HOST}:{ADB_PORT}).\n"
            "    adb start-server\n"
            "를 실행한 뒤 다시 시도해 주세요."
        ) from e

    if not devices:
        raise RuntimeError(
            "연결된 기기가 없습니다.\n"
            "  1. C타입 케이블로 유선 연결\n"
            "  2. 개발자 옵션 > USB 디버깅 켜기\n"
            "  3. 폰에 뜨는 'USB 디버깅을 허용하시겠습니까' 를 허용\n"
            "  4. adb devices 로 목록에 뜨는지 확인"
        )

    if serial is None:
        return devices[0]

    for device in devices:
        if device.serial == serial:
            return device

    available = ", ".join(d.serial for d in devices)
    raise RuntimeError(f"'{serial}' 기기를 찾을 수 없습니다. 연결된 기기: {available}")


class DeviceDisconnected(RuntimeError):
    """플레이 도중 기기와의 연결이 끊겼다.

    이 상황에서 계속 진행하면 안 된다. 화면을 못 읽으니 판이 어떤 상태인지 알 수
    없고, 드래그도 안 나가는데 내부 판만 혼자 진행해서 재연결되는 순간 엉뚱한
    곳을 긁게 된다. 조용히 넘기는 대신 바로 멈춘다.
    """


def device_call(fn, *args, **kwargs):
    """ppadb 호출 하나를 감싸 연결 끊김을 DeviceDisconnected 로 바꾼다.

    ppadb 는 원인을 가리지 않고 예외를 던진다. 케이블이 빠지면 adb 서버가
    'device not found' 로 답해 RuntimeError 가 되고, 전송 중에 끊기면 소켓
    예외나 빈 응답 파싱 실패(ValueError)가 된다. 셋 다 같은 뜻이다.

    감싸는 범위를 ppadb 호출 한 줄로 좁게 잡는다. 넓게 잡으면 우리가 직접
    던진 RuntimeError 까지 연결 문제로 둔갑한다.
    """
    try:
        return fn(*args, **kwargs)
    except (OSError, ValueError, RuntimeError) as e:
        raise DeviceDisconnected(
            f"기기와의 연결이 끊겼습니다: {type(e).__name__}: {e}\n"
            "  케이블, USB 디버깅, 화면 잠금을 확인하고 adb devices 로 다시 잡히는지 보세요."
        ) from e
