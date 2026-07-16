import os
import re
import time
import logging
import warnings
import importlib
from dataclasses import dataclass

MODEL_REGISTRY = {
    "MaskablePPO":  ("sb3_contrib",       "MaskablePPO",  True),
    "RecurrentPPO": ("sb3_contrib",       "RecurrentPPO", False),
    "QRDQN":        ("sb3_contrib",       "QRDQN",        False),
    "TRPO":         ("sb3_contrib",       "TRPO",         False),
    "PPO":          ("stable_baselines3", "PPO",          False),
    "A2C":          ("stable_baselines3", "A2C",          False),
    "DQN":          ("stable_baselines3", "DQN",          False),
    "SAC":          ("stable_baselines3", "SAC",          False),
    "TD3":          ("stable_baselines3", "TD3",          False),
    "DDPG":         ("stable_baselines3", "DDPG",         False),
}

_NAME_PATTERN = re.compile(r"^(?P<name>.+)_V(?P<version>[\d.]+)_(?P<steps>\d+)$")


@dataclass
class ModelMeta:
    name: str        # "MaskablePPO"
    version: str     # "5.0"
    steps: int       # 10000000
    maskable: bool   # 액션마스킹 사용 여부
    path: str        # 원본 경로


def parse_model_filename(path: str) -> ModelMeta:
    """경로에서 (모델이름)_V(버전)_(학습횟수) 를 뽑아 ModelMeta 반환."""
    # 확장자/폴더 제거 (윈도우 백슬래시 경로도 안전하게)
    stem = os.path.splitext(os.path.basename(path.replace("\\", "/")))[0]

    m = _NAME_PATTERN.match(stem)
    if not m:
        raise ValueError(
            f"파일명 규격을 인식할 수 없습니다: '{stem}'\n"
            f"기대 형식: (모델이름)_V(버전)_(학습횟수)  예) MaskablePPO_V5.0_10000000"
        )

    name = m.group("name")
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"레지스트리에 없는 모델입니다: '{name}'. "
            f"common.MODEL_REGISTRY 에 추가해 주세요. (등록됨: {list(MODEL_REGISTRY)})"
        )

    _, _, maskable = MODEL_REGISTRY[name]
    return ModelMeta(
        name=name,
        version=m.group("version"),
        steps=int(m.group("steps")),
        maskable=maskable,
        path=path,
    )


def load_model(path: str, env=None):
    """파일명 기반으로 알맞은 클래스를 골라 모델을 로드. (model, meta) 반환."""
    meta = parse_model_filename(path)
    module_name, class_name, _ = MODEL_REGISTRY[meta.name]
    cls = getattr(importlib.import_module(module_name), class_name)
    print(cls)
    model = cls.load(path, env=env)
    return model, meta


def quiet_logs():
    """라이브러리 경고/로그 도배 최소화."""
    warnings.filterwarnings("ignore")
    for name in ("gymnasium", "gym", "stable_baselines3", "sb3_contrib"):
        logging.getLogger(name).setLevel(logging.ERROR)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def format_model_info(model, meta: ModelMeta) -> str:
    """모델 정보를 보기 좋은 문자열로."""
    policy = type(getattr(model, "policy", model)).__name__
    device = str(getattr(model, "device", "?"))
    obs = getattr(model, "observation_space", "?")
    act = getattr(model, "action_space", "?")
    lines = [
        "┌─ Model Info ───────────────────────────────",
        f"│ Algorithm  : {meta.name}",
        f"│ Version    : V{meta.version}",
        f"│ Train steps: {meta.steps:,}",
        f"│ Maskable   : {meta.maskable}",
        f"│ Policy     : {policy}",
        f"│ Device     : {device}",
        f"│ Obs space  : {obs}",
        f"│ Act space  : {act}",
        f"│ File       : {meta.path}",
        "└────────────────────────────────────────────",
    ]
    return "\n".join(lines)


def run_episode(model, env, meta: ModelMeta, *,
                seed=None, init_board=None,
                render=False, delay=0.0):
    """
    에피소드 1회 실행. (steps, score) 반환.

    seed        : reset 시드 (재현용)
    init_board  : 2차원 배열 보드. 주어지면 seed보다 우선해 판을 고정.
    render      : True면 매 수 기보 출력 + env.render()
    delay       : render=True 일 때 한 수 대기(초). render=False면 무시.
    """
    reset_kwargs = {}
    if seed is not None:
        reset_kwargs["seed"] = seed
    if init_board is not None:
        reset_kwargs["options"] = {"init_board": init_board}

    obs, info = env.reset(**reset_kwargs)
    terminated = truncated = False
    steps = 0
    score = info.get("score", 0)

    if render:
        env.render()

    while not (terminated or truncated):
        if meta.maskable:
            action_masks = env.unwrapped.get_action_mask()
            if not action_masks.any():
                break  # 유효한 수 없음 -> 종료
            action, _ = model.predict(obs, action_masks=action_masks,
                                      deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(int(action))
        steps += 1
        score = info.get("score", score)

        if render:
            print(f"[step {steps:>3}] reward={reward:>5.1f}  score={score}")
            env.render()
            if delay > 0:
                time.sleep(delay)

    return steps, score