import os
import re
import importlib
from agents.dtos import AIInfo
from ai.envs.apple_env import AppleGameEnv

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

def _parse_filename(path: str) -> AIInfo:
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

    return AIInfo(
        model_name=name,
        agent_version=m.group("version"),
        source_path=path,
        model_policy="-1",
        model_obs="-1",
        model_act="-1",
        total_train_steps=int(m.group("steps")),
        use_action_masking=maskable
    )

def load_model(path: str, env: AppleGameEnv, device: str = "cpu"):
    info = _parse_filename(path)
    module_name, class_name, _ = MODEL_REGISTRY[info.model_name]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls.load(path, env=env, device=device)

    # device = str(getattr(model, "device", "?"))

    info.model_policy = type(getattr(model, "policy", model)).__name__
    info.model_obs = getattr(model, "observation_space", "?")
    info.model_act = getattr(model, "action_space", "?")

    return model, info

