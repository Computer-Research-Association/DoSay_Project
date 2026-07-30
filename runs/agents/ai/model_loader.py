import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3.common.save_util import load_from_zip_file

from agents.dtos import AIInfo
from ai.wrappers.action_mask import wrap_with_mask

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

# 버전 폴더 안의 체크포인트 파일명 규격
_NAME_PATTERN = re.compile(r"^(?P<name>.+)_V(?P<version>[\d.]+)_(?P<steps>\d+)$")


def find_checkpoint(version_dir: Path) -> Path | None:
    """버전 폴더의 체크포인트. 없으면 None (= 아직 학습되지 않은 버전)."""
    zips = sorted(version_dir.glob("*.zip"))
    return zips[0] if zips else None


def _load_sibling(version_dir: Path, stem: str):
    path = version_dir / f"{stem}.py"
    if not path.exists():
        raise FileNotFoundError(
            f"버전 폴더에 {stem}.py 가 없습니다: {version_dir}\n"
            f"각 버전 폴더는 env.py(make_env) 와 model.py(POLICY_CLASS, make_policy_kwargs) 를 "
            "갖고 있어야 합니다."
        )

    mod_name = f"_version_{version_dir.name.replace('.', '_')}_{stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} 를 모듈로 로드할 수 없습니다.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _parse_checkpoint_name(ckpt: Path, version_dir: Path) -> tuple[str, str, int]:
    """체크포인트 파일명에서 (알고리즘, 버전, 학습스텝)을 얻고 폴더명과 대조"""
    m = _NAME_PATTERN.match(ckpt.stem)
    if not m:
        raise ValueError(
            f"파일명 규격을 인식할 수 없습니다: '{ckpt.stem}'\n"
            f"기대 형식: (모델이름)_V(버전)_(학습횟수)  예) MaskablePPO_V5.0_10000000"
        )

    name = m.group("name")
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"레지스트리에 없는 모델입니다: '{name}'. "
            f"MODEL_REGISTRY 에 추가해 주세요. (등록됨: {list(MODEL_REGISTRY)})"
        )

    version = m.group("version")
    folder_version = version_dir.name.lstrip("Vv")
    if version != folder_version:
        raise ValueError(
            f"폴더명과 체크포인트 파일명의 버전이 다릅니다.\n"
            f"  폴더  : {version_dir.name}  (-> {folder_version})\n"
            f"  파일  : {ckpt.name}  (-> {version})\n"
            "이름이 실제 내용을 잘못 가리키면 어떤 코드로 학습된 모델인지 추적할 수 없습니다."
        )

    return name, version, int(m.group("steps"))


def _verify_no_drift(model, saved_keys: set[str], ckpt: Path, version_dir: Path) -> None:
    """체크포인트와 실제로 만들어진 정책의 파라미터가 완전히 일치하는지 검사"""
    live_keys = set(model.policy.state_dict().keys())
    missing = saved_keys - live_keys
    unexpected = live_keys - saved_keys
    if not (missing or unexpected):
        return

    def _preview(keys: set[str]) -> str:
        head = sorted(keys)[:5]
        return ", ".join(head) + (f" ... (총 {len(keys)}개)" if len(keys) > len(head) else "")

    detail = []
    if missing:
        detail.append(f"  체크포인트에만 있음: {_preview(missing)}")
    if unexpected:
        detail.append(f"  현재 코드에만 있음  : {_preview(unexpected)}")

    raise RuntimeError(
        f"체크포인트와 버전 폴더의 신경망 구조가 일치하지 않습니다: {ckpt.name}\n"
        + "\n".join(detail)
        + f"\n{version_dir / 'model.py'} 가 학습 이후 수정된 것으로 보입니다.\n"
        "체크포인트가 있는 버전 폴더는 동결 대상입니다. 구조를 바꾸려면 "
        "새 버전 폴더를 만들어 거기서 다시 학습하세요."
    )


def load_model(version_dir: Path, grid_shape: tuple[int, int], device: str = "cpu"):
    """버전 폴더에서 (모델, 환경, 정보)를 복원"""
    rows, cols = grid_shape

    ckpt = find_checkpoint(version_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"'{version_dir.name}' 에는 학습된 체크포인트(.zip)가 없습니다.\n"
            f"먼저 {version_dir / 'train.py'} 를 실행해 모델을 만들어 주세요."
        )

    model_name, version, steps = _parse_checkpoint_name(ckpt, version_dir)
    module_name, class_name, maskable = MODEL_REGISTRY[model_name]
    cls = getattr(importlib.import_module(module_name), class_name)

    env_mod = _load_sibling(version_dir, "env")
    model_mod = _load_sibling(version_dir, "model")

    env: gym.Env = wrap_with_mask(env_mod.make_env(rows, cols, render_mode="ansi"))

    custom_objects: dict[str, Any] = {
        "policy_class": model_mod.POLICY_CLASS,
        "policy_kwargs": model_mod.make_policy_kwargs(rows, cols),
    }

    _, params, _ = load_from_zip_file(ckpt, load_data=False, device=device)
    saved_keys = set(params["policy"].keys())  # type: ignore

    model = cls.load(ckpt, env=env, device=device, custom_objects=custom_objects)
    _verify_no_drift(model, saved_keys, ckpt, version_dir)

    info = AIInfo(
        model_type="AI",
        model_name=model_name,
        agent_version=version,
        source_path=ckpt.relative_to(Path.cwd()),
        model_policy=type(getattr(model, "policy", model)).__name__,
        model_obs=getattr(model, "observation_space", "?"),
        model_act=getattr(model, "action_space", "?"),
        total_train_steps=steps,
        use_action_masking=maskable,
    )

    return model, env, info
