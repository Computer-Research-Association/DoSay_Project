import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3.common.save_util import load_from_zip_file

from agents.dtos import AIInfo
from agents.ai.search import SearchConfig
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

# 체크포인트 파일명 규격: V(버전)_(알고리즘)_(학습스텝)
# 버전에 글자를 허용한다 (V8a, V8b 처럼 같은 세대의 변형을 나란히 두기 위해).
# 버전 부분에서 밑줄은 일부러 제외 — 버전/알고리즘/스텝 경계가 모호해지면 안 된다.
_NAME_PATTERN = re.compile(r"^V(?P<version>[0-9A-Za-z.]+)_(?P<name>.+)_(?P<steps>\d+)$")

# 옛 규격 (알고리즘)_V(버전)_(스텝). 규격을 바꾸기 전에 만들어 둔 체크포인트와,
# 규격 변경 시점에 이미 돌고 있던 학습이 저장하는 파일을 위해 남겨 둔다.
# runs/rename_checkpoints.py 로 일괄 변환할 수 있다.
_LEGACY_PATTERN = re.compile(r"^(?P<name>.+)_V(?P<version>[0-9A-Za-z.]+)_(?P<steps>\d+)$")


def _match_name(stem: str) -> re.Match | None:
    return _NAME_PATTERN.match(stem) or _LEGACY_PATTERN.match(stem)

# train.py 가 체크포인트/최종본 구분 없이 모두 넣어 두는 폴더
MODELS_DIRNAME = "models"


def find_checkpoints(version_dir: Path) -> list[Path]:
    """버전 폴더가 가진 모든 체크포인트를 학습 스텝 오름차순으로.

    models/ 아래가 정식 위치이고, 버전 폴더 바로 아래도 함께 훑는다
    (models/ 도입 이전에 학습한 모델을 그대로 쓸 수 있게).
    파일명 규격에 맞지 않는 zip 은 조용히 건너뛰고,
    양쪽에 같은 이름이 있으면 models/ 쪽만 남긴다.
    """
    candidates = [*(version_dir / MODELS_DIRNAME).glob("*.zip"), *version_dir.glob("*.zip")]

    found: dict[str, tuple[int, Path]] = {}
    for path in candidates:
        m = _match_name(path.stem)
        if m and path.stem not in found:
            found[path.stem] = (int(m.group("steps")), path)

    return [path for _, path in sorted(found.values(), key=lambda item: (item[0], item[1].name))]


def describe_checkpoint(checkpoint: Path) -> str:
    """선택 목록에 보여줄 한 줄 설명."""
    m = _match_name(checkpoint.stem)
    if m is None:
        return checkpoint.name
    return f"{int(m.group('steps')):>12,} steps   ({m.group('name')})"


def resolve_version_dir(checkpoint: Path) -> Path:
    """체크포인트가 속한 버전 폴더. models/ 하위든 버전 폴더 바로 아래든 모두 지원."""
    parent = checkpoint.parent
    return parent.parent if parent.name == MODELS_DIRNAME else parent


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
    m = _match_name(ckpt.stem)
    if not m:
        raise ValueError(
            f"파일명 규격을 인식할 수 없습니다: '{ckpt.stem}'\n"
            f"기대 형식: V(버전)_(모델이름)_(학습횟수)  예) V9b_DQN_12000000"
        )

    name = m.group("name")
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"레지스트리에 없는 모델입니다: '{name}'. "
            f"MODEL_REGISTRY 에 추가해 주세요. (등록됨: {list(MODEL_REGISTRY)})"
        )

    # ai/models/<알고리즘>/<버전>/ 이므로 상위 폴더가 알고리즘 이름이어야 한다
    algorithm_folder = version_dir.parent.name
    if name != algorithm_folder:
        raise ValueError(
            f"폴더 구조와 체크포인트의 알고리즘이 다릅니다.\n"
            f"  폴더  : {algorithm_folder}\n"
            f"  파일  : {ckpt.name}  (-> {name})\n"
            f"{version_dir} 는 {algorithm_folder} 폴더 아래에 있습니다."
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


def _resolve_checkpoint(source: Path) -> tuple[Path, Path]:
    """(체크포인트, 버전 폴더). source 는 체크포인트 파일이거나 버전 폴더."""
    if source.is_dir():
        checkpoints = find_checkpoints(source)
        if not checkpoints:
            raise FileNotFoundError(
                f"'{source.name}' 에는 학습된 체크포인트(.zip)가 없습니다.\n"
                f"먼저 {source / 'train.py'} 를 실행해 모델을 만들어 주세요."
            )
        return checkpoints[-1], source  # 지정이 없으면 가장 많이 학습된 것

    if not source.is_file():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {source}")

    return source, resolve_version_dir(source)


def _search_config(env_mod) -> SearchConfig:
    """버전 env.py 가 선언한 탐색 설정. 없으면 탐색 없음."""
    return SearchConfig(
        top_k=getattr(env_mod, "SEARCH_TOP_K", 0),
        depth=getattr(env_mod, "SEARCH_DEPTH", 1),
        beam_width=getattr(env_mod, "SEARCH_BEAM_WIDTH", 1),
        beam_top_k=getattr(env_mod, "SEARCH_BEAM_TOP_K", 0),
    )


def _training_only_objects(env_mod) -> dict[str, Any]:
    """학습 전용 상태를 로드할 때 무엇으로 대체할지. 버전 env.py 가 선언한다.

    SB3 의 model.save() 는 알고리즘 객체의 __dict__ 를 통째로 직렬화한다. 그래서
    학습 중에만 쓰는 보조 상태(교사용 env, 모방 버퍼 같은 것)까지 체크포인트에
    들어간다. 그중 **버전 폴더의 env.py 에 정의된 클래스**의 인스턴스는 특히 위험하다.

    학습할 때는 train.py 가 sys.path 에 버전 폴더를 넣고 `import env` 를 하므로
    그 클래스의 모듈 이름이 최상위 'env' 가 된다. cloudpickle 은 import 가능한
    모듈의 클래스를 **참조로** 저장하므로 체크포인트에는 "모듈 env 의 클래스"라고만
    적힌다. 그런데 measure.py 는 버전마다 env.py 를 `_version_<버전>_env` 라는 다른
    이름으로 로드하므로 'env' 라는 모듈이 존재하지 않고, 역직렬화가
    ModuleNotFoundError: No module named 'env' 로 죽는다. (실제로 QRDQN/V10c 가
    이 경로로 로드 불가 상태였다.)

    train.py 에 정의된 클래스는 같은 문제가 없다. __main__ 의 클래스는 cloudpickle 이
    **값으로** 저장하기 때문이다 — DQN/V10b 의 EliteBuffer 가 멀쩡한 이유다.

    근본 대책은 train.py 의 _excluded_save_params() 에 그 속성을 넣어 애초에
    저장하지 않는 것이고, 여기 있는 것은 **그 전에 만들어진 체크포인트를 재학습 없이
    살리기 위한 통로**다. SB3 의 json_to_data 는 custom_objects 에 키가 있으면
    역직렬화를 아예 건너뛰므로, 깨진 값이 들어 있어도 로드가 성공한다.
    """
    declared = getattr(env_mod, "LOAD_CUSTOM_OBJECTS", {})
    if not isinstance(declared, dict):
        raise TypeError(
            f"LOAD_CUSTOM_OBJECTS 는 dict 여야 합니다: {type(declared).__name__}\n"
            "  예) LOAD_CUSTOM_OBJECTS = {'_search_env': None, 'expert_obs': []}"
        )
    return dict(declared)


def load_model(source: Path, grid_shape: tuple[int, int], device: str = "cpu"):
    """체크포인트(또는 버전 폴더)에서 (모델, 환경, 정보, 탐색설정)을 복원"""
    rows, cols = grid_shape

    ckpt, version_dir = _resolve_checkpoint(source)
    model_name, version, steps = _parse_checkpoint_name(ckpt, version_dir)
    module_name, class_name, maskable = MODEL_REGISTRY[model_name]
    cls = getattr(importlib.import_module(module_name), class_name)

    env_mod = _load_sibling(version_dir, "env")
    model_mod = _load_sibling(version_dir, "model")

    env: gym.Env = wrap_with_mask(env_mod.make_env(rows, cols, render_mode="ansi"))

    custom_objects: dict[str, Any] = {
        "policy_class": model_mod.POLICY_CLASS,
        "policy_kwargs": model_mod.make_policy_kwargs(rows, cols),
        **_training_only_objects(env_mod),
    }

    _, params, _ = load_from_zip_file(ckpt, load_data=False, device=device)
    saved_keys = set(params["policy"].keys())  # type: ignore

    model = cls.load(ckpt, env=env, device=device, custom_objects=custom_objects)
    _verify_no_drift(model, saved_keys, ckpt, version_dir)
    search_config = _search_config(env_mod)

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
        search=search_config.describe(),
    )

    return model, env, info, search_config
