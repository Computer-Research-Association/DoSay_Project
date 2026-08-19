# Data Transfer Object -> 값 전달용 클래스 모음
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

@dataclass
class AIInfo:
    model_type: str           = field(metadata={"label": "Model Type"})
    model_name: str           = field(metadata={"label": "Algorithm"})
    agent_version: str        = field(metadata={"label": "Version"})
    source_path: Path         = field(metadata={"label": "File"})
    model_policy: str         = field(metadata={"label": "Policy"})
    model_obs: str            = field(metadata={"label": "Obs space"})
    model_act: str            = field(metadata={"label": "Act space"})
    total_train_steps: int    = field(metadata={"label": "Train steps"})
    use_action_masking: bool  = field(metadata={"label": "Maskable"})
    # 버전 env.py 의 SEARCH_* 설정 요약. "off" 면 정책만으로 둔다.
    search: Optional[str]     = field(metadata={"label": "Search"}, default=None)


@dataclass
class TrainInfo:
    agent_version: str        = field(metadata={"label": "Version"})
    total_timestep: int       = field(metadata={"label": "Total Timestep"})
    checkpoint_timestep: int  = field(metadata={"label": "Checkpoint Timestep"})
    save_path: Path           = field(metadata={"label": "Save to"})
    device: str               = field(metadata={"label": "Device"}, default="auto")
    n_envs: Optional[int]     = field(metadata={"label": "Envs"}, default=None)


@dataclass
class MeasureResult:
    """벤치마크 한 번의 결과. 박스 출력과 JSON 저장에 함께 쓴다."""
    agent: str                = field(metadata={"label": "Agent"})
    source_path: Path         = field(metadata={"label": "File"})
    episodes: int             = field(metadata={"label": "Episodes"})
    base_seed: int            = field(metadata={"label": "Base seed"})
    avg_moves: float          = field(metadata={"label": "Avg moves"})
    avg_score: float          = field(metadata={"label": "Avg score"})
    std_score: float          = field(metadata={"label": "Std score"})
    best_score: int           = field(metadata={"label": "Best score"})
    best_seed: int            = field(metadata={"label": "Best seed"})
    worst_score: int          = field(metadata={"label": "Worst score"})
    worst_seed: int           = field(metadata={"label": "Worst seed"})
    elapsed_sec: float        = field(metadata={"label": "Elapsed (s)"})
    sec_per_episode: float    = field(metadata={"label": "Sec/episode"})
    measured_at: str          = field(metadata={"label": "Measured at"})
    scores: list[int]         = field(metadata={"label": "Scores", "hidden": True}, default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_path"] = str(self.source_path)
        return data


@dataclass
class AlgoInfo:
    model_type: str           = field(metadata={"label": "Model Type"})
    algorithm_name: str       = field(metadata={"label": "Algorithm"})
    agent_version: str        = field(metadata={"label": "Version"})
    source_path: Path         = field(metadata={"label": "File"})
    beam_width: Optional[int] = field(metadata={"label": "Beam Width"}, default=None)
    max_depth: Optional[int]  = field(metadata={"label": "Beam Depth"}, default=None)