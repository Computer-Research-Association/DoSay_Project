# Data Transfer Object -> 값 전달용 클래스 모음
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

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


@dataclass
class AlgoInfo:
    model_type: str           = field(metadata={"label": "Model Type"})
    algorithm_name: str       = field(metadata={"label": "Algorithm"})
    agent_version: str        = field(metadata={"label": "Version"})
    source_path: Path         = field(metadata={"label": "File"})
    beam_width: Optional[int] = field(metadata={"label": "Beam Width"}, default=None)
    max_depth: Optional[int]  = field(metadata={"label": "Beam Depth"}, default=None)