from dataclasses import dataclass
from typing import Callable
from .feature_context import FeatureContext
from . import features

@dataclass
class FeatureSpec:
    name: str
    func: Callable[[FeatureContext], float]
    weight: float

FEATURES = [
    FeatureSpec("remove_nine", features.feature_remove_nine, weight = 1.0),
    FeatureSpec("start_at_middle", features.feature_start_at_middle, weight = 1.0),
    FeatureSpec("start_at_side", features.feature_start_at_side, weight = 1.0),
    
]

