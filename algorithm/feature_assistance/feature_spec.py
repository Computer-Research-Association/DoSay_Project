from dataclasses import dataclass
from typing import Callable
from algorithm.feature_assistance.feature_context import FeatureContext
from algorithm.feature import features

@dataclass
class FeatureSpec:
    name: str
    func: Callable[[FeatureContext], float]
    weight: float

FEATURES = [
    FeatureSpec("remove_nine", features.feature_remove_nine, weight = 1.0),
    FeatureSpec("remove_eight", features.feature_remove_eight, weight = 1.0),
    FeatureSpec("grouping", features.feature_remove_the_most_grouping, weight = 1.0),
]

