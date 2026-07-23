from dataclasses import dataclass
from typing import Callable

from game.board import Board

@dataclass
class HeuristicEntry:
    func: Callable
    weight: float
    name: str

class HeuristicRegistry:
    def __init__(self):
        self.entries: list[HeuristicEntry] = []

    def heuristic(self, weight: float):
        def decorator(func):
            self.entries.append(HeuristicEntry(func=func, weight=weight, name=func.__name__))
            return func
        return decorator

    def evaluate(self, state: Board) -> float:
        return sum(e.weight * e.func(state) for e in self.entries)

    # def evaluate_verbose(self, state) -> dict:
    #     raw = {e.name: e.func(state) for e in self.entries}
    #     weighted = {e.name: e.weight * raw[e.name] for e in self.entries}
    #     return {"raw": raw, "weighted": weighted, "total": sum(weighted.values())}