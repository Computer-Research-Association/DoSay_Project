import math

def sigmoid(x: float, k: float = 0.5, x0: float = 3.0) -> float:
    return 1 / (1 + math.exp(-k * (x - x0)))