"""한 보드(index) = max-of-16 × 10000 (numba). "idx score" 출력."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.anneal import anneal_once

idx = int(sys.argv[1]); inst = int(sys.argv[2]); iters = int(sys.argv[3])
boards = np.load(ROOT / "testcases" / "m10big" / "boards.npy")
best = max(anneal_once(boards[idx], iters=iters, rng_seed=i)[0] for i in range(inst))
print(f"{idx} {best}")
