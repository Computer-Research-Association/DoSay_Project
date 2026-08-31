"""한 (판 index, instance) 작업 — boards.npy[idx]를 어닐링. "idx score" 출력."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.anneal import anneal_once

idx = int(sys.argv[1]); inst = int(sys.argv[2]); iters = int(sys.argv[3])
boards = np.load(ROOT / "testcases" / "m10hq" / "boards.npy")
sc, _ = anneal_once(boards[idx], iters=iters, rng_seed=inst)
print(f"{idx} {sc}")
