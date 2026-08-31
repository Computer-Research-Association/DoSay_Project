"""한 보드 = max-of-16 × 10000 iter (numba). "seed score" 출력."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.board import make_board
from models.anneal import anneal_once

seed = int(sys.argv[1]); inst = int(sys.argv[2]); iters = int(sys.argv[3])
b = make_board(seed)
best = max(anneal_once(b, iters=iters, rng_seed=i)[0] for i in range(inst))
print(f"{seed} {best}")
