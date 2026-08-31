"""
DL 정책 — 학습된 넷으로 후보 수를 배치 추론해 argmax 선택 (action-max의 빠른 근사).
anneal.py의 rollout_policy(grid, actions, rng) 인터페이스 호환.
"""
import os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dl.train import Net, build_inputs   # noqa: E402

_net = None
_dev = None


def _load(device=None):
    global _net, _dev
    _dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    _net = Net().to(_dev)
    _net.load_state_dict(torch.load(ROOT / "dl" / "model.pt", map_location=_dev))
    _net.eval()


def rollout_policy(grid, actions, rng):
    if _net is None:
        _load()
    rects = np.asarray(actions, dtype=np.int16)
    boards = np.broadcast_to(grid, (len(actions),) + grid.shape)
    X = build_inputs(boards, rects).to(_dev)
    with torch.no_grad():
        scores = _net(X)
    return actions[int(scores.argmax())]
