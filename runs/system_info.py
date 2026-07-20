import os
import platform
from typing import Optional

from agents.base import Agent
from agents.ai.agent import AIAgent
from agents.utils import format_box

try: import psutil
except ImportError: psutil = None
try: import torch
except ImportError: torch = None


def _cpu_name() -> str:
    return platform.processor() or platform.uname().machine or "Unknown"


def _cpu_cores() -> str:
    logical = os.cpu_count() or "?"
    physical = psutil.cpu_count(logical=False) if psutil else "?"
    return f"{physical} physical / {logical} logical"


def _ram_total() -> str:
    if psutil is None:
        return "Unknown (psutil not installed)"
    gb = psutil.virtual_memory().total / (1024 ** 3)
    return f"{gb:.1f} GB"


def _detect_accelerator() -> str:
    if torch is None:
        return "N/A (torch not installed)"
    if torch.cuda.is_available():
        return "CUDA"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "MPS"
    return "Unavailable (CPU only)"


def _gpu_info(accelerator: str) -> tuple[str, str]:
    if accelerator == "CUDA" and torch is not None:
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        return name, f"{mem_gb:.1f} GB"
    if accelerator == "MPS":
        return "Apple Silicon GPU", "N/A (shared memory)"
    return "N/A", "N/A"


def _base_rows() -> list[tuple[str, str]]:
    return [
        ("OS", f"{platform.system()} {platform.release()}"),
        ("CPU", _cpu_name()),
        ("Cores", _cpu_cores()),
        ("RAM", _ram_total()),
        ("Python", platform.python_version()),
    ]


def _ai_rows(agent: Optional[Agent]) -> list[tuple[str, str]]:
    """AIAgent일 때만 PyTorch/가속기/GPU 관련 정보 추가"""
    if not isinstance(agent, AIAgent):
        return []

    accelerator = _detect_accelerator()
    gpu_name, gpu_mem = _gpu_info(accelerator)
    device_in_use = str(getattr(agent.model, "device", "?"))

    return [
        ("PyTorch", torch.__version__ if torch else "N/A"),
        ("Accelerator", accelerator),
        ("GPU", gpu_name),
        ("GPU Mem", gpu_mem),
        ("Device in use", device_in_use),
    ]


def format_system_info(agent: Optional[Agent] = None) -> str:
    rows = _base_rows() + _ai_rows(agent)
    label_width = max(len(label) for label, _ in rows)
    lines = [f"{label:<{label_width}} : {value}" for label, value in rows]
    return format_box("System Info", lines)