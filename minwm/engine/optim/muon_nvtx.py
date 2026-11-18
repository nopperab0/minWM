"""Optional asynchronous NVTX ranges for Muon profiling."""

import os
from contextlib import contextmanager

import torch


@contextmanager
def muon_range(name: str):
    """Mark an operation without synchronizing CUDA streams."""
    enabled = os.getenv("MINWM_MUON_NVTX") == "1" and torch.cuda.is_available()
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()
