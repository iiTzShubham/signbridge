"""Small helpers: make runs repeatable (seeds), pick the device (CUDA GPU / Apple MPS / CPU), and
measure peak memory.
"""
import os
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def select_device(requested="auto"):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS was requested but is unavailable in this PyTorch/macOS installation.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_memory_gib(device, peak_rss_bytes=None, peak_accelerator_bytes=None):
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3)
    if device.type == "mps" and peak_accelerator_bytes is not None:
        return round(peak_accelerator_bytes / (1024 ** 3), 3)
    if peak_rss_bytes is not None:
        return round(peak_rss_bytes / (1024 ** 3), 3)
    return None
