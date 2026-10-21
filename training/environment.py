"""Record the computer's hardware and software versions so every result file says where it was measured.
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def _sysctl(key):
    try:
        value = subprocess.check_output(["/usr/sbin/sysctl", "-n", key], text=True, timeout=3).strip()
        return int(value) if value.isdigit() else value
    except (OSError, subprocess.SubprocessError):
        return None


def hardware_info():
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "memory_bytes": _sysctl("hw.memsize"),
        "hardware_model": _sysctl("hw.model"),
        "gpu": None,
        "accelerator": "cpu",
    }
    if sys.platform == "darwin":
        try:
            raw = subprocess.check_output(
                ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"], text=True, timeout=8
            )
            displays = json.loads(raw).get("SPDisplaysDataType", [])
            info["gpu"] = [
                {"name": item.get("sppci_model"), "cores": item.get("sppci_cores")}
                for item in displays if item.get("sppci_model")
            ]
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    try:
        import torch
        if torch.cuda.is_available():
            info["accelerator"] = "cuda"
            info["gpu"] = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["accelerator"] = "mps"
    except ImportError:
        pass
    return info


def software_info():
    info = {"python": platform.python_version()}
    for module_name, package_name in (
        ("django", "Django"), ("torch", "torch"), ("torchvision", "torchvision"),
        ("cv2", "opencv-python-headless"), ("numpy", "numpy"), ("pandas", "pandas"),
        ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"),
        ("docx", "python-docx"), ("scipy", "scipy"),
    ):
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", None)
            if module_name == "django":
                version = getattr(module, "get_version", lambda: version)()
            info[package_name] = version or "installed; version unavailable"
        except ImportError:
            info[package_name] = "not installed"
    return info


def runtime_memory_gib():
    value = _sysctl("hw.memsize")
    return round(value / (1024 ** 3), 1) if isinstance(value, int) else None
