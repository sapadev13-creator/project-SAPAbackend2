"""Deteksi perangkat inferensi: CUDA GPU → Apple MPS → CPU."""

from __future__ import annotations

import os

import torch


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def detect_inference_device() -> str:
    """
    Pilih device inferensi sesuai hardware.
    Override via env SAPA_DEVICE atau INFERENCE_DEVICE: auto | cpu | cuda | mps | cuda:0
    """
    override = (
        os.getenv("SAPA_DEVICE") or os.getenv("INFERENCE_DEVICE") or "auto"
    ).strip().lower()

    if override not in ("", "auto"):
        if override.startswith("cuda"):
            return override if _cuda_available() else "cpu"
        if override == "mps":
            return "mps" if _mps_available() else "cpu"
        if override.startswith("cpu"):
            return "cpu"

    if _cuda_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def build_device_info(device: str | None = None) -> dict:
    """Metadata perangkat untuk logging dan endpoint health."""
    dev = device or detect_inference_device()
    info: dict = {
        "device": dev,
        "cuda_available": _cuda_available(),
        "mps_available": _mps_available(),
        "using_gpu": dev.startswith("cuda") or dev == "mps",
    }

    if dev.startswith("cuda") and _cuda_available():
        idx = 0
        if ":" in dev:
            try:
                idx = int(dev.split(":", 1)[1])
            except ValueError:
                idx = 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_count"] = torch.cuda.device_count()
    elif dev == "mps":
        info["gpu_name"] = "Apple MPS"
    else:
        info["gpu_name"] = None

    return info
