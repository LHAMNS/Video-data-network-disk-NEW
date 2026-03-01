"""
Cross-platform hardware detection and acceleration support.

Supports:
  - NVIDIA GPUs (CUDA / NVENC) on Linux, Windows, macOS
  - AMD GPUs (ROCm / AMF / OpenCL) on Linux, Windows
  - Intel GPUs (OneAPI / QSV / VAAPI) on Linux, Windows
  - Apple M-series (Metal / VideoToolbox) on macOS
  - Multi-core CPU acceleration (Numba / multiprocessing) everywhere

The module auto-detects the best available backend and exposes a unified
HardwareAccelerator interface so the rest of the codebase does not need
platform-specific branches.
"""

import os
import sys
import logging
import subprocess
import platform
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    name: str = "Unknown"
    vendor: str = "unknown"            # nvidia, amd, intel, apple
    memory_gb: float = 0.0
    compute_backend: str = "none"      # cuda, rocm, metal, opencl, none
    video_encode_api: str = "none"     # nvenc, amf, qsv, videotoolbox, vaapi, none
    device_index: int = 0
    compute_capability: str = ""


@dataclass
class CPUInfo:
    """Information about the CPU."""
    name: str = "Unknown"
    vendor: str = "unknown"            # intel, amd, apple, arm
    cores_physical: int = 1
    cores_logical: int = 1
    architecture: str = ""             # x86_64, aarch64, arm64
    features: List[str] = field(default_factory=list)  # sse, avx, avx2, avx512, neon


@dataclass
class HardwareProfile:
    """Complete hardware profile for the current system."""
    platform: str = ""                 # linux, windows, darwin
    cpu: CPUInfo = field(default_factory=CPUInfo)
    gpus: List[GPUInfo] = field(default_factory=list)
    best_gpu: Optional[GPUInfo] = None
    ffmpeg_encoders: List[str] = field(default_factory=list)
    ffmpeg_decoders: List[str] = field(default_factory=list)
    recommended_backend: str = "cpu"   # cuda, rocm, metal, opencl, cpu
    recommended_video_api: str = "none"


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _run_cmd(cmd: list, timeout: int = 10) -> Optional[str]:
    """Run a command and return stdout, or None on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout
        return None
    except Exception:
        return None


def _detect_platform() -> str:
    s = sys.platform
    if s.startswith("linux"):
        return "linux"
    if s == "darwin":
        return "darwin"
    if s in ("win32", "cygwin"):
        return "windows"
    return s


def _detect_cpu() -> CPUInfo:
    info = CPUInfo()
    info.cores_physical = mp.cpu_count() or 1
    info.cores_logical = mp.cpu_count() or 1
    info.architecture = platform.machine()

    plat = _detect_platform()

    # Try to get CPU brand string
    try:
        if plat == "linux":
            out = _run_cmd(["lscpu"])
            if out:
                for line in out.splitlines():
                    if "Model name:" in line:
                        info.name = line.split(":", 1)[1].strip()
                    if "Thread(s) per core:" in line:
                        tpc = int(line.split(":", 1)[1].strip())
                        info.cores_physical = max(1, info.cores_logical // max(1, tpc))
                    if "Flags:" in line or "flags:" in line.lower():
                        flags = line.split(":", 1)[1].strip().split()
                        for feat in ("sse", "sse2", "avx", "avx2", "avx512f", "neon"):
                            if feat in flags:
                                info.features.append(feat)
        elif plat == "darwin":
            out = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
            if out:
                info.name = out.strip()
            pcount = _run_cmd(["sysctl", "-n", "hw.physicalcpu"])
            if pcount:
                info.cores_physical = int(pcount.strip())
            # M-series detection
            if "Apple" in info.name or info.architecture == "arm64":
                info.vendor = "apple"
                info.features.append("neon")
            elif "Intel" in info.name:
                info.vendor = "intel"
            elif "AMD" in info.name:
                info.vendor = "amd"
        elif plat == "windows":
            out = _run_cmd(["wmic", "cpu", "get", "Name", "/value"])
            if out:
                for line in out.splitlines():
                    if line.startswith("Name="):
                        info.name = line.split("=", 1)[1].strip()
    except Exception as e:
        logger.debug(f"CPU detection detail error: {e}")

    # Vendor heuristic
    name_lower = info.name.lower()
    if info.vendor == "unknown":
        if "intel" in name_lower:
            info.vendor = "intel"
        elif "amd" in name_lower or "ryzen" in name_lower or "epyc" in name_lower:
            info.vendor = "amd"
        elif "apple" in name_lower:
            info.vendor = "apple"

    return info


def _detect_nvidia_gpu() -> Optional[GPUInfo]:
    """Detect NVIDIA GPU via nvidia-smi or CUDA."""
    # Try nvidia-smi first (works on Linux, Windows, macOS with drivers)
    out = _run_cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if out:
        for i, line in enumerate(out.strip().splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                gpu = GPUInfo(
                    name=parts[0],
                    vendor="nvidia",
                    memory_gb=float(parts[1]) / 1024.0,
                    compute_backend="cuda",
                    video_encode_api="nvenc",
                    device_index=i,
                )
                return gpu

    # Fallback: try CuPy
    try:
        import cupy as cp
        dev = cp.cuda.Device(0)
        gpu = GPUInfo(
            name=dev.name.decode() if hasattr(dev.name, 'decode') else str(dev.name),
            vendor="nvidia",
            memory_gb=dev.mem_info[1] / (1024 ** 3),
            compute_backend="cuda",
            video_encode_api="nvenc",
            device_index=0,
            compute_capability=f"{dev.compute_capability[0]}.{dev.compute_capability[1]}",
        )
        return gpu
    except Exception:
        pass

    return None


def _detect_amd_gpu() -> Optional[GPUInfo]:
    """Detect AMD GPU via rocm-smi or system info."""
    # Try ROCm (Linux)
    out = _run_cmd(["rocm-smi", "--showproductname"])
    if out and "GPU" in out:
        name = "AMD GPU"
        for line in out.splitlines():
            if "GPU" in line and ":" in line:
                name = line.split(":", 1)[1].strip()
                break
        mem_out = _run_cmd(["rocm-smi", "--showmeminfo", "vram"])
        mem_gb = 0.0
        if mem_out:
            for line in mem_out.splitlines():
                if "Total" in line:
                    try:
                        val = "".join(c for c in line.split(":")[-1] if c.isdigit() or c == ".")
                        mem_gb = float(val) / (1024 * 1024) if float(val) > 10000 else float(val)
                    except Exception:
                        pass
        return GPUInfo(
            name=name,
            vendor="amd",
            memory_gb=mem_gb,
            compute_backend="rocm",
            video_encode_api="amf",
        )

    # Windows: try WMIC or DX diag
    plat = _detect_platform()
    if plat == "windows":
        out = _run_cmd(["wmic", "path", "win32_videocontroller", "get", "Name,AdapterRAM", "/value"])
        if out and "AMD" in out.upper():
            name = "AMD GPU"
            mem_gb = 0.0
            for line in out.splitlines():
                if line.startswith("Name=") and "AMD" in line.upper():
                    name = line.split("=", 1)[1].strip()
                if line.startswith("AdapterRAM="):
                    try:
                        mem_gb = int(line.split("=", 1)[1].strip()) / (1024 ** 3)
                    except Exception:
                        pass
            return GPUInfo(
                name=name, vendor="amd", memory_gb=mem_gb,
                compute_backend="opencl", video_encode_api="amf",
            )

    return None


def _detect_intel_gpu() -> Optional[GPUInfo]:
    """Detect Intel integrated or discrete GPU."""
    plat = _detect_platform()

    if plat == "linux":
        # Check for Intel GPU via lspci
        out = _run_cmd(["lspci"])
        if out:
            for line in out.splitlines():
                if "VGA" in line and ("Intel" in line or "intel" in line):
                    name = line.split(":", 2)[-1].strip() if ":" in line else "Intel GPU"
                    return GPUInfo(
                        name=name, vendor="intel",
                        compute_backend="opencl", video_encode_api="qsv",
                    )
    elif plat == "windows":
        out = _run_cmd(["wmic", "path", "win32_videocontroller", "get", "Name", "/value"])
        if out and "Intel" in out:
            name = "Intel GPU"
            for line in out.splitlines():
                if line.startswith("Name=") and "Intel" in line:
                    name = line.split("=", 1)[1].strip()
            return GPUInfo(
                name=name, vendor="intel",
                compute_backend="opencl", video_encode_api="qsv",
            )

    return None


def _detect_apple_gpu() -> Optional[GPUInfo]:
    """Detect Apple M-series GPU (macOS only)."""
    if _detect_platform() != "darwin":
        return None

    arch = platform.machine()
    if arch != "arm64":
        return None

    # M-series GPU info from system_profiler
    out = _run_cmd(["system_profiler", "SPDisplaysDataType"])
    if out:
        name = "Apple Silicon GPU"
        mem_gb = 0.0
        for line in out.splitlines():
            line = line.strip()
            if "Chipset Model:" in line:
                name = line.split(":", 1)[1].strip()
            if "VRAM" in line:
                try:
                    val = "".join(c for c in line.split(":")[-1] if c.isdigit() or c == ".")
                    mem_gb = float(val)
                except Exception:
                    pass

        # Apple Silicon shares system memory
        if mem_gb == 0:
            mem_out = _run_cmd(["sysctl", "-n", "hw.memsize"])
            if mem_out:
                try:
                    mem_gb = int(mem_out.strip()) / (1024 ** 3)
                except Exception:
                    pass

        return GPUInfo(
            name=name, vendor="apple", memory_gb=mem_gb,
            compute_backend="metal", video_encode_api="videotoolbox",
        )

    return None


def _detect_ffmpeg_capabilities() -> tuple:
    """Detect available FFmpeg encoders and decoders."""
    encoders = []
    decoders = []

    enc_out = _run_cmd(["ffmpeg", "-encoders", "-hide_banner"])
    if enc_out:
        hw_encoders = [
            "h264_nvenc", "hevc_nvenc",       # NVIDIA
            "h264_amf", "hevc_amf",           # AMD
            "h264_qsv", "hevc_qsv",           # Intel QSV
            "h264_videotoolbox", "hevc_videotoolbox",  # Apple
            "h264_vaapi", "hevc_vaapi",        # VAAPI (Linux)
            "rawvideo",                         # Software fallback
        ]
        for enc in hw_encoders:
            if enc in enc_out:
                encoders.append(enc)

    dec_out = _run_cmd(["ffmpeg", "-decoders", "-hide_banner"])
    if dec_out:
        hw_decoders = [
            "h264_cuvid", "hevc_cuvid",       # NVIDIA
            "h264_qsv", "hevc_qsv",           # Intel
            "h264_v4l2m2m",                    # Linux V4L2
            "rawvideo",
        ]
        for dec in hw_decoders:
            if dec in dec_out:
                decoders.append(dec)

    return encoders, decoders


# ---------------------------------------------------------------------------
# Main detection
# ---------------------------------------------------------------------------

_cached_profile: Optional[HardwareProfile] = None


def detect_hardware(force_refresh: bool = False) -> HardwareProfile:
    """
    Detect all available hardware and return a HardwareProfile.
    Results are cached after first call.
    """
    global _cached_profile
    if _cached_profile is not None and not force_refresh:
        return _cached_profile

    logger.info("Detecting hardware capabilities...")

    profile = HardwareProfile()
    profile.platform = _detect_platform()
    profile.cpu = _detect_cpu()

    # Detect GPUs (try all vendors)
    for detector in (_detect_nvidia_gpu, _detect_amd_gpu, _detect_intel_gpu, _detect_apple_gpu):
        try:
            gpu = detector()
            if gpu:
                profile.gpus.append(gpu)
                logger.info(f"Detected GPU: {gpu.name} ({gpu.vendor}, {gpu.compute_backend}, "
                           f"{gpu.memory_gb:.1f} GB, encode: {gpu.video_encode_api})")
        except Exception as e:
            logger.debug(f"GPU detection error in {detector.__name__}: {e}")

    # Pick best GPU
    backend_priority = {"cuda": 4, "rocm": 3, "metal": 3, "opencl": 2, "none": 0}
    if profile.gpus:
        profile.best_gpu = max(profile.gpus, key=lambda g: backend_priority.get(g.compute_backend, 0))
        profile.recommended_backend = profile.best_gpu.compute_backend
        profile.recommended_video_api = profile.best_gpu.video_encode_api
    else:
        profile.recommended_backend = "cpu"

    # Detect FFmpeg hw acceleration
    try:
        profile.ffmpeg_encoders, profile.ffmpeg_decoders = _detect_ffmpeg_capabilities()
    except Exception as e:
        logger.debug(f"FFmpeg detection error: {e}")

    # Log summary
    logger.info(f"Platform: {profile.platform}, CPU: {profile.cpu.name} "
               f"({profile.cpu.cores_physical}C/{profile.cpu.cores_logical}T, {profile.cpu.vendor})")
    logger.info(f"GPUs found: {len(profile.gpus)}, Recommended: {profile.recommended_backend}")
    logger.info(f"FFmpeg encoders: {profile.ffmpeg_encoders}")

    _cached_profile = profile
    return profile


# ---------------------------------------------------------------------------
# Acceleration helpers
# ---------------------------------------------------------------------------

def get_optimal_thread_count(profile: Optional[HardwareProfile] = None) -> int:
    """Get optimal number of worker threads for the current hardware."""
    if profile is None:
        profile = detect_hardware()

    cores = profile.cpu.cores_physical
    # Leave 1 core free for the main thread and OS
    return max(1, cores - 1)


def get_compute_array_module(profile: Optional[HardwareProfile] = None):
    """
    Get the best available array computation module.

    Returns:
        Module compatible with numpy API (cupy for NVIDIA, numpy for CPU)
    """
    if profile is None:
        profile = detect_hardware()

    backend = profile.recommended_backend

    if backend == "cuda":
        try:
            import cupy as cp
            logger.info("Using CuPy (CUDA) for array computation")
            return cp
        except ImportError:
            logger.warning("CuPy not available, falling back to NumPy")

    if backend == "rocm":
        try:
            import cupy as cp  # CuPy also supports ROCm
            logger.info("Using CuPy (ROCm) for array computation")
            return cp
        except ImportError:
            logger.warning("CuPy (ROCm) not available, falling back to NumPy")

    import numpy as np
    logger.info("Using NumPy (CPU) for array computation")
    return np


def get_ffmpeg_encode_args(profile: Optional[HardwareProfile] = None) -> List[str]:
    """
    Get FFmpeg encoding arguments for the best available hw encoder.
    Returns args for uncompressed/raw video as primary (since we write AVI directly),
    but also provides the hw-accelerated preset if needed for re-encoding.
    """
    if profile is None:
        profile = detect_hardware()

    # For our use case (uncompressed AVI), we don't need FFmpeg encoding.
    # But provide hw-accelerated args for verification/repair steps.
    encoders = profile.ffmpeg_encoders

    if "h264_nvenc" in encoders:
        return ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll"]
    if "h264_amf" in encoders:
        return ["-c:v", "h264_amf", "-quality", "speed"]
    if "h264_qsv" in encoders:
        return ["-c:v", "h264_qsv", "-preset", "veryfast"]
    if "h264_videotoolbox" in encoders:
        return ["-c:v", "h264_videotoolbox", "-realtime", "1"]
    if "h264_vaapi" in encoders:
        return ["-c:v", "h264_vaapi"]

    # Software fallback
    return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "0"]


def get_hardware_summary(profile: Optional[HardwareProfile] = None) -> Dict[str, Any]:
    """Get a JSON-serializable hardware summary for the web UI."""
    if profile is None:
        profile = detect_hardware()

    return {
        "platform": profile.platform,
        "cpu": {
            "name": profile.cpu.name,
            "vendor": profile.cpu.vendor,
            "cores_physical": profile.cpu.cores_physical,
            "cores_logical": profile.cpu.cores_logical,
            "architecture": profile.cpu.architecture,
            "features": profile.cpu.features,
        },
        "gpus": [
            {
                "name": g.name,
                "vendor": g.vendor,
                "memory_gb": round(g.memory_gb, 1),
                "compute_backend": g.compute_backend,
                "video_encode_api": g.video_encode_api,
            }
            for g in profile.gpus
        ],
        "recommended_backend": profile.recommended_backend,
        "recommended_video_api": profile.recommended_video_api,
        "ffmpeg_encoders": profile.ffmpeg_encoders,
        "ffmpeg_decoders": profile.ffmpeg_decoders,
        "optimal_threads": get_optimal_thread_count(profile),
    }
