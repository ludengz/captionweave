"""Small, lazy boundary between media jobs and local speech recognition."""
import platform
import sys
from typing import Protocol


class Backend(Protocol):
    language: str | None
    actual_device: str | None

    def block(self, audio, offset: float) -> dict:
        """Recognize normalized mono 16 kHz samples at a playback offset."""
        ...


def metal_platform_supported():
    if sys.platform != "darwin" or platform.machine().lower() != "arm64":
        return False
    version = platform.mac_ver()[0].split(".")[0]
    return version.isdigit() and int(version) >= 14


def resolve_backend(name="auto", device="auto"):
    if name not in {"auto", "faster-whisper", "mlx-whisper"}:
        raise ValueError(f"Unknown ASR backend: {name}")
    if device not in {"auto", "cpu", "cuda", "metal"}:
        raise ValueError(f"Unknown ASR device: {device}")
    if sys.platform == "darwin" and device == "cuda":
        raise ValueError("CUDA is not supported on macOS; use metal on Apple Silicon or cpu")
    if name == "auto":
        name = "mlx-whisper" if device == "metal" or (device == "auto" and metal_platform_supported()) else "faster-whisper"
    if name == "mlx-whisper":
        if not metal_platform_supported():
            raise ValueError("MLX/Metal requires Apple Silicon, native arm64 Python, and macOS 14 or newer")
        if device not in {"auto", "metal"}:
            raise ValueError("mlx-whisper supports the metal device; use faster-whisper for cpu")
    elif device == "metal":
        raise ValueError("faster-whisper does not support metal; use --backend mlx-whisper")
    return name


def create_backend(options) -> Backend:
    name = resolve_backend(options.get("backend", "auto"), options.get("device", "auto"))
    if name == "mlx-whisper":
        from .mlx_whisper import MlxWhisperBackend
        return MlxWhisperBackend(options)
    from .faster_whisper import FasterWhisperBackend
    return FasterWhisperBackend(options)


def backend_identity(backend):
    """Include adapter revisions and dependencies without importing model packages."""
    identity = getattr(backend, "identity", None)
    if identity is not None:
        value = identity()
        if not isinstance(value, dict) or not value.get("name"):
            raise ValueError("Backend identity must be a JSON object with a nonempty name")
        return value
    cls = type(backend)
    return {"name": f"{cls.__module__}.{cls.__qualname__}", "revision": 1, "versions": {}}


def prepare_runtime(name, device, argv):
    if resolve_backend(name, device) == "mlx-whisper":
        return
    from .faster_whisper import configure_gpu_libraries
    configure_gpu_libraries(device, argv)


def diagnostics():
    from .faster_whisper import diagnostics as inspect_faster_whisper
    from .mlx_whisper import diagnostics as inspect_mlx
    backends = {"faster-whisper": inspect_faster_whisper(), "mlx-whisper": inspect_mlx()}
    selected = resolve_backend()
    devices = ["cpu", "metal"] if metal_platform_supported() else ["cpu"] if sys.platform == "darwin" else ["cpu", "cuda"]
    return {**backends[selected], "backend": selected, "devices": devices, "backends": backends,
            "cuda_runtime": "not_loaded", "metal_runtime": "not_loaded"}
