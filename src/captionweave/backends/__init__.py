"""Small, lazy boundary between media jobs and local speech recognition."""
from typing import Protocol


class Backend(Protocol):
    language: str | None
    actual_device: str | None

    def block(self, audio, offset: float) -> dict:
        """Recognize normalized mono 16 kHz samples at a playback offset."""
        ...


def create_backend(options) -> Backend:
    name = options.get("backend", "faster-whisper")
    if name != "faster-whisper":
        raise ValueError(f"Unknown ASR backend: {name}")
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
    if name != "faster-whisper":
        raise ValueError(f"Unknown ASR backend: {name}")
    from .faster_whisper import configure_gpu_libraries
    configure_gpu_libraries(device, argv)


def diagnostics():
    from .faster_whisper import diagnostics as inspect_faster_whisper
    return inspect_faster_whisper()
