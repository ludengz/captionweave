"""Faster Whisper adapter; model dependencies are loaded only for recognition."""
import os
import site
import sys
from pathlib import Path

from captionweave.core import SAMPLE_RATE, clip_windows
from .common import package_versions as inspect_versions, quality_flags
from . import resolve_backend


DLL_HANDLES = []
DLL_PATHS = set()
PACKAGES = ("faster-whisper", "ctranslate2", "numpy")


def package_versions():
    return inspect_versions(PACKAGES)


def diagnostics():
    return {"backend": "faster-whisper", "packages": package_versions(),
            "devices": ["cpu"] if sys.platform == "darwin" else ["cpu", "cuda"], "cuda_runtime": "not_loaded"}


def configure_gpu_libraries(device, argv=None):
    if device == "cpu" or sys.platform == "darwin":
        return
    paths = []
    for directory in [*site.getsitepackages(), site.getusersitepackages()]:
        for pattern in ("nvidia/*/lib", "nvidia/*/bin"):
            paths.extend(str(path) for path in Path(directory).glob(pattern) if path.is_dir())
    paths = list(dict.fromkeys(paths))
    if os.name == "nt":
        for path in paths:
            if path not in DLL_PATHS:
                DLL_HANDLES.append(os.add_dll_directory(path))
                DLL_PATHS.add(path)
    elif sys.platform.startswith("linux") and paths and argv is not None:
        if os.environ.get("CAPTIONWEAVE_CUDA_PATH_READY"):
            return
        environment = dict(os.environ)
        existing = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(paths + ([existing] if existing else []))
        environment["CAPTIONWEAVE_CUDA_PATH_READY"] = "1"
        # A module command works for both python -m and installed console scripts.
        os.execve(sys.executable, [sys.executable, "-m", "captionweave", *argv], environment)


class FasterWhisperBackend:
    def __init__(self, options):
        resolve_backend("faster-whisper", options["device"])
        self.options = dict(options)
        self.model = None
        self.pipeline = None
        self.language = None if options["language"] == "auto" else options["language"]
        self.actual_device = None

    def identity(self):
        versions = package_versions()
        missing = [name for name, version in versions.items() if version is None]
        if missing:
            raise RuntimeError("Install ASR dependencies with python -m pip install 'captionweave[asr]' "
                               f"(missing: {', '.join(missing)})")
        return {"name": "faster-whisper", "revision": 1, "versions": versions}

    def load(self):
        if self.model is not None:
            return
        configure_gpu_libraries(self.options["device"])
        import ctranslate2
        from faster_whisper import BatchedInferencePipeline, WhisperModel
        device = self.options["device"]
        if device == "auto":
            device = "cuda" if sys.platform != "darwin" and ctranslate2.get_cuda_device_count() else "cpu"
        compute = self.options["compute_type"] or ("float16" if device == "cuda" else "int8")
        print(f"Loading {self.options['model']} on {device} ({compute})", flush=True)
        self.model = WhisperModel(self.options["model"], device=device, compute_type=compute,
                                  local_files_only=self.options["offline"])
        self.pipeline = BatchedInferencePipeline(self.model)
        self.actual_device = device

    @staticmethod
    def records(segments, offset):
        rows = []
        for segment in segments:
            if not segment.text.strip():
                continue
            row = {"start": round(offset + segment.start, 3), "end": round(offset + segment.end, 3),
                   "text": segment.text.strip(), "avg_logprob": segment.avg_logprob,
                   "no_speech_prob": segment.no_speech_prob, "compression_ratio": segment.compression_ratio,
                   "words": [[round(offset + w.start, 3), round(offset + w.end, 3), w.word,
                              round(w.probability, 4)] for w in segment.words or []]}
            row["flags"] = quality_flags(row)
            rows.append(row)
        return rows

    def _decode_window(self, audio, offset):
        segments, _ = self.model.transcribe(
            audio, language=self.language, vad_filter=False, beam_size=5, word_timestamps=True,
            condition_on_previous_text=False, temperature=(0.0, 0.2), hallucination_silence_threshold=2.0)
        return self.records(segments, offset)

    def _prepare_audio(self, audio):
        import numpy as np
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) < SAMPLE_RATE // 10 or np.max(np.abs(audio)) < 1e-5:
            return None
        self.load()
        return audio

    def block(self, audio, offset):
        audio = self._prepare_audio(audio)
        if audio is None:
            return {"segments": [], "alternatives": [], "digital_silence": True, "language": self.language}
        segments, info = self.pipeline.transcribe(
            audio, language=self.language, vad_filter=False,
            clip_timestamps=clip_windows(len(audio) / SAMPLE_RATE), batch_size=self.options["batch_size"],
            beam_size=5, word_timestamps=True, without_timestamps=False, condition_on_previous_text=False)
        primary = self.records(segments, offset)
        self.language = info.language
        alternatives = []
        if self.options["mode"] == "accurate":
            windows = sorted({int(max(0, r["start"] - offset) // 30) for r in primary if r["flags"]})
            for index in windows:
                left, right = index * 30, min((index + 1) * 30, len(audio) / SAMPLE_RATE)
                if right - left < 0.1:
                    continue
                alternatives.extend(self._decode_window(
                    audio[round(left * SAMPLE_RATE):round(right * SAMPLE_RATE)], offset + left))
        return {"segments": primary, "alternatives": alternatives, "digital_silence": False,
                "language": self.language}

    def recheck(self, audio, offset):
        audio = self._prepare_audio(audio)
        if audio is None:
            return []
        return self._decode_window(audio, offset)
