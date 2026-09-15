"""Apple Silicon speech recognition using MLX on a scoped Metal stream."""
from pathlib import Path

from captionweave.core import SAMPLE_RATE, clip_windows
from . import resolve_backend, metal_platform_supported
from .common import package_versions, quality_flags


PACKAGES = ("mlx-whisper", "mlx", "mlx-metal", "numpy", "huggingface-hub", "tiktoken", "numba", "scipy", "torch")
MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-turbo",
}


def diagnostics():
    return {"backend": "mlx-whisper", "packages": package_versions(PACKAGES),
            "devices": ["metal"] if metal_platform_supported() else [], "metal_runtime": "not_loaded"}


def model_source(value):
    path = Path(value).expanduser()
    if path.is_dir():
        return str(path.resolve())
    if value in MODEL_REPOS:
        return MODEL_REPOS[value]
    if len(value.split("/")) == 2 and not value.startswith((".", "/", "~")):
        return value
    raise ValueError(f"Unknown MLX model: {value}; use a supported short name, a Hub repository, or a local MLX directory")


def has_audio(audio):
    return len(audio) >= SAMPLE_RATE // 10 and max(map(abs, audio), default=0) >= 1e-5


class MlxWhisperBackend:
    def __init__(self, options):
        resolve_backend("mlx-whisper", options["device"])
        if options.get("compute_type") not in {None, "float16"}:
            raise ValueError("mlx-whisper supports --compute-type float16 only")
        self.options = dict(options)
        self.model_source = model_source(options["model"])
        self.language = None if options["language"] == "auto" else options["language"]
        self.actual_device = None
        self.model_path = None
        self._mx = None
        self._transcribe = None

    def identity(self):
        versions = package_versions(PACKAGES)
        missing = [name for name, version in versions.items() if version is None]
        if missing:
            raise RuntimeError("Install Metal dependencies with python -m pip install 'captionweave[metal]' "
                               f"(missing: {', '.join(missing)})")
        return {"name": "mlx-whisper", "revision": 1, "versions": versions,
                "model_source": self.model_source}

    def load(self):
        if self._transcribe is not None:
            return
        import mlx.core as mx
        import mlx_whisper
        try:
            if not mx.metal.is_available():
                raise RuntimeError("MLX was built without Metal")
            with mx.stream(mx.gpu):
                mx.eval(mx.array([1.0], dtype=mx.float32) + 1)
        except RuntimeError as error:
            raise RuntimeError("Metal GPU execution is unavailable; use --backend faster-whisper --device cpu") from error
        path = Path(self.model_source)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(
                repo_id=self.model_source, local_files_only=self.options["offline"],
                allow_patterns=["config.json", "weights.safetensors", "weights.npz"]))
        if not (path / "config.json").is_file() or not any(
                (path / name).is_file() for name in ["weights.safetensors", "weights.npz"]):
            raise ValueError("Incomplete MLX model: config.json and weights.safetensors or weights.npz are required")
        self.model_path = str(path.resolve())
        self._mx = mx
        self._transcribe = mlx_whisper.transcribe
        self.actual_device = "metal"
        print(f"Loading {self.model_source} on metal (float16)", flush=True)

    @staticmethod
    def records(segments, offset):
        rows = []
        for segment in segments:
            if not segment["text"].strip():
                continue
            row = {"start": round(float(offset + segment["start"]), 3), "end": round(float(offset + segment["end"]), 3),
                   "text": segment["text"].strip(),
                   "words": [[round(float(offset + word["start"]), 3), round(float(offset + word["end"]), 3),
                              word["word"], round(float(word["probability"]), 4)] for word in segment.get("words", [])]}
            for name in ["avg_logprob", "no_speech_prob", "compression_ratio"]:
                if name in segment:
                    row[name] = float(segment[name])
            row["flags"] = quality_flags(row)
            rows.append(row)
        return rows

    def _decode_window(self, audio, offset, recheck=False):
        self.load()
        import numpy as np
        samples = np.asarray(audio, dtype=np.float32)
        with self._mx.stream(self._mx.gpu):
            result = self._transcribe(
                samples, path_or_hf_repo=self.model_path, language=self.language, task="transcribe",
                verbose=None, fp16=True, word_timestamps=True, without_timestamps=False, condition_on_previous_text=False,
                temperature=(0.0, 0.2) if recheck else 0.0, best_of=5,
                clip_timestamps=[0.0, len(samples) / SAMPLE_RATE],
                hallucination_silence_threshold=2.0 if recheck else None)
        self.language = result.get("language") or self.language
        return self.records(result["segments"], offset)

    def block(self, audio, offset):
        primary, alternatives = [], []
        digital_silence = True
        for window in clip_windows(len(audio) / SAMPLE_RATE):
            left, right = window["start"], window["end"]
            samples = audio[round(left * SAMPLE_RATE):round(right * SAMPLE_RATE)]
            if not has_audio(samples):
                continue
            digital_silence = False
            rows = self._decode_window(samples, offset + left)
            primary.extend(rows)
            if self.options["mode"] == "accurate" and any(row["flags"] for row in rows):
                alternatives.extend(self._decode_window(samples, offset + left, recheck=True))
        return {"segments": primary, "alternatives": alternatives,
                "digital_silence": digital_silence, "language": self.language}

    def recheck(self, audio, offset):
        if not has_audio(audio):
            return []
        return self._decode_window(audio, offset, recheck=True)
