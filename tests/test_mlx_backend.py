import contextlib
import sys
import tempfile
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

OPTIONS = {"backend": "mlx-whisper", "model": "large-v3", "language": "auto", "device": "metal",
           "compute_type": None, "offline": True, "batch_size": 8, "mode": "balanced"}


class MlxBackendTests(unittest.TestCase):
    def backend(self, **overrides):
        from captionweave.backends.mlx_whisper import MlxWhisperBackend
        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"), patch(
                "platform.mac_ver", return_value=("14.0", ("", "", ""), "arm64")):
            return MlxWhisperBackend({**OPTIONS, **overrides})

    @contextlib.contextmanager
    def runtime(self, transcribe=None, metal=True, evaluate=None):
        core = SimpleNamespace(
            gpu="gpu", float32="float32", metal=SimpleNamespace(is_available=lambda: metal),
            stream=Mock(side_effect=lambda device: contextlib.nullcontext()),
            array=Mock(return_value=1.0), eval=evaluate or Mock())
        numpy = SimpleNamespace(float32="float32", asarray=Mock(side_effect=lambda audio, dtype: array("f", audio)))
        whisper = SimpleNamespace(transcribe=transcribe or Mock(return_value={"language": "en", "segments": []}))
        with patch.dict(sys.modules, {"mlx": SimpleNamespace(core=core), "mlx.core": core,
                                      "mlx_whisper": whisper, "numpy": numpy}):
            yield core, numpy, whisper

    def test_supported_precision_and_platform_fail_before_loading_dependencies(self):
        for precision in ["float32", "int8", "int8_float16"]:
            with self.subTest(precision=precision), self.assertRaisesRegex(ValueError, "float16"):
                self.backend(compute_type=precision)
        with self.assertRaisesRegex(ValueError, "metal"):
            self.backend(device="cpu")

    def test_identity_has_model_mapping_and_dependency_versions_without_loading(self):
        backend = self.backend()
        with patch("importlib.metadata.version", return_value="test"), patch.dict(sys.modules, {"mlx": None}):
            identity = backend.identity()
        self.assertEqual(identity["name"], "mlx-whisper")
        self.assertEqual(identity["model_source"], "mlx-community/whisper-large-v3-mlx")
        self.assertIn("mlx-metal", identity["versions"])
        self.assertIsNone(backend.actual_device)

    def test_missing_dependencies_offer_metal_installation(self):
        import importlib.metadata
        backend = self.backend()
        with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
            with self.assertRaisesRegex(RuntimeError, r"captionweave\[metal\]"):
                backend.identity()

    def test_digital_silence_and_short_audio_never_load_mlx_or_change_language(self):
        for samples in [array("f"), array("f", [0.2]) * 100, array("f", [0]) * 16000]:
            backend = self.backend(language="ja")
            with patch.object(backend, "load", side_effect=AssertionError("Unexpected model load")):
                self.assertEqual(backend.block(samples, 120), {
                    "segments": [], "alternatives": [], "digital_silence": True, "language": "ja"})
                self.assertEqual(backend.recheck(samples, 120), [])

    def test_local_model_validation_rejects_non_mlx_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text("{}")
            (folder / "model.bin").write_bytes(b"synthetic")
            backend = self.backend(model=str(folder))
            with self.runtime(), self.assertRaisesRegex(ValueError, "weights"):
                backend.load()

    def test_offline_download_resolves_local_path_and_preserves_supported_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text("{}")
            (folder / "weights.npz").write_bytes(b"synthetic")
            download = Mock(return_value=str(folder))
            backend = self.backend(language="en")
            transcribe = Mock(return_value={"language": "en", "segments": [
                {"start": 0.25, "end": 0.75, "text": " Hello. ", "avg_logprob": -0.2,
                 "no_speech_prob": 0.1, "compression_ratio": 1.0,
                 "words": [{"start": 0.25, "end": 0.75, "word": " Hello", "probability": 0.9}]}]})
            with self.runtime(transcribe) as (mx, np, _), patch.dict(sys.modules, {
                    "huggingface_hub": SimpleNamespace(snapshot_download=download)}):
                rows = backend.recheck(array("f", [0.25]) * 16000, 100.125)
            self.assertTrue(download.call_args.kwargs["local_files_only"])
            self.assertEqual(set(download.call_args.kwargs["allow_patterns"]),
                             {"config.json", "weights.npz", "weights.safetensors"})
            self.assertEqual(transcribe.call_args.kwargs["path_or_hf_repo"], str(folder.resolve()))
            self.assertEqual(transcribe.call_args.kwargs["language"], "en")
            self.assertTrue(transcribe.call_args.kwargs["word_timestamps"])
            self.assertTrue(transcribe.call_args.kwargs["fp16"])
            self.assertFalse(transcribe.call_args.kwargs["condition_on_previous_text"])
            self.assertNotIn("beam_size", transcribe.call_args.kwargs)
            self.assertNotIn("device", transcribe.call_args.kwargs)
            self.assertNotIn("local_files_only", transcribe.call_args.kwargs)
            self.assertTrue(all(call.args == (mx.gpu,) for call in mx.stream.call_args_list))
            mx.eval.assert_called()
            self.assertEqual(backend.actual_device, "metal")
            self.assertEqual((rows[0]["start"], rows[0]["end"]), (100.375, 100.875))
            self.assertEqual(rows[0]["words"], [[100.375, 100.875, " Hello", 0.9]])
            self.assertEqual(rows[0]["flags"], [])

    def test_missing_offline_weights_do_not_fall_through_to_remote_transcribe(self):
        with tempfile.TemporaryDirectory() as temporary:
            download = Mock(return_value=temporary)
            backend = self.backend()
            with self.runtime() as (_, _, whisper), patch.dict(sys.modules, {
                    "huggingface_hub": SimpleNamespace(snapshot_download=download)}):
                with self.assertRaisesRegex(ValueError, "MLX model"):
                    backend.recheck(array("f", [0.25]) * 16000, 0)
                whisper.transcribe.assert_not_called()

    def test_no_metal_device_fails_before_model_download(self):
        for available, evaluate in [(False, None), (True, Mock(side_effect=RuntimeError("No GPU device")))]:
            with self.subTest(available=available), self.runtime(metal=available, evaluate=evaluate), patch.dict(
                    sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=Mock(side_effect=AssertionError))}):
                backend = self.backend()
                with self.assertRaisesRegex(RuntimeError, "Metal"):
                    backend.load()
                self.assertIsNone(backend.actual_device)

    def test_full_coverage_windows_and_flagged_recheck_keep_absolute_timeline(self):
        backend = self.backend(mode="accurate")
        calls = []

        def decode(samples, offset, recheck=False):
            calls.append((len(samples), offset, recheck))
            backend.language = "es"
            return [{"start": offset + 0.1, "end": offset + 0.2, "text": "Hola.", "words": [],
                     "flags": ["low_confidence"] if offset == 30.125 and not recheck else []}]

        with patch.object(backend, "_decode_window", side_effect=decode):
            result = backend.block(array("f", [0.25]) * (16000 * 61), 0.125)
        self.assertEqual(calls, [(480000, 0.125, False), (480000, 30.125, False),
                                 (480000, 30.125, True), (16000, 60.125, False)])
        self.assertEqual([r["start"] for r in result["segments"]], [0.225, 30.225, 60.225])
        self.assertEqual(len(result["alternatives"]), 1)
        self.assertEqual(result["language"], "es")
        self.assertFalse(result["digital_silence"])

    def test_detected_language_flows_to_subsequent_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text("{}")
            (folder / "weights.safetensors").write_bytes(b"synthetic")
            backend = self.backend(model=str(folder))
            with self.runtime() as (_, _, whisper):
                whisper.transcribe.side_effect = [
                    {"language": "es", "segments": []}, {"language": "es", "segments": []}]
                backend.block(array("f", [0.25]) * (16000 * 31), 0)
                self.assertEqual([call.kwargs["language"] for call in whisper.transcribe.call_args_list], [None, "es"])
                backend.block(array("f", [0]) * 16000, 31)
                self.assertEqual(backend.language, "es")


if __name__ == "__main__":
    unittest.main()
