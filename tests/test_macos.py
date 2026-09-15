import contextlib
import sys
import unittest
from unittest.mock import Mock, patch

from captionweave.backends import create_backend, prepare_runtime
from captionweave.backends.faster_whisper import FasterWhisperBackend, configure_gpu_libraries
from captionweave.cli import build_parser, doctor


@contextlib.contextmanager
def mac(machine="arm64", version="14.0"):
    with patch("sys.platform", "darwin"), patch("platform.machine", return_value=machine), patch(
            "platform.mac_ver", return_value=(version, ("", "", ""), machine)):
        yield


OPTIONS = {"backend": "auto", "model": "large-v3", "language": "auto", "device": "auto",
           "compute_type": None, "offline": True, "batch_size": 2, "mode": "accurate"}


class MacRoutingTests(unittest.TestCase):
    def test_cli_can_select_automatic_or_explicit_metal_backend(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["run", "example.wav"]).backend, "auto")
        args = parser.parse_args(["run", "example.wav", "--backend", "mlx-whisper", "--device", "metal"])
        self.assertEqual((args.backend, args.device), ("mlx-whisper", "metal"))
        args = parser.parse_args(["recheck", "job", "--backend", "mlx-whisper", "--device", "metal"])
        self.assertEqual((args.backend, args.device), ("mlx-whisper", "metal"))

    def test_apple_silicon_selects_mlx_without_loading_any_asr_runtime(self):
        with mac(), patch.dict(sys.modules, {"mlx": None, "mlx_whisper": None, "ctranslate2": None}):
            backend = create_backend(OPTIONS)
        self.assertEqual(type(backend).__name__, "MlxWhisperBackend")
        self.assertIsNone(backend.actual_device)

    def test_intel_and_older_macos_select_cpu_backend(self):
        for machine, version in [("x86_64", "14.0"), ("arm64", "13.6")]:
            with self.subTest(machine=machine, version=version), mac(machine, version):
                self.assertIsInstance(create_backend(OPTIONS), FasterWhisperBackend)

    def test_explicit_cpu_uses_faster_whisper_on_apple_silicon(self):
        with mac():
            self.assertIsInstance(create_backend({**OPTIONS, "device": "cpu"}), FasterWhisperBackend)

    def test_macos_rejects_cuda_before_preparing_or_loading_runtime(self):
        with mac(), self.assertRaisesRegex(ValueError, "CUDA.*macOS"):
            create_backend({**OPTIONS, "device": "cuda"})
        with mac(), self.assertRaisesRegex(ValueError, "CUDA.*macOS"):
            prepare_runtime("faster-whisper", "cuda", [])

    def test_metal_rejects_non_apple_platforms_and_incompatible_backend(self):
        with patch("sys.platform", "linux"), self.assertRaisesRegex(ValueError, "Apple Silicon"):
            create_backend({**OPTIONS, "device": "metal"})
        with mac("x86_64"), self.assertRaisesRegex(ValueError, "Apple Silicon"):
            create_backend({**OPTIONS, "backend": "mlx-whisper"})
        with mac(), self.assertRaisesRegex(ValueError, "faster-whisper.*metal"):
            create_backend({**OPTIONS, "backend": "faster-whisper", "device": "metal"})

    def test_macos_never_scans_nvidia_libraries(self):
        with mac(), patch("captionweave.backends.faster_whisper.site.getsitepackages", side_effect=AssertionError):
            configure_gpu_libraries("auto")
            prepare_runtime("auto", "auto", [])

    def test_faster_whisper_auto_on_mac_does_not_probe_cuda(self):
        from types import SimpleNamespace
        model = Mock()
        modules = {"ctranslate2": SimpleNamespace(get_cuda_device_count=Mock(side_effect=AssertionError)),
                   "faster_whisper": SimpleNamespace(WhisperModel=model, BatchedInferencePipeline=lambda value: value)}
        with mac(), patch.dict(sys.modules, modules):
            backend = FasterWhisperBackend({**OPTIONS, "backend": "faster-whisper"})
            backend.load()
        self.assertEqual(model.call_args.kwargs["device"], "cpu")
        self.assertEqual(model.call_args.kwargs["compute_type"], "int8")
        self.assertEqual(backend.actual_device, "cpu")

    def test_doctor_reports_metal_selection_without_loading_gpu_runtime(self):
        with mac(), patch.dict(sys.modules, {"mlx": None, "mlx_whisper": None}):
            result = doctor()
        self.assertEqual(result["backend"], "mlx-whisper")
        self.assertEqual(result["devices"], ["cpu", "metal"])
        self.assertEqual(result["metal_runtime"], "not_loaded")
        self.assertIn("mlx-whisper", result["backends"])


if __name__ == "__main__":
    unittest.main()
