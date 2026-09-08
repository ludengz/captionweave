import importlib.metadata
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from captionweave.backends import backend_identity, create_backend
from captionweave.backends.faster_whisper import FasterWhisperBackend, configure_gpu_libraries
from captionweave.core import SAMPLE_RATE, read_audio


OPTIONS = {"backend": "faster-whisper", "model": "not-loaded", "language": "auto", "device": "cpu",
           "compute_type": None, "offline": True, "batch_size": 2, "mode": "balanced"}


class BackendContractTests(unittest.TestCase):
    def test_factory_rejects_unimplemented_backends(self):
        for name in ("mlx", "metal", "unknown"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unknown ASR backend"):
                create_backend({**OPTIONS, "backend": name})

    def test_injected_backend_identity_needs_no_package_metadata(self):
        class First:
            pass

        class Second:
            pass

        with patch("importlib.metadata.version", side_effect=AssertionError("Unexpected package lookup")):
            self.assertNotEqual(backend_identity(First()), backend_identity(Second()))
            self.assertEqual(backend_identity(First()), backend_identity(First()))

    def test_real_backend_identity_includes_installed_versions_without_loading(self):
        versions = {"faster-whisper": "1", "ctranslate2": "2", "numpy": "3"}
        backend = create_backend(OPTIONS)
        with patch("importlib.metadata.version", side_effect=versions.__getitem__):
            identity = backend_identity(backend)
        self.assertEqual(identity, {"name": "faster-whisper", "revision": 1, "versions": versions})
        self.assertIsNone(backend.model)

    def test_missing_dependencies_give_install_extra_instructions(self):
        with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
            with self.assertRaisesRegex(RuntimeError, r"captionweave\[asr\]"):
                backend_identity(create_backend(OPTIONS))

    def test_loading_resolves_auto_device_and_preserves_explicit_compute_type(self):
        calls = []
        model = object()

        def whisper_model(name, **options):
            calls.append((name, options))
            return model

        modules = {"ctranslate2": SimpleNamespace(get_cuda_device_count=lambda: 1),
                   "faster_whisper": SimpleNamespace(WhisperModel=whisper_model, BatchedInferencePipeline=lambda model: model)}
        backend = FasterWhisperBackend({**OPTIONS, "device": "auto", "compute_type": "int8_float16"})
        with patch.dict(sys.modules, modules), patch("captionweave.backends.faster_whisper.configure_gpu_libraries"):
            backend.load()
            backend.load()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], {"device": "cuda", "compute_type": "int8_float16", "local_files_only": True})
        self.assertEqual(backend.actual_device, "cuda")

    def test_audio_reader_uses_normalized_standard_library_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "samples.wav"
            with wave.open(str(path), "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                output.writeframes(b"\x00\x80\x00\x00\xff\x7f")
            samples = read_audio(path, 0, 3 / 16000)
        self.assertIsInstance(samples, array)
        self.assertEqual(samples.typecode, "f")
        self.assertEqual(list(samples), [-1, 0, 32767 / 32768])

    def test_import_doctor_and_custom_jobs_work_with_all_asr_imports_blocked(self):
        source_root = Path(__file__).resolve().parents[1] / "src"
        script = r'''
import importlib.abc
import json
import shutil
import sys
import tempfile
import wave
from pathlib import Path

class BlockASR(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"numpy", "ctranslate2", "faster_whisper", "torch", "pyannote"}:
            raise AssertionError("Unexpected ASR import: " + fullname)

sys.meta_path.insert(0, BlockASR())
from captionweave.cli import doctor
from captionweave.core import transcribe_media
from captionweave.subtitles import render_job
assert doctor()["cuda_runtime"] == "not_loaded"

class Engine:
    language = None
    actual_device = "test"
    def block(self, audio, offset):
        return {"language": "en", "alternatives": [], "segments": [
            {"start": 0.1, "end": 0.4, "text": "Hello.", "flags": []}]}

if shutil.which("ffmpeg") and shutil.which("ffprobe"):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "source.wav"
        with wave.open(str(source), "wb") as output:
            output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            output.writeframes(b"\0\0" * 16000)
        options = {"language": "auto", "start": 0, "duration": None, "audio_stream": 0}
        job = transcribe_media(source, root / "jobs", options, Engine())
        assert render_job(job, root / "output.srt")["status"] == "complete"
'''
        environment = {**os.environ, "PYTHONPATH": str(source_root), "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class BackendBlockTests(unittest.TestCase):
    def setUp(self):
        self.numpy = SimpleNamespace(
            float32=object(), asarray=Mock(side_effect=lambda values, dtype: array("f", values)),
            abs=lambda values: map(abs, values), max=max)
        modules = patch.dict(sys.modules, {"numpy": self.numpy, "faster_whisper": None, "ctranslate2": None})
        modules.start()
        self.addCleanup(modules.stop)

    @staticmethod
    def segment(start, end, text="Hello.", words=None, **diagnostics):
        return SimpleNamespace(start=start, end=end, text=text, words=words,
                               **{"avg_logprob": -0.2, "no_speech_prob": 0.1,
                                  "compression_ratio": 1.0, **diagnostics})

    def backend(self, segments, detected_language="en", **options):
        backend = FasterWhisperBackend({**OPTIONS, **options})
        backend.pipeline = SimpleNamespace(transcribe=Mock(
            return_value=(iter(segments), SimpleNamespace(language=detected_language))))
        backend.model = SimpleNamespace(transcribe=Mock(side_effect=AssertionError("Unexpected alternate recognition")))
        return backend

    def test_digital_silence_and_short_audio_do_not_load_a_model(self):
        samples = {"empty": array("f"), "short": array("f", [0.5]) * (SAMPLE_RATE // 10 - 1),
                   "zero": array("f", [0]) * SAMPLE_RATE,
                   "near_zero": array("f", [-0.000001]) * SAMPLE_RATE}
        for name, audio in samples.items():
            with self.subTest(audio=name):
                backend = FasterWhisperBackend({**OPTIONS, "language": "ja"})
                with patch.object(backend, "load", side_effect=AssertionError("Unexpected model load")) as load:
                    result = backend.block(audio, 120)
                load.assert_not_called()
                self.assertEqual(result, {"segments": [], "alternatives": [],
                                          "digital_silence": True, "language": "ja"})
                self.assertIsNone(backend.model)

    def test_block_preserves_segment_word_offsets_and_full_timeline_clips(self):
        word = SimpleNamespace(start=0.12345, end=0.56789, word=" Hello", probability=0.987654)
        backend = self.backend([self.segment(0.1254, 0.8756, " Hello. ", [word]),
                                self.segment(2, 3, " \t ")])
        audio = array("f", [-0.25]) * (SAMPLE_RATE * 30) + array("f", [0]) * (SAMPLE_RATE // 4)
        result = backend.block(audio, 120)

        self.numpy.asarray.assert_called_once_with(audio, dtype=self.numpy.float32)
        call = backend.pipeline.transcribe.call_args
        self.assertEqual(call.args, (audio,))
        self.assertEqual(call.kwargs, {"language": None, "vad_filter": False,
                                      "clip_timestamps": [{"start": 0.0, "end": 30.0}, {"start": 30.0, "end": 30.25}],
                                      "batch_size": 2, "beam_size": 5, "word_timestamps": True,
                                      "without_timestamps": False, "condition_on_previous_text": False})
        self.assertEqual(result, {"segments": [
            {"start": 120.125, "end": 120.876, "text": "Hello.", "avg_logprob": -0.2,
             "no_speech_prob": 0.1, "compression_ratio": 1.0,
             "words": [[120.123, 120.568, " Hello", 0.9877]], "flags": []}],
            "alternatives": [], "digital_silence": False, "language": "en"})
        backend.model.transcribe.assert_not_called()

    def test_detected_language_is_passed_to_later_blocks_and_survives_silence(self):
        for requested in ("auto", "ja"):
            with self.subTest(requested=requested):
                backend = self.backend([], detected_language="ja", language=requested)
                backend.pipeline.transcribe.side_effect = [
                    (iter([]), SimpleNamespace(language="ja")),
                    (iter([]), SimpleNamespace(language="ja"))]
                audio = array("f", [0.25]) * SAMPLE_RATE
                first = backend.block(audio, 0)
                second = backend.block(audio, 1)
                silence = backend.block(array("f", [0]) * SAMPLE_RATE, 2)
                self.assertEqual([call.kwargs["language"] for call in backend.pipeline.transcribe.call_args_list],
                                 [None if requested == "auto" else "ja", "ja"])
                self.assertEqual([row["language"] for row in (first, second, silence)], ["ja", "ja", "ja"])
                self.assertEqual(backend.language, "ja")
                self.assertTrue(silence["digital_silence"])

    def test_balanced_mode_retains_flagged_primary_without_rechecking(self):
        backend = self.backend([self.segment(0.1, 0.5, avg_logprob=-1.0)])
        result = backend.block(array("f", [0.25]) * SAMPLE_RATE, 30)
        self.assertEqual(result["segments"][0]["flags"], ["low_confidence"])
        self.assertEqual(result["alternatives"], [])
        backend.model.transcribe.assert_not_called()

    def test_accurate_mode_rechecks_each_flagged_window_once_on_original_timeline(self):
        backend = self.backend([
            self.segment(64, 65, "Later.", avg_logprob=-1.0),
            self.segment(1, 2, "First.", avg_logprob=-1.0),
            self.segment(8, 9, "Repeat.", compression_ratio=3.0),
            self.segment(31, 32, "Clear."),
            self.segment(90.5, 91, "Tail.", no_speech_prob=0.8)], mode="accurate")
        word = SimpleNamespace(start=0.125, end=0.5, word=" Check", probability=0.9)
        backend.model.transcribe.side_effect = [
            (iter([self.segment(0.125, 0.5, text, [word])]), SimpleNamespace(language="fr"))
            for text in ("First check.", "Later check.", "Tail check.")]
        windows = [array("f", [value]) * (SAMPLE_RATE * duration)
                   for value, duration in ((0.125, 30), (0.25, 30), (0.375, 30), (0.5, 2))]
        audio = windows[0] + windows[1] + windows[2] + windows[3]
        result = backend.block(audio, 100)

        calls = backend.model.transcribe.call_args_list
        self.assertEqual([call.args for call in calls], [(windows[0],), (windows[2],), (windows[3],)])
        for call in calls:
            self.assertEqual(call.kwargs, {"language": "en", "vad_filter": False, "beam_size": 5,
                                          "word_timestamps": True, "condition_on_previous_text": False,
                                          "temperature": (0.0, 0.2), "hallucination_silence_threshold": 2.0})
        self.assertEqual([row["text"] for row in result["segments"]],
                         ["Later.", "First.", "Repeat.", "Clear.", "Tail."])
        self.assertEqual([row["flags"] for row in result["segments"]],
                         [["low_confidence"], ["low_confidence"], ["repetition"], [], ["possible_non_speech"]])
        self.assertEqual(result["alternatives"], [
            {"start": start, "end": end, "text": text, "avg_logprob": -0.2,
             "no_speech_prob": 0.1, "compression_ratio": 1.0,
             "words": [[start, end, " Check", 0.9]], "flags": []}
            for start, end, text in ((100.125, 100.5, "First check."),
                                     (160.125, 160.5, "Later check."), (190.125, 190.5, "Tail check."))])
        self.assertEqual(result["language"], "en")
        self.assertEqual(backend.language, "en")
        self.assertFalse(result["digital_silence"])

    def test_accurate_mode_skips_a_flagged_tail_under_one_tenth_second(self):
        backend = self.backend([self.segment(60.01, 60.04, avg_logprob=-1.0)], mode="accurate")
        audio = array("f", [0.25]) * (SAMPLE_RATE * 60 + SAMPLE_RATE // 20)
        result = backend.block(audio, 300)
        self.assertEqual(backend.pipeline.transcribe.call_args.kwargs["clip_timestamps"],
                         [{"start": 0.0, "end": 30.0}, {"start": 30.0, "end": 60.0}])
        self.assertEqual(result["segments"][0]["start"], 360.01)
        self.assertEqual(result["segments"][0]["flags"], ["low_confidence"])
        self.assertEqual(result["alternatives"], [])
        backend.model.transcribe.assert_not_called()


class GpuRuntimeTests(unittest.TestCase):
    def test_cpu_never_discovers_gpu_libraries(self):
        with patch("captionweave.backends.faster_whisper.site.getsitepackages", side_effect=AssertionError):
            configure_gpu_libraries("cpu", ["run", "source.wav", "--device", "cpu"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux loader setup")
    def test_module_and_console_reexec_use_a_canonical_module_invocation(self):
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / "nvidia" / "cudnn" / "lib"
            library.mkdir(parents=True)
            for entrypoint in (["captionweave", "run", "source.wav"], ["/installed/bin/captionweave", "run", "source.wav"]):
                with self.subTest(entrypoint=entrypoint), patch("sys.argv", entrypoint), patch(
                        "captionweave.backends.faster_whisper.site.getsitepackages", return_value=[temp]), patch(
                        "captionweave.backends.faster_whisper.site.getusersitepackages", return_value=temp), patch.dict(
                        os.environ, {"LD_LIBRARY_PATH": "/existing"}, clear=True), patch("os.execve") as reexec:
                    configure_gpu_libraries("cuda", ["run", "source.wav"])
                executable, arguments, environment = reexec.call_args.args
                self.assertEqual(executable, sys.executable)
                self.assertEqual(arguments, [sys.executable, "-m", "captionweave", "run", "source.wav"])
                self.assertEqual(environment["LD_LIBRARY_PATH"], f"{library}:/existing")
                self.assertEqual(environment["CAPTIONWEAVE_CUDA_PATH_READY"], "1")

    def test_library_calls_do_not_reexecute_the_host_process(self):
        with patch("os.execve", side_effect=AssertionError("Unexpected process replacement")):
            configure_gpu_libraries("auto")

    def test_linux_ready_sentinel_prevents_reexecution(self):
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / "nvidia" / "cudnn" / "lib"
            library.mkdir(parents=True)
            linux = SimpleNamespace(name="posix", pathsep=":", environ={
                "CAPTIONWEAVE_CUDA_PATH_READY": "1", "LD_LIBRARY_PATH": str(library)},
                execve=Mock(side_effect=AssertionError("Unexpected process replacement")))
            with patch("captionweave.backends.faster_whisper.os", linux), patch(
                    "captionweave.backends.faster_whisper.sys.platform", "linux"), patch(
                    "captionweave.backends.faster_whisper.site.getsitepackages", return_value=[temp]), patch(
                    "captionweave.backends.faster_whisper.site.getusersitepackages", return_value=temp):
                configure_gpu_libraries("cuda", ["run", "source.wav"])
            linux.execve.assert_not_called()

    def test_windows_dll_registration_is_idempotent_and_retains_handles(self):
        with tempfile.TemporaryDirectory() as temp:
            libraries = [Path(temp) / "nvidia" / "cudnn" / directory for directory in ("lib", "bin")]
            for library in libraries:
                library.mkdir(parents=True)
            handles = [object(), object()]
            windows = SimpleNamespace(name="nt", add_dll_directory=Mock(side_effect=handles),
                                      execve=Mock(side_effect=AssertionError("Unexpected process replacement")))
            retained, registered = [], set()
            with patch("captionweave.backends.faster_whisper.os", windows), patch(
                    "captionweave.backends.faster_whisper.DLL_HANDLES", retained), patch(
                    "captionweave.backends.faster_whisper.DLL_PATHS", registered), patch(
                    "captionweave.backends.faster_whisper.site.getsitepackages", return_value=[temp]), patch(
                    "captionweave.backends.faster_whisper.site.getusersitepackages", return_value=temp):
                configure_gpu_libraries("cuda", ["run", "source.wav"])
                configure_gpu_libraries("auto", ["run", "source.wav"])
            self.assertEqual(windows.add_dll_directory.call_count, 2)
            self.assertEqual({call.args[0] for call in windows.add_dll_directory.call_args_list},
                             {str(path) for path in libraries})
            self.assertEqual(registered, {str(path) for path in libraries})
            self.assertEqual(retained, handles)
            windows.execve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
