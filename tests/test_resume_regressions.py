import json
import os
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class ResumeMediaStatRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source.wav"
        with wave.open(str(self.source), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\0\0" * 32000)
        self.options = {"model": "not-loaded", "language": "ja", "device": "cpu",
                        "compute_type": "int8", "batch_size": 2, "mode": "balanced",
                        "start": 0, "duration": None, "audio_stream": 0, "offline": True}

    class Engine:
        language = "ja"
        actual_device = "test"

        def __init__(self, fail_at=None):
            self.fail_at = fail_at
            self.calls = []

        def block(self, audio, offset):
            self.calls.append(offset)
            if self.fail_at is not None and offset >= self.fail_at:
                raise RuntimeError("Interrupted block")
            return {"language": "ja", "alternatives": [], "segments": [
                {"start": offset + 0.1, "end": offset + 0.7, "text": "こんにちは", "flags": []}]}

    def touch_source(self):
        stat = self.source.stat()
        os.utime(self.source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    def manifest_stat(self, job):
        return json.loads((job / "manifest.json").read_text(encoding="utf-8"))["media"]["stat"]

    def current_stat(self):
        stat = self.source.stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def test_touch_reuses_completed_cache_refreshes_manifest_and_renders(self):
        from captionweave.subtitles import render_job
        from captionweave.core import transcribe_media

        with patch("captionweave.core.BLOCK_SECONDS", 1):
            job = transcribe_media(self.source, self.root / "jobs", self.options, self.Engine())
            original_stat = self.manifest_stat(job)
            self.touch_source()
            cached = self.Engine(fail_at=0)
            self.assertEqual(transcribe_media(self.source, self.root / "jobs", self.options, cached), job)

        self.assertEqual(cached.calls, [])
        self.assertNotEqual(self.manifest_stat(job), original_stat)
        self.assertEqual(self.manifest_stat(job), self.current_stat())
        report = render_job(job, self.root / "output.ja.srt")
        self.assertEqual(report["status"], "complete")

    def test_touch_refreshes_partial_resume_manifest(self):
        from captionweave.core import transcribe_media

        with patch("captionweave.core.BLOCK_SECONDS", 1):
            with self.assertRaisesRegex(RuntimeError, "Interrupted"):
                transcribe_media(self.source, self.root / "jobs", self.options, self.Engine(fail_at=1))
            job = next((self.root / "jobs").iterdir())
            original_stat = self.manifest_stat(job)
            self.touch_source()
            resumed = self.Engine()
            self.assertEqual(transcribe_media(self.source, self.root / "jobs", self.options, resumed), job)

        self.assertEqual(resumed.calls, [1])
        self.assertNotEqual(self.manifest_stat(job), original_stat)
        self.assertEqual(self.manifest_stat(job), self.current_stat())

    def test_content_change_creates_a_new_job(self):
        from captionweave.core import transcribe_media

        with patch("captionweave.core.BLOCK_SECONDS", 1):
            first = transcribe_media(self.source, self.root / "jobs", self.options, self.Engine())
            with self.source.open("r+b") as stream:
                stream.seek(128)
                value = stream.read(1)
                stream.seek(128)
                stream.write(bytes([value[0] ^ 1]))
            second_engine = self.Engine()
            second = transcribe_media(self.source, self.root / "jobs", self.options, second_engine)

        self.assertNotEqual(first, second)
        self.assertEqual(second_engine.calls, [0, 1])

    def test_corrupt_completed_transcript_is_still_rejected(self):
        from captionweave.core import transcribe_media

        with patch("captionweave.core.BLOCK_SECONDS", 1):
            job = transcribe_media(self.source, self.root / "jobs", self.options, self.Engine())
            (job / "transcript.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "corrupted"):
                transcribe_media(self.source, self.root / "jobs", self.options, self.Engine(fail_at=0))

    def test_backend_names_and_versions_produce_separate_caches(self):
        from captionweave.core import transcribe_media

        class VersionedEngine(self.Engine):
            def __init__(self, name, version):
                super().__init__()
                self.name, self.version = name, version

            def identity(self):
                return {"name": self.name, "revision": 1, "versions": {"engine": self.version}}

        with patch("captionweave.core.BLOCK_SECONDS", 1):
            jobs = [transcribe_media(self.source, self.root / "jobs", self.options, VersionedEngine(name, version))
                    for name, version in [("first", "1"), ("second", "1"), ("first", "2")]]
        self.assertEqual(len(set(jobs)), 3)
        identity = json.loads((jobs[0] / "manifest.json").read_text(encoding="utf-8"))["identity"]
        self.assertEqual(identity["backend"], {"name": "first", "revision": 1, "versions": {"engine": "1"}})

    def test_auto_language_is_reset_between_inputs_and_restored_between_blocks(self):
        from captionweave.core import transcribe_media

        class DetectingEngine(self.Engine):
            def __init__(self):
                super().__init__()
                self.languages = []

            def block(self, audio, offset):
                self.languages.append(self.language)
                return super().block(audio, offset)

        second_source = self.root / "second.wav"
        shutil.copyfile(self.source, second_source)
        engine = DetectingEngine()
        options = {**self.options, "language": "auto"}
        with patch("captionweave.core.BLOCK_SECONDS", 1):
            first = transcribe_media(self.source, self.root / "jobs", options, engine)
            second = transcribe_media(second_source, self.root / "jobs", options, engine)
        self.assertNotEqual(first, second)
        self.assertEqual(engine.languages, [None, "ja", None, "ja"])
        self.assertEqual(json.loads((second / "transcript.json").read_text(encoding="utf-8"))["language"], "ja")


if __name__ == "__main__":
    unittest.main()
