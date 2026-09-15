import copy
import contextlib
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from array import array
import math
import struct

from captionweave.core import SAMPLE_RATE, atomic_json, digest


class RecheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.job = Path(self.temp.name)
        self.transcript = {
            "schema_version": 1, "job_id": "recheck-test", "language": "ja", "duration": 200,
            "segments": [
                {"id": "s000001", "start": 111, "end": 114, "text": "こんにちは", "flags": []},
                {"id": "s000002", "start": 144, "end": 147, "text": "ありがとう", "flags": []},
            ], "alternatives": [], "processed_intervals": [[100, 160]],
        }
        atomic_json(self.job / "transcript.json", self.transcript)
        atomic_json(self.job / "manifest.json", {
            "job_id": "recheck-test", "asr_complete": True, "processed_start": 100, "processed_duration": 60,
            "transcript_digest": digest(self.transcript),
        })
        cycle = b"".join(struct.pack("<h", round(math.sin(i * 2 * math.pi / 80) * 1000)) for i in range(80))
        with wave.open(str(self.job / "aligned.wav"), "wb") as wav:
            wav.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            wav.writeframes(cycle * (60 * SAMPLE_RATE // 80))
        self.options = dict(model="test-model", language="ja", device="cpu", compute_type="int8", offline=True)

    class Engine:
        def __init__(self, fail_at=None):
            self.calls = []
            self.fail_at = fail_at

        def recheck(self, audio, offset):
            self.calls.append((offset, len(audio), max(map(abs, audio), default=0)))
            if len(self.calls) == self.fail_at:
                raise RuntimeError("inference interrupted")
            return [{"start": offset + 5, "end": offset + 6, "text": "こんにちは", "flags": [],
                     "words": [[offset + 5, offset + 6, "こんにちは", 0.9]]}]

    def test_context_windows_cover_long_speech_and_clip_to_saved_timeline(self):
        from captionweave.recheck import context_windows
        rows = [{"start": 100, "end": 101}, {"start": 117, "end": 151}, {"start": 159, "end": 160}]
        windows = context_windows(rows, 100, 60, context=5)
        self.assertTrue(all(100 <= a < b <= 160 and b - a <= 30 for a, b in windows))
        for point in [100.1, 117, 129, 140, 150.9, 159.9]:
            self.assertTrue(any(a <= point < b for a, b in windows))
        for value in [-1, float("nan"), float("inf"), 16]:
            with self.assertRaises(ValueError):
                context_windows(rows, 100, 60, context=value)

    def test_fractional_job_start_never_seeks_before_saved_audio(self):
        from captionweave.recheck import recheck_audio
        start = 100.0001
        self.transcript["processed_intervals"] = [[start, start + 60]]
        self.transcript["segments"][0].update(start=start, end=101.5)
        atomic_json(self.job / "transcript.json", self.transcript)
        manifest = json.loads((self.job / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(processed_start=start, transcript_digest=digest(self.transcript))
        atomic_json(self.job / "manifest.json", manifest)
        engine = self.Engine()
        result = recheck_audio(self.job, self.transcript, self.transcript["segments"][:1], self.options,
                               gain_db=0, recognizer=engine)
        self.assertEqual(result["windows"][0][0], start)
        self.assertEqual(engine.calls[0][0], start)

    def test_gain_preserves_samples_and_does_not_clip_or_mutate_original(self):
        from captionweave.recheck import bounded_gain
        original = array("f", [0, 0.01, -0.02, 0.1])
        saved = original[:]
        gained, db = bounded_gain(original, 9)
        self.assertEqual(original, saved)
        self.assertEqual(len(gained), len(original))
        self.assertAlmostEqual(db, 9)
        self.assertGreater(max(map(abs, gained)), max(map(abs, original)))
        for signal in [array("f", [1, -1]), array("f", [0]) * 16]:
            gained, db = bounded_gain(signal, 12)
            self.assertEqual(gained, signal)
            self.assertEqual(db, 0)
        gained, db = bounded_gain(array("f", [0.8, -0.8]), 12)
        self.assertLessEqual(max(map(abs, gained)), 0.980001)
        self.assertLess(db, 2)

    def test_one_db_gain_is_not_dropped_by_floating_point_rounding(self):
        from captionweave.recheck import recheck_audio
        engine = self.Engine()
        result = recheck_audio(self.job, self.transcript, self.transcript["segments"][:1], self.options,
                               gain_db=1, recognizer=engine)
        self.assertEqual(result["created_passes"], 2)
        self.assertEqual(result["gain_skipped_windows"], 0)

    def test_recheck_preserves_source_and_resumes_individual_passes_after_failure(self):
        from captionweave.recheck import recheck_audio, load_recheck_evidence
        before = (self.job / "transcript.json").read_bytes()
        engine = self.Engine(fail_at=2)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            recheck_audio(self.job, self.transcript, self.transcript["segments"], self.options, recognizer=engine)
        self.assertEqual(len(list((self.job / "rechecks").glob("*.json"))), 1)
        resumed = self.Engine()
        result = recheck_audio(self.job, self.transcript, self.transcript["segments"], self.options, recognizer=resumed)
        self.assertEqual(len(resumed.calls), 3)
        self.assertEqual(result["cached_passes"], 1)
        self.assertEqual((self.job / "transcript.json").read_bytes(), before)
        evidence = load_recheck_evidence(self.job, self.transcript)
        self.assertEqual(len(evidence), 4)
        self.assertEqual({r["method"] for r in evidence}, {"original", "gain"})
        self.assertTrue(all(r["start"] >= 100 for r in evidence))
        cached = self.Engine()
        recheck_audio(self.job, self.transcript, self.transcript["segments"], self.options, recognizer=cached)
        self.assertEqual(cached.calls, [])

    def test_evidence_is_exported_for_selected_accepted_translation_with_provenance(self):
        from captionweave.subtitles import export_requests, import_response
        from captionweave.recheck import recheck_audio
        request = json.loads(export_requests(self.job, "zh")[0].read_text(encoding="utf-8"))
        response = {k: request[k] for k in ["schema_version", "job_id", "source_digest", "target_language", "request_id"]}
        response["translations"] = [{"id": r["id"], "text": None, "status": "unclear"} for r in request["items"]]
        atomic_json(self.job / "response.json", response)
        import_response(self.job, self.job / "response.json")
        ledger = (self.job / "translations/zh/ledger.json").read_bytes()
        recheck_audio(self.job, self.transcript, self.transcript["segments"][:1], self.options, recognizer=self.Engine())
        updated = json.loads(export_requests(self.job, "zh", ids=["s000001"])[0].read_text(encoding="utf-8"))
        alternatives = updated["items"][0]["alternatives"]
        self.assertEqual({a["method"] for a in alternatives}, {"original", "gain"})
        self.assertTrue(all(a["recheck_id"] and a["language"] == "ja" for a in alternatives))
        self.assertEqual((self.job / "translations/zh/ledger.json").read_bytes(), ledger)

    def test_completed_export_does_not_read_unused_recheck_audio(self):
        from captionweave.subtitles import export_requests, import_response
        from captionweave.recheck import recheck_audio
        recheck_audio(self.job, self.transcript, self.transcript["segments"][:1], self.options, recognizer=self.Engine())
        request = json.loads(export_requests(self.job, "zh")[0].read_text(encoding="utf-8"))
        response = {k: request[k] for k in ["schema_version", "job_id", "source_digest", "target_language", "request_id"]}
        response["translations"] = [{"id": row["id"], "text": "你好。", "status": "translated"} for row in request["items"]]
        atomic_json(self.job / "response.json", response)
        import_response(self.job, self.job / "response.json")
        (self.job / "aligned.wav").unlink()
        self.assertEqual(export_requests(self.job, "zh"), [])

    def test_changed_source_revision_ignores_old_evidence_and_corrupt_evidence_fails(self):
        from captionweave.recheck import recheck_audio, load_recheck_evidence
        recheck_audio(self.job, self.transcript, self.transcript["segments"][:1], self.options, recognizer=self.Engine())
        changed = copy.deepcopy(self.transcript)
        changed["segments"][0]["text"] = "changed source"
        self.assertEqual(load_recheck_evidence(self.job, changed), [])
        path = next((self.job / "rechecks").glob("*.json"))
        record = json.loads(path.read_text(encoding="utf-8"))
        record["segments"][0]["text"] = "corrupted"
        atomic_json(path, record)
        with self.assertRaisesRegex(ValueError, "corrupt"):
            load_recheck_evidence(self.job, self.transcript)
        for malformed in [[], None, {"schema_version": 1}]:
            atomic_json(path, malformed)
            with self.assertRaisesRegex(ValueError, "corrupt"):
                load_recheck_evidence(self.job, self.transcript)

    def test_changed_audio_bytes_exclude_stale_evidence_without_changing_source_ids(self):
        from captionweave.recheck import recheck_audio, load_recheck_evidence
        from captionweave.subtitles import export_requests
        rows = self.transcript["segments"][:1]
        recheck_audio(self.job, self.transcript, rows, self.options, recognizer=self.Engine())
        with wave.open(str(self.job / "aligned.wav"), "rb") as wav:
            parameters = wav.getparams()
            changed = b"".join(struct.pack("<h", -value[0])
                               for value in struct.iter_unpack("<h", wav.readframes(wav.getnframes())))
        with wave.open(str(self.job / "aligned.wav"), "wb") as wav:
            wav.setparams(parameters)
            wav.writeframes(changed)
        self.assertEqual(load_recheck_evidence(self.job, self.transcript), [])
        packet = json.loads(export_requests(self.job, "zh", ids=["s000001"])[0].read_text(encoding="utf-8"))
        self.assertFalse(packet["items"][0].get("recheck_context"))
        recheck_audio(self.job, self.transcript, rows, self.options, recognizer=self.Engine())
        self.assertEqual(len(load_recheck_evidence(self.job, self.transcript)), 2)
        self.assertEqual(json.loads((self.job / "transcript.json").read_text(encoding="utf-8")), self.transcript)

    def test_cli_recheck_exports_requests_and_review_finds_unflagged_uncertainty(self):
        from captionweave.cli import main
        from captionweave.subtitles import import_response
        arguments = ["recheck", str(self.job), "--ids", "s000001", "--target", "zh", "--offline"]
        output = io.StringIO()
        with patch("captionweave.recheck.create_backend", return_value=self.Engine()), contextlib.redirect_stdout(output):
            code = main(arguments)
        result = json.loads(output.getvalue())
        self.assertEqual((code, result["status"]), (3, "translation_required"))
        request = json.loads(Path(result["requests"][0]).read_text(encoding="utf-8"))
        response = {key: request[key] for key in ["schema_version", "job_id", "source_digest", "target_language", "request_id"]}
        response["translations"] = [{"id": "s000001", "text": None, "status": "unclear"}]
        atomic_json(self.job / "response.json", response)
        import_response(self.job, self.job / "response.json")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["review", str(self.job), "--target", "zh"]), 0)
        review = json.loads(output.getvalue())
        self.assertEqual([r["id"] for r in review["segments"]], ["s000001"])
        self.assertEqual({r["method"] for r in review["alternatives"]}, {"original", "gain"})
        output = io.StringIO()
        with patch("captionweave.recheck.create_backend", return_value=self.Engine()) as factory, contextlib.redirect_stdout(output):
            self.assertEqual(main(["recheck", str(self.job), "--target", "zh", "--offline"]), 3)
        self.assertEqual(factory.return_value.calls, [])
        self.assertTrue(json.loads(output.getvalue())["requires_replace"])

    def test_zero_duration_source_keeps_window_evidence_for_timing_review(self):
        from captionweave.subtitles import export_requests
        from captionweave.recheck import recheck_audio
        self.transcript["segments"][0]["end"] = 111
        atomic_json(self.job / "transcript.json", self.transcript)
        manifest = json.loads((self.job / "manifest.json").read_text(encoding="utf-8"))
        manifest["transcript_digest"] = digest(self.transcript)
        atomic_json(self.job / "manifest.json", manifest)
        recheck_audio(self.job, self.transcript, self.transcript["segments"][:1], self.options, recognizer=self.Engine())
        packet = json.loads(export_requests(self.job, "zh", ids=["s000001"])[0].read_text(encoding="utf-8"))
        item = packet["items"][0]
        self.assertFalse(item.get("alternatives"))
        self.assertEqual({r["method"] for r in item["recheck_context"]}, {"original", "gain"})
        self.assertTrue(all(r["end"] > r["start"] for r in item["recheck_context"]))
        from captionweave.cli import main
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["review", str(self.job), "--ids", "s000001"]), 0)
        self.assertEqual(len(json.loads(output.getvalue())["recheck_context"]), 2)

    def test_recheck_without_target_collects_source_evidence_without_translation(self):
        from captionweave.cli import main
        output = io.StringIO()
        with patch("captionweave.recheck.create_backend", return_value=self.Engine()), contextlib.redirect_stdout(output):
            code = main(["recheck", str(self.job), "--ids", "s000001", "--gain-db", "0"])
        result = json.loads(output.getvalue())
        self.assertEqual((code, result["status"], result["requests"]), (0, "review_required", []))
        self.assertEqual(result["created_passes"], 1)
        self.assertFalse((self.job / "translations").exists())

    def test_no_candidates_does_not_create_backend_or_translation_requests(self):
        from captionweave.cli import main
        output = io.StringIO()
        with patch("captionweave.recheck.create_backend") as factory, contextlib.redirect_stdout(output):
            self.assertEqual(main(["recheck", str(self.job)]), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "no_candidates")
        factory.assert_not_called()
        self.assertFalse((self.job / "rechecks").exists())

    def test_console_recheck_without_candidates_does_not_prepare_gpu_runtime(self):
        from captionweave.cli import main
        output = io.StringIO()
        with patch("sys.argv", ["captionweave", "recheck", str(self.job)]), contextlib.redirect_stdout(output):
            with patch("captionweave.cli.prepare_runtime", side_effect=AssertionError("Unexpected GPU setup")):
                self.assertEqual(main(), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "no_candidates")

    def test_recheck_inherits_saved_target_model_and_offline_setting(self):
        from captionweave.cli import main
        manifest = json.loads((self.job / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(target_language="es-419", identity={"options": {
            "backend": "faster-whisper", "model": "test-model", "offline": True}})
        atomic_json(self.job / "manifest.json", manifest)
        output = io.StringIO()
        with patch("captionweave.recheck.create_backend", return_value=self.Engine()) as factory, contextlib.redirect_stdout(output):
            self.assertEqual(main(["recheck", str(self.job), "--ids", "s000001", "--language", "en"]), 3)
        options = factory.call_args.args[0]
        self.assertEqual((options["language"], options["model"], options["offline"]), ("en", "test-model", True))
        packet = json.loads(Path(json.loads(output.getvalue())["requests"][0]).read_text(encoding="utf-8"))
        self.assertEqual(packet["target_language"], "es-419")

    def test_recheck_rejects_changed_transcript_and_incomplete_audio_before_inference(self):
        from captionweave.recheck import recheck_audio
        engine = self.Engine()
        changed = copy.deepcopy(self.transcript)
        changed["segments"][0]["text"] = "Changed."
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            recheck_audio(self.job, changed, changed["segments"], self.options, recognizer=engine)
        with wave.open(str(self.job / "aligned.wav"), "wb") as wav:
            wav.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            wav.writeframes(b"\0\0" * SAMPLE_RATE)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            recheck_audio(self.job, self.transcript, self.transcript["segments"], self.options, recognizer=engine)
        self.assertEqual(engine.calls, [])

    def test_truncated_pcm_is_rejected_even_when_header_duration_matches(self):
        from captionweave.core import wav_duration
        from captionweave.recheck import recheck_audio
        audio_path = self.job / "aligned.wav"
        with audio_path.open("r+b") as stream:
            stream.truncate(44 + SAMPLE_RATE)
        self.assertEqual(wav_duration(audio_path), 60)
        engine = self.Engine()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            recheck_audio(self.job, self.transcript, self.transcript["segments"][:1],
                           self.options, recognizer=engine)
        self.assertEqual(engine.calls, [])
        self.assertFalse((self.job / "rechecks").exists())

    def test_recheck_cache_includes_backend_revision(self):
        from captionweave.recheck import recheck_audio
        engine = self.Engine()
        engine.identity = lambda: {"name": "test", "revision": 1}
        rows = self.transcript["segments"][:1]
        recheck_audio(self.job, self.transcript, rows, self.options, recognizer=engine)
        engine.identity = lambda: {"name": "test", "revision": 2}
        result = recheck_audio(self.job, self.transcript, rows, self.options, recognizer=engine)
        self.assertEqual((result["created_passes"], result["cached_passes"]), (2, 0))

    def test_cli_can_recheck_an_existing_job_with_a_different_backend(self):
        from captionweave.cli import main
        before = (self.job / "transcript.json").read_bytes()
        output = io.StringIO()
        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"), patch(
                "platform.mac_ver", return_value=("14.0", ("", "", ""), "arm64")), patch(
                "captionweave.recheck.create_backend", return_value=self.Engine()) as factory, contextlib.redirect_stdout(output):
            code = main(["recheck", str(self.job), "--ids", "s000001", "--backend", "mlx-whisper",
                         "--device", "metal", "--model", "tiny", "--compute-type", "float16"])
        self.assertEqual(code, 0)
        self.assertEqual(factory.call_args.args[0]["backend"], "mlx-whisper")
        self.assertEqual(factory.call_args.args[0]["device"], "metal")
        self.assertEqual((self.job / "transcript.json").read_bytes(), before)

    def test_corrected_timing_selects_the_review_window_and_exports_its_evidence(self):
        from captionweave.subtitles import export_requests, import_response, select_review_segments
        from captionweave.recheck import recheck_audio
        packet = json.loads(export_requests(self.job, "es", ids=["s000001"])[0].read_text(encoding="utf-8"))
        response = {key: packet[key] for key in ["schema_version", "job_id", "source_digest", "target_language", "request_id"]}
        response["translations"] = [{"id": "s000001", "text": "Hola [?]", "status": "partial",
                                     "reason": "The following phrase is unresolved.", "start": 125, "end": 128}]
        atomic_json(self.job / "response.json", response)
        import_response(self.job, self.job / "response.json")
        rows = select_review_segments(self.job, "es")
        self.assertEqual((rows[0]["start"], rows[0]["end"]), (125, 128))
        result = recheck_audio(self.job, self.transcript, rows, self.options, gain_db=0, recognizer=self.Engine())
        self.assertEqual(result["windows"], [(120, 133)])
        reviewed = json.loads(export_requests(self.job, "es", ids=["s000001"])[0].read_text(encoding="utf-8"))
        self.assertEqual(reviewed["items"][0]["alternatives"][0]["start"], 125)


if __name__ == "__main__":
    unittest.main()
