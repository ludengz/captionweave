import copy
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


class TimestampTests(unittest.TestCase):
    def test_timestamp_carries_milliseconds_into_next_minute(self):
        from captionweave.core import timestamp as fmt_ts
        self.assertEqual(fmt_ts(59.9996), "00:01:00,000")

    def test_clip_windows_never_overlap_or_exceed_whisper_limit(self):
        from captionweave.core import clip_windows
        for duration in [0.01, 29.9, 30.0, 30.1, 65.3, 600.0]:
            clips = clip_windows(duration)
            for index, clip in enumerate(clips):
                self.assertGreater(clip["end"], clip["start"])
                self.assertLessEqual(clip["end"] - clip["start"], 30)
                self.assertLessEqual(clip["end"], duration)
                if index:
                    self.assertEqual(clips[index - 1]["end"], clip["start"])


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.job = Path(self.tmp.name) / "job"
        self.job.mkdir()
        self.output = Path(self.tmp.name) / "output"
        self.output.mkdir()
        self.transcript = {
            "schema_version": 1, "job_id": "test-job", "language": "ja", "duration": 10.0,
            "segments": [
                {"id": "s000001", "start": 1.0, "end": 2.0, "text": "こんにちは", "flags": []},
                {"id": "s000002", "start": 3.0, "end": 4.0, "text": "ありがとう", "flags": []},
            ],
        }
        (self.job / "transcript.json").write_text(json.dumps(self.transcript), encoding="utf-8")

    def request_and_response(self):
        from captionweave.subtitles import export_requests
        paths = export_requests(self.job, "zh", batch_size=60)
        request = json.loads(paths[0].read_text(encoding="utf-8"))
        response = {k: request[k] for k in ["schema_version", "job_id", "source_digest", "target_language", "request_id"]}
        response["translations"] = [
            {"id": "s000002", "text": "谢谢。", "status": "translated"},
            {"id": "s000001", "text": "你好。", "status": "translated"},
        ]
        return request, response

    def save_response(self, value):
        path = self.job / "response.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_translation_uses_ids_instead_of_array_order(self):
        from captionweave.subtitles import import_response, translated_segments
        _, response = self.request_and_response()
        import_response(self.job, self.save_response(response))
        rows = translated_segments(self.job, "zh")
        self.assertEqual([r["text"] for r in rows], ["你好。", "谢谢。"])

    def test_wrong_source_duplicate_or_missing_ids_are_rejected(self):
        from captionweave.subtitles import import_response
        _, response = self.request_and_response()
        wrong = copy.deepcopy(response)
        wrong["source_digest"] = "wrong"
        with self.assertRaises(ValueError):
            import_response(self.job, self.save_response(wrong))
        duplicate = copy.deepcopy(response)
        duplicate["translations"][1]["id"] = "s000002"
        with self.assertRaises(ValueError):
            import_response(self.job, self.save_response(duplicate))
        missing = copy.deepcopy(response)
        missing["translations"].pop()
        with self.assertRaises(ValueError):
            import_response(self.job, self.save_response(missing))

    def test_pending_template_cannot_be_imported_as_completed_translation(self):
        from captionweave.subtitles import import_response
        _, response = self.request_and_response()
        response["translations"][0].update(text=None, status="pending")
        with self.assertRaises(ValueError):
            import_response(self.job, self.save_response(response))

    def test_source_edit_invalidates_previous_translation(self):
        from captionweave.subtitles import import_response, translated_segments
        _, response = self.request_and_response()
        import_response(self.job, self.save_response(response))
        self.transcript["segments"][0]["text"] = "こんばんは"
        (self.job / "transcript.json").write_text(json.dumps(self.transcript), encoding="utf-8")
        with self.assertRaises(ValueError):
            translated_segments(self.job, "zh")

    def test_unclear_and_omission_are_explicit_and_counted(self):
        from captionweave.subtitles import import_response, render_job
        _, response = self.request_and_response()
        response["translations"][0].update(text=None, status="omit", reason="Non-speech artifact")
        response["translations"][1].update(text=None, status="unclear")
        import_response(self.job, self.save_response(response))
        report = render_job(self.job, self.output / "output.zh.srt", "zh")
        self.assertEqual((report["subtitle_count"], report["omitted_segments"], report["unclear_segments"]), (1, 1, 1))
        self.assertIn("[?]", (self.output / "output.zh.srt").read_text(encoding="utf-8"))
        self.assertTrue(report["review_recommended"])

    def test_import_is_idempotent_and_corrections_require_replace(self):
        from captionweave.subtitles import import_response, translated_segments
        _, response = self.request_and_response()
        path = self.save_response(response)
        self.assertEqual(import_response(self.job, path), 0)
        self.assertEqual(import_response(self.job, path), 0)
        response["translations"][0]["text"] = "非常感谢。"
        path = self.save_response(response)
        with self.assertRaises(ValueError):
            import_response(self.job, path)
        import_response(self.job, path, replace=True)
        self.assertEqual(translated_segments(self.job, "zh")[1]["text"], "非常感谢。")

    def test_tampered_request_is_rejected(self):
        from captionweave.subtitles import import_response
        request, response = self.request_and_response()
        path = self.job / "translations/zh/requests" / f"{request['request_id']}.json"
        request["items"][0]["text"] = "different text"
        path.write_text(json.dumps(request), encoding="utf-8")
        with self.assertRaises(ValueError):
            import_response(self.job, self.save_response(response))
        self.assertFalse((self.job / "translations/zh/ledger.json").exists())

    def test_invalid_timing_and_untyped_ids_are_rejected_before_writes(self):
        from captionweave.subtitles import import_response
        _, response = self.request_and_response()
        for change in [{"start": -1}, {"end": float("nan")}, {"start": 5, "end": 4}, {"id": []}]:
            bad = copy.deepcopy(response)
            bad["translations"][0].update(change)
            with self.assertRaises(ValueError):
                import_response(self.job, self.save_response(bad))
        self.assertFalse((self.job / "translations/zh/ledger.json").exists())

    def test_existing_outputs_are_backed_up_and_identical_render_is_unchanged(self):
        from captionweave.subtitles import import_response, render_job
        _, response = self.request_and_response()
        import_response(self.job, self.save_response(response))
        output = self.output / "result.zh.srt"
        output.write_bytes(b"old non-UTF8 subtitle \xff")
        report = render_job(self.job, output, "zh")
        self.assertEqual(Path(report["backups"][0]).read_bytes(), b"old non-UTF8 subtitle \xff")
        modified = output.stat().st_mtime_ns
        report = render_job(self.job, output, "zh")
        self.assertEqual(output.stat().st_mtime_ns, modified)
        self.assertEqual(report["backups"], [])


class LayoutTests(unittest.TestCase):
    def row(self, key, start, end, text, **extra):
        return {"id": key, "source_ids": [key], "start": start, "end": end,
                "text": text, "source_text": text, "flags": [], **extra}

    def test_wrapping_preserves_chinese_content_without_inserted_spaces(self):
        from captionweave.subtitles import timed_captions
        text = "这是一个用于验证中文字幕断行后不插入额外空格的测试文本。" * 3
        rows = timed_captions([self.row("s1", 1, 10, text)], 12)
        self.assertEqual("".join(row["text"] for row in rows), text)
        self.assertTrue(all(len(row["display_text"].splitlines()) <= 2 for row in rows))
        self.assertEqual(rows[0]["start"], 1)
        self.assertEqual(rows[-1]["end"], 10)

    def test_overlapping_captions_preserve_all_source_ids(self):
        from captionweave.subtitles import timed_captions
        source = [self.row("s1", 1, 3, "First."), self.row("s2", 2.5, 4, "Second."), self.row("s3", 4, 4.1, "Yes.")]
        rows = timed_captions(source, 10, 48)
        self.assertEqual({key for row in rows for key in row["source_ids"]}, {"s1", "s2", "s3"})
        self.assertTrue(all(left["end"] <= right["start"] for left, right in zip(rows, rows[1:])))

    def test_speaker_labels_survive_rendering(self):
        from captionweave.subtitles import timed_captions
        rows = timed_captions([self.row("s1", 1, 2, "你好。", speaker="Speaker 1")], 10)
        self.assertIn("Speaker 1", rows[0]["display_text"])

    def test_ass_text_does_not_execute_recognized_override_tags(self):
        from captionweave.subtitles import ass_text
        self.assertNotIn("{", ass_text(r"{\an8}hello"))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class ResumeTests(unittest.TestCase):
    def test_interruption_resumes_only_unfinished_blocks_and_detects_corrupt_cache(self):
        from captionweave.core import transcribe_media
        class Engine:
            language = "ja"
            actual_device = "test"

            def __init__(self, fail=False):
                self.fail = fail
                self.calls = []

            def block(self, audio, offset):
                self.calls.append(offset)
                if self.fail and offset >= 1:
                    raise RuntimeError("Interrupted block")
                return {"language": "ja", "alternatives": [], "segments": [
                    {"start": offset + 0.1, "end": offset + 0.7, "text": "こんにちは", "flags": []}]}
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\0\0" * 32000)
            options = dict(model="not-loaded", language="ja", device="cpu", compute_type="int8",
                           batch_size=2, mode="balanced", start=0, duration=None, audio_stream=0,
                           offline=True)
            with patch("captionweave.core.BLOCK_SECONDS", 1):
                first = Engine(fail=True)
                with self.assertRaisesRegex(RuntimeError, "Interrupted"):
                    transcribe_media(source, Path(temp) / "jobs", options, first)
                second = Engine()
                job = transcribe_media(source, Path(temp) / "jobs", options, second)
                self.assertEqual(second.calls, [1])
                third = Engine()
                self.assertEqual(transcribe_media(source, Path(temp) / "jobs", options, third), job)
                self.assertEqual(third.calls, [])
                (job / "transcript.json").write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "corrupted"):
                    transcribe_media(source, Path(temp) / "jobs", options, Engine())

    def test_job_lock_prevents_concurrent_writers_and_releases_after_failure(self):
        from captionweave.core import job_lock
        with tempfile.TemporaryDirectory() as temp:
            with job_lock(temp):
                with self.assertRaises(RuntimeError):
                    with job_lock(temp):
                        self.fail("The second writer acquired the lock")
            with job_lock(temp):
                pass


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class AudioTimelineTests(unittest.TestCase):
    @staticmethod
    def samples(wav):
        return [sample[0] for sample in struct.iter_unpack("<h", wav.readframes(wav.getnframes()))]

    def test_aac_timestamp_gap_is_retained_in_pcm(self):
        from captionweave.core import extract_audio, probe_media
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "gap.m4a"
            output = Path(temp) / "aligned.wav"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                "-af", "asetpts=PTS+gte(T\\,1)*0.5/TB", "-c:a", "aac", str(source),
            ], check=True)
            info = probe_media(source)
            extract_audio(source, output, info, start=0, duration=None)
            with wave.open(str(output)) as wav:
                rate = wav.getframerate()
                audio = self.samples(wav)
            self.assertAlmostEqual(len(audio) / rate, info["duration"], delta=0.05)
            self.assertGreater(info["duration"], 2.4)
            self.assertLess(max(abs(value) for value in audio[int(1.15 * rate):int(1.35 * rate)]), 0.002 * 32768)

    def test_audio_start_offset_and_trailing_silence_match_video(self):
        from captionweave.core import extract_audio, probe_media
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "delayed.mkv", Path(temp) / "aligned.wav"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=16x16:r=10:d=4",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
                "-filter_complex", "[1:a]asetpts=PTS+1/TB[a]", "-map", "0:v", "-map", "[a]",
                "-c:v", "libx264", "-c:a", "aac", str(source)], check=True)
            info = probe_media(source)
            extract_audio(source, output, info)
            with wave.open(str(output)) as wav:
                audio = self.samples(wav)
            self.assertAlmostEqual(len(audio) / 16000, info["duration"], delta=0.05)
            self.assertLess(max(abs(value) for value in audio[:8000]), 10)
            self.assertGreater(max(abs(value) for value in audio[19000:24000]), 500)
            self.assertLess(max(abs(value) for value in audio[-8000:]), 10)

    def test_selects_the_requested_audio_track(self):
        from captionweave.core import extract_audio, probe_media
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "tracks.mkv", Path(temp) / "selected.wav"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=400:duration=2",
                "-f", "lavfi", "-i", "sine=frequency=800:duration=2",
                "-map", "0:a", "-map", "1:a", "-c:a", "aac", str(source)], check=True)
            info = probe_media(source, 1)
            extract_audio(source, output, info)
            with wave.open(str(output)) as wav:
                data = self.samples(wav)[4000:20000]
            crossings = sum(left <= 0 < right for left, right in zip(data, data[1:]))
            peak = crossings * 16000 / (len(data) - 1)
            self.assertAlmostEqual(peak, 800, delta=5)
            with self.assertRaises(ValueError):
                probe_media(source, 2)


class InputTests(unittest.TestCase):
    def test_same_stem_containers_and_directories_cannot_clobber_outputs(self):
        from captionweave.cli import default_output
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            sources = [root / "a/movie.mp4", root / "a/movie.mkv", root / "b/movie.mp4"]
            outputs = [default_output(p, sources, "zh", root / "out") for p in sources]
            self.assertEqual(len(set(outputs)), 3)

    def test_directory_expansion_ignores_cache_and_duplicate_inputs(self):
        from captionweave.cli import expand_inputs
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "movie.mp4").touch()
            (root / "jobs").mkdir()
            (root / "jobs/aligned.wav").touch()
            files = expand_inputs([str(root), str(root / "movie.mp4")], recursive=True, work_dir=root / "jobs")
            self.assertEqual(files, [root / "movie.mp4"])


if __name__ == "__main__":
    unittest.main()
