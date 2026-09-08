import json
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _render_competing(job, output, entered, release, results, hold_lock):
    import captionweave.subtitles

    original = captionweave.subtitles.publish_files
    if hold_lock:
        def delayed(files):
            entered.set()
            if not release.wait(15):
                raise RuntimeError("test publisher was not released")
            return original(files)
        captionweave.subtitles.publish_files = delayed
    try:
        report = captionweave.subtitles.render_job(job, output)
        results.put(("ok", report["job_id"]))
    except Exception as error:
        results.put(("error", type(error).__name__, str(error)))


class SubtitleRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def make_job(self, name="job", source=None, segments=None):
        job = self.root / name
        job.mkdir()
        transcript = {
            "schema_version": 1,
            "job_id": name,
            "language": "ja",
            "duration": 10.0,
            "segments": segments or [{
                "id": "s000001", "start": 1.0, "end": 2.0,
                "text": "こんにちは", "flags": [],
            }],
        }
        if source:
            transcript["source"] = str(source)
        (job / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
        return job

    def import_translation(self, job, changes=None):
        from captionweave.subtitles import export_requests, import_response

        request = json.loads(export_requests(job, "zh")[0].read_text(encoding="utf-8"))
        response = {key: request[key] for key in [
            "schema_version", "job_id", "source_digest", "target_language", "request_id",
        ]}
        response["translations"] = []
        for item in request["items"]:
            value = {"id": item["id"], "text": "你好。", "status": "translated"}
            value.update((changes or {}).get(item["id"], {}))
            response["translations"].append(value)
        response_path = job / "response.json"
        response_path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
        import_response(job, response_path)

    def test_equal_target_captions_keep_each_distinct_source(self):
        from captionweave.subtitles import artifact_texts, timed_captions

        rows = timed_captions([
            {"id": "s1", "source_ids": ["s1"], "start": 1, "end": 2, "text": "好的。", "source_text": "はい。", "flags": []},
            {"id": "s2", "source_ids": ["s2"], "start": 2.2, "end": 3, "text": "好的。", "source_text": "ええ。", "flags": []},
        ], 10)
        self.assertEqual(rows[0]["source_ids"], ["s1", "s2"])
        self.assertEqual(rows[0]["source_text"], "はい。 ええ。")
        rendered = artifact_texts(rows, "Noto Sans CJK SC", 58, bilingual=True)["srt"]
        self.assertIn("はい。 ええ。\n好的。", rendered)

    def test_fractional_eof_is_clamped_to_representable_srt_timestamp(self):
        from captionweave.subtitles import timed_captions

        rows = timed_captions([{
            "id": "s1", "source_ids": ["s1"], "start": 9.5, "end": 10.001,
            "text": "最后一句。", "source_text": "最後の一言。", "flags": [],
        }], 10.0006)
        self.assertEqual((rows[0]["start"], rows[0]["end"]), (9.5, 10.0))

    def test_source_text_cannot_inject_extra_srt_cues(self):
        from captionweave.subtitles import render_job

        job = self.make_job()
        self.import_translation(job, {"s000001": {
            "source_text": "こんにちは\n\n99\n00:00:12,000 --> 00:00:13,000\nInjected caption",
        }})
        output = self.root / "result.zh.srt"
        report = render_job(job, output, "zh")
        rendered = (self.root / "result.ja-zh.srt").read_text(encoding="utf-8")
        self.assertEqual(report["subtitle_count"], 1)
        self.assertNotIn("\n\n99\n", rendered)
        self.assertIn("こんにちは 99 00:00:12,000 --> 00:00:13,000 Injected caption", rendered)

    def test_timing_and_source_overrides_survive_word_timing_heuristic(self):
        from captionweave.subtitles import render_job

        job = self.make_job(segments=[{
            "id": "s000001", "start": 1.0, "end": 5.0, "text": "元の認識", "flags": [],
            "words": [[1.1, 1.2, "短"], [4.0, 4.1, "非常に長い語句"]],
        }])
        self.import_translation(job, {"s000001": {
            "source_text": "修正した原文", "start": 3.0, "end": 4.0,
        }})
        output = self.root / "override.zh.srt"
        render_job(job, output, "zh")
        rendered_rows = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual((rendered_rows[0]["start"], rendered_rows[0]["end"]), (3.0, 4.0))
        bilingual = self.root / "override.ja-zh.srt"
        self.assertIn("修正した原文", bilingual.read_text(encoding="utf-8"))

    def test_public_quality_report_is_backed_up_as_an_artifact(self):
        from captionweave.subtitles import render_job

        job = self.make_job()
        output = self.root / "quality.zh.srt"
        public_quality = output.with_suffix(".quality.json")
        public_quality.write_bytes(b"old public quality\xff")
        report = render_job(job, output)
        backups = [Path(path) for path in report["backups"]]
        quality_backup = next(path for path in backups if path.name == public_quality.name)
        self.assertEqual(quality_backup.read_bytes(), b"old public quality\xff")

    def test_repeated_render_is_byte_identical_with_windows_default_newlines(self):
        from captionweave.subtitles import render_job

        job = self.make_job()
        self.import_translation(job)
        output = self.root / "repeat.zh.srt"
        fdopen = os.fdopen

        def windows_fdopen(fd, mode="r", *args, **kwargs):
            if "b" not in mode and kwargs.get("newline") is None:
                kwargs["newline"] = "\r\n"
            return fdopen(fd, mode, *args, **kwargs)

        with patch("captionweave.subtitles.os.fdopen", side_effect=windows_fdopen):
            first = render_job(job, output, "zh")
            paths = [Path(path) for path in first["files"]] + [job / "quality.json"]
            before = {path: path.read_bytes() for path in paths}
            for path in paths:
                os.utime(path, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
            mtimes = {path: path.stat().st_mtime_ns for path in paths}
            second = render_job(job, output, "zh")

        self.assertEqual(first["backups"], [])
        self.assertEqual(second["backups"], [])
        self.assertEqual({path: path.read_bytes() for path in paths}, before)
        self.assertEqual({path: path.stat().st_mtime_ns for path in paths}, mtimes)
        self.assertEqual(output.read_bytes(), "1\n00:00:01,000 --> 00:00:02,000\n你好。\n".encode("utf-8"))
        for path, content in before.items():
            with self.subTest(path=path):
                self.assertNotIn(b"\r\n", content)
        self.assertFalse(any(path.is_file() for path in self.root.rglob(".captionweave-backups/**/*")))

    def test_second_replace_failure_rolls_back_existing_and_new_files(self):
        import captionweave.subtitles

        existing = self.root / "existing.srt"
        absent = self.root / "absent.ass"
        existing.write_text("old", encoding="utf-8")
        replace = os.replace

        def fail_second_publish(source, destination):
            if Path(destination) == absent and Path(source).name.startswith(".absent.ass-"):
                raise OSError("injected second replacement failure")
            return replace(source, destination)

        with patch("captionweave.subtitles.os.replace", side_effect=fail_second_publish):
            with self.assertRaisesRegex(OSError, "injected second"):
                captionweave.subtitles.publish_files({existing: "new", absent: "new"})
        self.assertEqual(existing.read_text(encoding="utf-8"), "old")
        self.assertFalse(absent.exists())

    def test_failed_rollback_is_recovered_from_the_immutable_journal(self):
        import captionweave.subtitles

        existing = self.root / "existing.srt"
        absent = self.root / "absent.ass"
        existing.write_text("old", encoding="utf-8")
        replace = os.replace

        def fail_commit_and_restore(source, destination):
            name = Path(source).name
            if Path(destination) == absent and name.startswith(".absent.ass-"):
                raise OSError("injected commit failure")
            if Path(destination) == existing and name.startswith(".existing.srt-restore-"):
                raise OSError("injected rollback failure")
            return replace(source, destination)

        with patch("captionweave.subtitles.os.replace", side_effect=fail_commit_and_restore):
            with self.assertRaisesRegex(RuntimeError, "needs recovery"):
                captionweave.subtitles.publish_files({existing: "new", absent: "new"})
        journals = list((self.root / ".captionweave-backups").glob("*/publish-journal.json"))
        self.assertEqual(len(journals), 1)
        self.assertEqual(existing.read_text(encoding="utf-8"), "new")
        captionweave.subtitles._recover_pending_publications({existing: "new", absent: "new"})
        self.assertEqual(existing.read_text(encoding="utf-8"), "old")
        self.assertFalse(absent.exists())
        self.assertFalse(journals[0].exists())

    def test_recovery_journal_cannot_escape_the_current_output_bundle(self):
        import captionweave.subtitles

        safe = self.root / "safe.srt"
        escaped = self.root / "outside.txt"
        escaped.write_text("do not touch", encoding="utf-8")
        journal = self.root / ".captionweave-backups" / "20260101T000000.000000Z" / "publish-journal.json"
        journal.parent.mkdir(parents=True)
        journal.write_text(json.dumps({
            "schema_version": 1,
            "targets": [
                {"path": str(safe), "backup": None, "new_sha256": "0" * 64},
                {"path": str(escaped), "backup": None, "new_sha256": "0" * 64},
            ],
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "outside this output bundle"):
            captionweave.subtitles.publish_files({safe: "new"})
        self.assertEqual(escaped.read_text(encoding="utf-8"), "do not touch")

    def test_recovery_preserves_user_edit_and_does_not_restore_other_targets(self):
        import captionweave.subtitles

        edited = self.root / "edited.srt"
        untouched = self.root / "untouched.ass"
        edited.write_text("old edited", encoding="utf-8")
        untouched.write_text("old untouched", encoding="utf-8")
        replace = os.replace

        def fail_commit_and_rollback(source, destination):
            name = Path(source).name
            if Path(destination) == untouched and name.startswith(".untouched.ass-"):
                raise OSError("injected commit failure")
            if Path(destination) == edited and name.startswith(".edited.srt-restore-"):
                raise OSError("injected rollback failure")
            return replace(source, destination)

        with patch("captionweave.subtitles.os.replace", side_effect=fail_commit_and_rollback):
            with self.assertRaisesRegex(RuntimeError, "needs recovery"):
                captionweave.subtitles.publish_files({edited: "published", untouched: "also published"})
        edited.write_text("user edit after interruption", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "manual review"):
            captionweave.subtitles._recover_pending_publications({edited: "published", untouched: "also published"})
        self.assertEqual(edited.read_text(encoding="utf-8"), "user edit after interruption")
        self.assertEqual(untouched.read_text(encoding="utf-8"), "old untouched")
        self.assertTrue(list((self.root / ".captionweave-backups").glob("*/publish-journal.json")))

    def test_render_rejects_every_output_path_inside_job_before_writes(self):
        from captionweave.subtitles import render_job

        job = self.make_job()
        original = (job / "transcript.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "inside the job"):
            render_job(job, job / "transcript.srt")
        self.assertEqual((job / "transcript.json").read_bytes(), original)
        self.assertFalse((job / ".captionweave-publish-locks").exists())
        self.assertFalse((job / ".captionweave-backups").exists())

    def test_render_rejects_sidecars_that_collide_with_unusual_source_suffixes(self):
        from captionweave.subtitles import render_job

        source = self.root / "movie.json"
        source.write_text("not media, but protected input", encoding="utf-8")
        job = self.make_job(source=source)
        with self.assertRaisesRegex(ValueError, "input media"):
            render_job(job, self.root / "movie.srt")
        self.assertEqual(source.read_text(encoding="utf-8"), "not media, but protected input")

    def test_render_resolves_symlinked_output_directory_before_guarding_job(self):
        from captionweave.subtitles import render_job

        job = self.make_job()
        link = self.root / "output-link"
        try:
            link.symlink_to(job, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "inside the job"):
            render_job(job, link / "result.srt")

    def test_render_rejects_an_artifact_symlink_into_job_before_writes(self):
        from captionweave.subtitles import render_job

        job = self.make_job()
        protected = job / "aligned.wav"
        protected.write_bytes(b"preserve job data")
        output = self.root / "outside.srt"
        try:
            output.with_suffix(".ass").symlink_to(protected)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "inside the job"):
            render_job(job, output)
        self.assertEqual(protected.read_bytes(), b"preserve job data")
        self.assertFalse(output.exists())

    def test_render_rejects_resolved_artifact_aliases_before_writes(self):
        from captionweave.subtitles import render_job

        job = self.make_job()
        output = self.root / "aliases.srt"
        try:
            output.with_suffix(".ass").symlink_to(output.name)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "colliding artifact"):
            render_job(job, output)
        self.assertFalse(output.exists())

    def test_distinct_jobs_cannot_publish_the_same_bundle_concurrently(self):
        first = self.make_job("first")
        second = self.make_job("second")
        output = self.root / "shared.ja.srt"
        context = multiprocessing.get_context("spawn")
        entered, release = context.Event(), context.Event()
        results = context.Queue()
        publisher = context.Process(target=_render_competing,
                                    args=(str(first), str(output), entered, release, results, True))
        contender = context.Process(target=_render_competing,
                                    args=(str(second), str(output), entered, release, results, False))
        publisher.start()
        try:
            self.assertTrue(entered.wait(15), "first publisher did not acquire the output lock")
            contender.start()
            contender.join(15)
            self.assertFalse(contender.is_alive(), "contending publisher blocked instead of failing")
            contender_result = results.get(timeout=5)
            self.assertEqual(contender_result[0:2], ("error", "RuntimeError"))
            release.set()
            publisher.join(15)
            self.assertFalse(publisher.is_alive(), "first publisher did not finish")
            publisher_result = results.get(timeout=5)
            self.assertEqual(publisher_result[0], "ok")
        finally:
            release.set()
            for process in (publisher, contender):
                if process.pid is not None:
                    process.join(5)
                    if process.is_alive():
                        process.terminate()
                        process.join(5)
                    process.close()
            results.close()
            results.join_thread()


if __name__ == "__main__":
    unittest.main()
