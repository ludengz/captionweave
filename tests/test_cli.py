import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from captionweave.cli import build_parser, default_output, default_outputs, expand_inputs, main
from captionweave.core import digest


class LiteralInputTests(unittest.TestCase):
    def test_existing_filename_with_brackets_takes_precedence_over_glob(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            literal = root / "lecture[final].mp4"
            literal.touch()
            (root / "lecturef.mp4").touch()
            self.assertEqual(expand_inputs([str(literal)]), [literal.resolve()])

    def test_glob_expansion_still_accepts_nonliteral_patterns(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "lecture.mp4"
            source.touch()
            self.assertEqual(expand_inputs([str(source.parent / "*.mp4")]), [source.resolve()])


class OutputAllocationTests(unittest.TestCase):
    def test_default_outputs_use_the_current_working_directory(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.chdir(temp):
            source = Path(temp) / "media" / "movie.mp4"
            self.assertEqual(default_output(source, [source], "en"), Path(temp) / "outputs" / "movie.en.srt")

    def test_batch_outputs_remain_unique_after_container_and_stem_collisions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = [root / "movie.mp4", root / "movie.mkv", root / "movie.mp4.mkv"]
            outputs = default_outputs(sources, "zh", root / "out")
            self.assertEqual(len(set(outputs.values())), len(sources))
            self.assertEqual(
                [default_output(source, sources, "zh", root / "out") for source in sources],
                [outputs[source.resolve()] for source in sources],
            )

    def test_batch_outputs_remain_unique_after_clip_and_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prefix = "a" * 200
            sources = [root / f"{prefix}1.mp4", root / f"{prefix}2.mkv"]
            outputs = default_outputs(sources, "zh", root / "out", start=10, duration=20)
            self.assertEqual(len(set(outputs.values())), len(sources))
            self.assertTrue(all(".clip-10-30" in path.name for path in outputs.values()))

    def test_single_long_sources_keep_a_stable_digest_when_truncated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prefix = "a" * 200
            first, second = root / f"{prefix}1.mp4", root / f"{prefix}2.mp4"
            first_output = default_output(first, [first], "zh")
            second_output = default_output(second, [second], "zh")
            self.assertNotEqual(first_output, second_output)
            self.assertTrue(first_output.name.endswith(f"-{digest(str(first.resolve()))[:8]}.zh.srt"))
            self.assertTrue(second_output.name.endswith(f"-{digest(str(second.resolve()))[:8]}.zh.srt"))
            self.assertLessEqual(len(first_output.name.removesuffix(".zh.srt").encode()), 180)
            self.assertLessEqual(len(second_output.name.removesuffix(".zh.srt").encode()), 180)


class CliLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original_directory = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.original_directory)
        self.source = self.root / "clip.mp4"
        self.source.write_bytes(b"placeholder media")
        self.work_dir = self.root / "jobs"

    def call(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(arguments)
        return code, json.loads(stdout.getvalue()), stderr.getvalue()

    def fake_transcribe_media(self, source, work_dir, options, recognizer, restart):
        job = Path(work_dir) / "fake-job"
        job.mkdir(parents=True, exist_ok=True)
        source = Path(source).resolve()
        transcript = {
            "schema_version": 1,
            "job_id": "fake-job",
            "source": str(source),
            "duration": 5.0,
            "language": "ja",
            "processed_intervals": [[0, 5]],
            "segments": [{"id": "s000001", "i": 1, "start": 1.0, "end": 2.0,
                          "text": "こんにちは", "flags": []}],
            "alternatives": [],
        }
        manifest = {"schema_version": 1, "job_id": "fake-job", "processed_duration": 5.0,
                    "asr_complete": True}
        (job / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
        (job / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        print("fake ASR progress")
        return job

    @staticmethod
    def response_for(request, text, **overrides):
        response = {key: request[key] for key in ["schema_version", "job_id", "source_digest", "target_language", "request_id"]}
        response["translations"] = [{"id": request["items"][0]["id"], "text": text, "status": "translated", **overrides}]
        return response

    def test_translation_lifecycle_and_changed_render_target_preserve_existing_outputs(self):
        with patch("captionweave.cli.create_backend", return_value=object()), patch(
                "captionweave.cli.transcribe_media", side_effect=self.fake_transcribe_media):
            code, run_result, stderr = self.call([
                "run", str(self.source), "--target", "zh", "--work-dir", str(self.work_dir),
            ])
        self.assertEqual(code, 3)
        self.assertEqual(run_result["results"][0]["status"], "translation_required")
        self.assertIn("fake ASR progress", stderr)
        job = Path(run_result["results"][0]["job"])
        request = json.loads(Path(run_result["results"][0]["requests"][0]).read_text(encoding="utf-8"))
        zh_response = self.root / "zh-response.json"
        zh_response.write_text(json.dumps(self.response_for(
            request, "你好。", source_text="おはよう", start=0.5, end=1.5), ensure_ascii=False), encoding="utf-8")

        code, imported, _ = self.call(["import", str(job), str(zh_response)])
        self.assertEqual((code, imported["status"], imported["remaining"]), (0, "ready_to_render", 0))
        code, rendered, _ = self.call(["render", str(job)])
        self.assertEqual((code, rendered["status"]), (0, "complete"))
        zh_output = self.root / "outputs" / "clip.zh.srt"
        self.assertTrue(zh_output.is_file())
        self.assertIn("00:00:00,500", zh_output.read_text(encoding="utf-8"))
        self.assertTrue(zh_output.with_suffix(".ass").is_file())
        self.assertTrue(zh_output.with_suffix(".quality.json").is_file())
        chinese_files = {path: path.read_bytes() for path in zh_output.parent.glob("clip.zh.*")}
        chinese_files.update({path: path.read_bytes() for path in zh_output.parent.glob("clip.ja-zh.*")})

        code, exported, _ = self.call(["export", str(job), "--target", "en"])
        self.assertEqual((code, exported["status"]), (0, "translation_required"))
        en_request = json.loads(Path(exported["requests"][0]).read_text(encoding="utf-8"))
        en_response = self.root / "en-response.json"
        en_response.write_text(json.dumps(self.response_for(en_request, "Hello.")), encoding="utf-8")
        code, imported, _ = self.call(["import", str(job), str(en_response)])
        self.assertEqual((code, imported["status"]), (0, "ready_to_render"))
        code, rendered, _ = self.call(["render", str(job), "--target", "en"])
        self.assertEqual((code, rendered["status"]), (0, "complete"))
        en_output = self.root / "outputs" / "clip.en.srt"
        self.assertTrue(en_output.is_file())
        self.assertIn("Hello.", en_output.read_text(encoding="utf-8"))
        self.assertEqual({path: path.read_bytes() for path in chinese_files}, chinese_files)

    def test_clip_output_uses_the_actual_selected_duration(self):
        with patch("captionweave.cli.create_backend", return_value=object()), patch(
                "captionweave.cli.transcribe_media", side_effect=self.fake_transcribe_media):
            code, result, _ = self.call([
                "run", str(self.source), "--target", "zh", "--duration", "20", "--work-dir", str(self.work_dir),
            ])
        self.assertEqual(code, 3)
        self.assertEqual(Path(result["results"][0]["output"]), self.root / "outputs" / "clip.clip-0-5.zh.srt")

    def test_run_defaults_to_auto_source_and_workspace_local_paths(self):
        with patch("captionweave.cli.create_backend", return_value=object()), patch(
                "captionweave.cli.transcribe_media", side_effect=self.fake_transcribe_media) as transcribe:
            code, result, _ = self.call(["run", str(self.source)])
        self.assertEqual(code, 0)
        self.assertEqual(transcribe.call_args.args[2]["language"], "auto")
        self.assertEqual(Path(result["results"][0]["job"]), self.root / ".captionweave" / "jobs" / "fake-job")
        self.assertTrue((self.root / "outputs" / "clip.ja.srt").is_file())

    def test_explicit_output_is_preserved(self):
        output = self.root / "chosen" / "captions.srt"
        with patch("captionweave.cli.create_backend", return_value=object()), patch(
                "captionweave.cli.transcribe_media", side_effect=self.fake_transcribe_media):
            code, _, _ = self.call(["run", str(self.source), "-o", str(output)])
        self.assertEqual(code, 0)
        self.assertTrue(output.is_file())

    def test_export_requires_an_explicit_target(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            build_parser().parse_args(["export", str(self.work_dir)])
        self.assertEqual(error.exception.code, 2)

    def test_render_without_saved_output_defaults_to_current_outputs_directory(self):
        with contextlib.redirect_stdout(io.StringIO()):
            job = self.fake_transcribe_media(self.source, self.work_dir, {}, None, False)
        code, result, _ = self.call(["render", str(job)])
        self.assertEqual((code, result["status"]), (0, "complete"))
        self.assertTrue((self.root / "outputs" / "clip.ja.srt").is_file())

    def test_render_forwards_custom_unclear_text(self):
        with patch("captionweave.cli.create_backend", return_value=object()), patch(
                "captionweave.cli.transcribe_media", side_effect=self.fake_transcribe_media):
            _, result, _ = self.call(["run", str(self.source), "--target", "en"])
        job = Path(result["results"][0]["job"])
        request = json.loads(Path(result["results"][0]["requests"][0]).read_text(encoding="utf-8"))
        response = self.response_for(request, None, status="unclear", reason="Speech is unclear")
        response_path = self.root / "response.json"
        response_path.write_text(json.dumps(response), encoding="utf-8")
        self.call(["import", str(job), str(response_path)])
        code, result, _ = self.call(["render", str(job), "--unclear-text", "[inaudible]"])
        self.assertEqual((code, result["status"]), (0, "complete"))
        self.assertIn("[inaudible]", (self.root / "outputs" / "clip.en.srt").read_text(encoding="utf-8"))

    def test_legacy_flags_are_rejected(self):
        for arguments in (["--diarize"], ["--words"], ["--split-gaps"], ["--max-gap", "2"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                build_parser().parse_args(["run", str(self.source), *arguments])

    def test_render_has_language_neutral_presentation_defaults(self):
        args = build_parser().parse_args(["render", str(self.work_dir)])
        self.assertEqual(args.font, "sans-serif")
        self.assertEqual(args.unclear_text, "[?]")


if __name__ == "__main__":
    unittest.main()
