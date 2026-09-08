import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path


class LanguageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.job = self.root / "job"
        self.job.mkdir()
        self.transcript = {
            "schema_version": 1, "job_id": "synthetic-language-job", "language": "EN_us", "duration": 10.0,
            "segments": [{"id": "stable-001", "start": 1.0, "end": 4.0,
                          "text": "Hello, world.", "flags": []}],
        }
        self.write_transcript()

    def write_transcript(self):
        (self.job / "transcript.json").write_text(json.dumps(self.transcript), encoding="utf-8")

    def translate(self, target, text=None, status="translated"):
        from captionweave.subtitles import export_requests, import_response
        request_path = export_requests(self.job, target)[0]
        request = json.loads(request_path.read_text(encoding="utf-8"))
        response = {key: request[key] for key in [
            "schema_version", "job_id", "source_digest", "target_language", "request_id",
        ]}
        response["translations"] = [{"id": "stable-001", "text": text, "status": status}]
        response_path = self.job / "response.json"
        response_path.write_text(json.dumps(response), encoding="utf-8")
        import_response(self.job, response_path)
        return request_path, request, response

    def test_language_tags_are_canonical_and_allow_practical_subtags(self):
        from captionweave.subtitles import language_tag
        for source, expected in {
            "EN_us": "en-US", "ZH_hANT_tw": "zh-Hant-TW", "es_419": "es-419",
            "sr_latn_rs": "sr-Latn-RS", "de-1901": "de-1901", "en-us-u-NU-latn": "en-US-u-nu-latn",
            "en-X-a": "en-x-a",
        }.items():
            with self.subTest(source=source):
                self.assertEqual(language_tag(source), expected)
        for bad in ["../en", "en/US", "en\\US", "en..us", " en", "en\n", "", None, "en-", "en-u", "en-x"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                language_tag(bad)

    def test_target_aliases_share_requests_and_ledger_and_canonical_artifact_names(self):
        from captionweave.subtitles import export_requests, render_job
        before = (self.job / "transcript.json").read_bytes()
        first = export_requests(self.job, "ZH_hANT")[0]
        self.assertEqual(first, export_requests(self.job, "zh-Hant")[0])
        path, request, _ = self.translate("zh_hant", "你好，世界。")
        self.assertEqual(path, first)
        self.assertEqual(request["source_language"], "en-US")
        self.assertEqual(request["target_language"], "zh-Hant")
        self.assertEqual(export_requests(self.job, "ZH-HANT"), [])
        report = render_job(self.job, self.root / "result.zh-Hant.srt", "ZH_hANT")
        self.assertEqual(report["language"], "zh-Hant")
        self.assertTrue((self.root / "result.en-US-zh-Hant.srt").is_file())
        self.assertEqual((self.job / "transcript.json").read_bytes(), before)

    def test_standalone_private_use_tags_are_lowercase(self):
        from captionweave.subtitles import language_tag
        for source, expected in {
            "X_PrIvAtE": "x-private", "x-A-b-Latn-US-12345678": "x-a-b-latn-us-12345678",
            "x-X": "x-x", "X_12345678": "x-12345678",
        }.items():
            with self.subTest(source=source):
                self.assertEqual(language_tag(source), expected)
                self.assertEqual(language_tag(expected), expected)

    def test_grandfathered_tags_preserve_registered_spelling_without_alias_remapping(self):
        from captionweave.subtitles import language_tag
        tags = [
            "en-GB-oed", "i-ami", "i-bnn", "i-default", "i-enochian", "i-hak", "i-klingon",
            "i-lux", "i-mingo", "i-navajo", "i-pwn", "i-tao", "i-tay", "i-tsu",
            "sgn-BE-FR", "sgn-BE-NL", "sgn-CH-DE", "art-lojban", "cel-gaulish", "no-bok",
            "no-nyn", "zh-guoyu", "zh-hakka", "zh-min", "zh-min-nan", "zh-xiang",
        ]
        for tag in tags:
            with self.subTest(tag=tag):
                self.assertEqual(language_tag(tag.upper().replace("-", "_")), tag)
                self.assertEqual(language_tag(tag), tag)

    def test_private_use_and_grandfathered_targets_complete_cli_exchange(self):
        from captionweave.cli import build_parser, main
        from captionweave.core import digest

        def call(arguments):
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(arguments)
            return code, json.loads(stdout.getvalue()) if stdout.getvalue() else None, stderr.getvalue()

        before = (self.job / "transcript.json").read_bytes()
        original_digest = digest({"language": "EN_us", "segments": [
            {"id": "stable-001", "text": "Hello, world."},
        ]})
        (self.job / "manifest.json").write_text("{}", encoding="utf-8")
        for alias, target, text in [
            ("X_PrIvAtE", "x-private", "Private greeting."),
            ("I_KLINGON", "i-klingon", "nuqneH"),
            ("SGN_be_fr", "sgn-BE-FR", "Hello."),
        ]:
            with self.subTest(target=target):
                self.assertEqual(build_parser().parse_args(["run", "clip.wav", "--target", alias]).target, target)
                code, exported, stderr = call(["export", str(self.job), "--target", alias])
                self.assertEqual((code, stderr), (0, ""))
                self.assertEqual(exported["status"], "translation_required")
                request_path = Path(exported["requests"][0])
                self.assertEqual(request_path.parent.parent, self.job / "translations" / target)
                request = json.loads(request_path.read_text(encoding="utf-8"))
                self.assertEqual(request["target_language"], target)
                self.assertEqual(request["source_digest"], original_digest)
                self.assertEqual(request["source_language"], "en-US")
                response = {key: request[key] for key in [
                    "schema_version", "job_id", "source_digest", "target_language", "request_id",
                ]}
                response["translations"] = [{"id": "stable-001", "text": text, "status": "translated"}]
                response_path = self.job / "response.json"
                response["target_language"] = alias
                response_path.write_text(json.dumps(response), encoding="utf-8")
                code, _, stderr = call(["import", str(self.job), str(response_path)])
                self.assertEqual(code, 1)
                self.assertIn("target_language", stderr)
                self.assertFalse((request_path.parent.parent / "ledger.json").exists())
                response["target_language"] = target
                response_path.write_text(json.dumps(response), encoding="utf-8")
                code, imported, stderr = call(["import", str(self.job), str(response_path)])
                self.assertEqual((code, stderr), (0, ""))
                self.assertEqual(imported, {"status": "ready_to_render", "remaining": 0})
                ledger = json.loads((request_path.parent.parent / "ledger.json").read_text(encoding="utf-8"))
                self.assertEqual(ledger["source_digest"], original_digest)
                code, exported, stderr = call(["export", str(self.job), "--target", alias])
                self.assertEqual((code, stderr), (0, ""))
                self.assertEqual(exported, {"status": "ready_to_render", "requests": []})
                output = self.root / f"result.{target}.srt"
                code, report, stderr = call(["render", str(self.job), "--target", alias, "-o", str(output)])
                self.assertEqual((code, stderr), (0, ""))
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["language"], target)
                self.assertEqual(report["source_digest"], original_digest)
                self.assertIn(text, output.read_text(encoding="utf-8"))
                self.assertTrue((self.root / f"result.en-US-{target}.srt").is_file())
                self.assertEqual((self.job / "transcript.json").read_bytes(), before)

    def test_invalid_private_use_and_grandfathered_targets_cannot_create_paths(self):
        from captionweave.cli import build_parser
        from captionweave.subtitles import export_requests, language_tag

        for bad in [
            "x", "x-", "x-abcdefghi", "x--private", "i", "i-private", "i-klingon-extra",
            "x-../outside", "x-private/../../outside", "x-private\\outside", "i-klingon/../../outside",
            "x-foo.bar", "x-foo:bar", "x-foo\x00bar", "x-prıvate", "x-K", "İ-klingon",
            "x-private\n", " x-private",
        ]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    language_tag(bad)
                with self.assertRaises(ValueError):
                    export_requests(self.job, bad)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    build_parser().parse_args(["export", str(self.job), "--target", bad])
                self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.job / "translations").exists())

    def test_distinct_target_variants_keep_separate_ledgers(self):
        from captionweave.subtitles import translated_segments
        self.translate("pt-br", "Olá, Brasil.")
        self.translate("pt-pt", "Olá, Portugal.")
        self.assertEqual(translated_segments(self.job, "PT_br")[0]["text"], "Olá, Brasil.")
        self.assertEqual(translated_segments(self.job, "PT_pt")[0]["text"], "Olá, Portugal.")

    def test_response_metadata_remains_exact(self):
        from captionweave.subtitles import import_response
        _, _, response = self.translate("fr-CA", "Bonjour.")
        response["target_language"] = "fr-ca"
        path = self.job / "response.json"
        path.write_text(json.dumps(response), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "target_language"):
            import_response(self.job, path)

    def test_neutral_default_and_custom_unclear_text(self):
        from captionweave.subtitles import render_job
        self.translate("ar", status="unclear")
        output = self.root / "uncertain.ar.srt"
        render_job(self.job, output, "ar")
        self.assertIn("[?]", output.read_text(encoding="utf-8"))
        self.assertIn("Style: Default,sans-serif,", output.with_suffix(".ass").read_text(encoding="utf-8"))
        report = render_job(self.job, output, "ar", unclear_text="[غير واضح]")
        self.assertIn("[غير واضح]", output.read_text(encoding="utf-8"))
        self.assertEqual(report["unclear_segments"], 1)
        self.assertTrue(report["review_recommended"])

    def test_invalid_unclear_text_is_rejected_before_publication(self):
        from captionweave.subtitles import render_job
        self.translate("fr", status="unclear")
        output = self.root / "unsafe.fr.srt"
        for marker in [None, "", "  ", "\u200b", "a\n\n99\n00:00:01,000 --> 00:00:02,000", "a\rline", "a\x00b"]:
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                render_job(self.job, output, "fr", unclear_text=marker)
        self.assertFalse(output.exists())

    def test_cjk_latin_cyrillic_and_arabic_remain_logical_unicode_text(self):
        from captionweave.subtitles import render_job
        samples = {"zh-Hant": "世界你好。", "fr": "Bonjour, déjà vu !", "ru": "Привет, мир!", "ar": "مَرْحَبًا بالعالم!"}
        for target, sample in samples.items():
            with self.subTest(target=target):
                self.translate(target, sample)
                output = self.root / f"unicode.{target}.srt"
                report = render_job(self.job, output, target)
                self.assertEqual(report["subtitle_count"], 1)
                for extension in ["srt", "ass", "txt"]:
                    self.assertIn(sample, output.with_suffix(f".{extension}").read_text(encoding="utf-8"))
                rows = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
                self.assertEqual(rows[0]["source_ids"], ["stable-001"])

    def test_subtitle_fields_cannot_inject_extra_records(self):
        from captionweave.subtitles import render_job
        payload = "hello\n\n99\n00:00:05,000 --> 00:00:06,000\nInjected"
        self.transcript["segments"][0]["speaker"] = payload
        self.write_transcript()
        self.translate("fr", payload)
        output = self.root / "injection.fr.srt"
        report = render_job(self.job, output, "fr", width=200)
        self.assertEqual(report["subtitle_count"], 1)
        for path in [output, self.root / "injection.en-US-fr.srt"]:
            content = path.read_text(encoding="utf-8")
            self.assertEqual(len(content.strip().split("\n\n")), 1)
            self.assertNotIn("\n\n99\n", content)

    def test_wrapping_preserves_long_text_and_combining_marks_across_cues(self):
        import unicodedata
        from captionweave.subtitles import timed_captions
        samples = [
            "世界您好，这是用于验证完整文本的字幕。" * 6,
            "Cafe\u0301 au lait and a long afternoon conversation. " * 6,
            "Это длинный разговор о повседневной жизни. " * 6,
            "مَرْحَبًا بالعالم هذا حوار طويل للاختبار. " * 6,
        ]
        for sample in samples:
            with self.subTest(sample=sample[:20]):
                sample = sample.strip()
                rows = timed_captions([{
                    "id": "stable-001", "source_ids": ["stable-001"], "start": 1.0, "end": 11.0,
                    "text": sample, "source_text": sample, "flags": [],
                }], 12, width=48)
                self.assertEqual("".join(row["text"] for row in rows), sample)
                self.assertGreater(len(rows), 1)
                self.assertEqual((rows[0]["start"], rows[-1]["end"]), (1.0, 11.0))
                for row in rows:
                    lines = row["display_text"].splitlines()
                    self.assertLessEqual(len(lines), 2)
                    for line in lines:
                        self.assertFalse(unicodedata.category(line[0]).startswith("M"))
                        columns = sum(0 if unicodedata.category(c).startswith("M")
                                      else 2 if unicodedata.east_asian_width(c) in {"W", "F"} else 1
                                      for c in line)
                        self.assertLessEqual(columns, 48)

    def test_default_line_width_counts_cjk_characters_as_two_columns(self):
        from captionweave.subtitles import timed_captions
        sample = "世" * 48
        rows = timed_captions([{
            "id": "stable-001", "source_ids": ["stable-001"], "start": 1.0, "end": 4.0,
            "text": sample, "source_text": sample, "flags": [],
        }], 10)
        self.assertEqual(rows[0]["display_text"].splitlines(), ["世" * 24, "世" * 24])

    def test_source_matching_target_variant_does_not_generate_duplicate_bilingual_files(self):
        from captionweave.subtitles import render_job
        self.translate("en-US", "Hello, world.")
        report = render_job(self.job, self.root / "same.en-US.srt", "EN_us")
        self.assertEqual(len(report["files"]), 5)
        self.assertFalse((self.root / "same.en-US-en-US.srt").exists())

    def test_invalid_source_language_cannot_escape_output_directory(self):
        from captionweave.subtitles import export_requests
        self.transcript["language"] = "../../outside"
        self.write_transcript()
        with self.assertRaises(ValueError):
            export_requests(self.job, "fr")


if __name__ == "__main__":
    unittest.main()
