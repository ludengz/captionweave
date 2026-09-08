import io
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.check_release import MAX_FILE_BYTES, audit_archive, audit_content, audit_tree


class ReleaseAuditTests(unittest.TestCase):
    def test_generic_multilingual_source_is_allowed(self):
        content = "message = 'Hello, 世界, مرحبا, Привет'\n".encode()
        self.assertEqual(audit_content("src/tool.py", content), [])

    def test_machine_paths_credentials_and_binary_media_are_rejected(self):
        cases = [
            ("README.md", ("/" + "home" + "/example-user/video.mp4").encode()),
            ("config.py", ("secret = 'sk-" + "a" * 32 + "'").encode()),
            ("media/video.mp4", b"video"),
            ("private/transcript.json", b"{}"),
        ]
        for name, content in cases:
            with self.subTest(name=name):
                self.assertTrue(audit_content(name, content))

    def test_git_ignored_jobs_are_excluded_but_force_tracked_jobs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text("jobs/\n", encoding="utf-8")
            (root / "README.md").write_text("Public example", encoding="utf-8")
            (root / "jobs").mkdir()
            (root / "jobs/transcript.json").write_text("{}", encoding="utf-8")
            self.assertEqual(audit_tree(root), [])
            subprocess.run(["git", "-C", str(root), "add", "-f", "jobs/transcript.json"], check=True)
            self.assertTrue(audit_tree(root))

    def test_archives_are_checked_without_extracting_their_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "safe.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("captionweave/__init__.py", "__version__ = '0.1.0'\n")
            self.assertEqual(audit_archive(wheel), [])
            bad = root / "unsafe.tar.gz"
            with tarfile.open(bad, "w:gz") as archive:
                member = tarfile.TarInfo("captionweave-0.1.0/jobs/transcript.json")
                payload = b"{}"
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            self.assertTrue(audit_archive(bad))

    def test_archive_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unsafe.whl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../escape.py", "pass")
            self.assertTrue(audit_archive(path))


    def test_runtime_records_in_custom_locations_are_rejected(self):
        cases = [
            ("custom/blocks/000001.json", b"{}"),
            ("custom/translations/es/requests/batch.json", b"{}"),
            ("custom/response.json", b"{}"),
            ("custom/review.json", b"{}"),
            ("data/example.json", b'{"job_id": "example", "segments": []}'),
            ("example.es.json", b'[{"start": 0, "end": 1, "text": "Synthetic"}]'),
        ]
        for name, content in cases:
            with self.subTest(name=name):
                self.assertTrue(audit_content(name, content))

    def test_archive_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "links.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                member = zipfile.ZipInfo("package/link")
                member.create_system = 3
                member.external_attr = 0o120777 << 16
                archive.writestr(member, "outside")
            self.assertTrue(audit_archive(wheel))
            source = root / "links.tar.gz"
            with tarfile.open(source, "w:gz") as archive:
                member = tarfile.TarInfo("package/link")
                member.type = tarfile.SYMTYPE
                member.linkname = "../outside"
                archive.addfile(member)
            self.assertTrue(audit_archive(source))

    def test_oversize_and_non_utf8_content_are_rejected(self):
        self.assertTrue(audit_content("large.txt", b"a" * (MAX_FILE_BYTES + 1)))
        self.assertTrue(audit_content("binary.txt", b"\xff"))

    def test_missing_tracked_file_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            source = root / "example.py"
            source.write_text("pass", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "example.py"], check=True)
            source.unlink()
            self.assertTrue(audit_tree(root))


if __name__ == "__main__":
    unittest.main()
