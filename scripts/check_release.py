#!/usr/bin/env python3
"""Check publishable files and distribution archives for accidental local data."""
import argparse
import json
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


MAX_FILE_BYTES = 4 * 1024 * 1024
PRIVATE_DIRECTORIES = {
    ".git", ".venv", "venv", "__pycache__", ".captionweave", ".codex", ".agents",
    "media", "jobs", "outputs", "recordings", "transcripts", "blocks", "translations", ".ssh",
}
PRIVATE_NAMES = {"transcript.json", "manifest.json", "ledger.json", "quality.json", "review.json", "response.json", ".lock"}
BINARY_SUFFIXES = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".wav", ".mp3",
    ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".bin", ".onnx",
    ".safetensors", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".pyc", ".pyo",
}
PATTERNS = {
    "absolute home-directory path": re.compile(r"/(?:home|Users)/[^/\s<>\"']+|[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s<>\"']+"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "API credential": re.compile(r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,}|AKIA[0-9A-Z]{16})\b"),
    "credential in connection URL": re.compile(r"(?:https?|ssh|postgres|mysql)://[^\s/:]+:[^\s/@]+@"),
}


def finding(path, reason):
    return {"path": str(path), "reason": reason}


def path_problem(name):
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", str(path)):
        return "absolute or traversing archive path"
    if any(part in PRIVATE_DIRECTORIES or part.startswith(".captionweave-") for part in path.parts):
        return "runtime or private directory"
    if path.name in PRIVATE_NAMES or path.name == ".env" or path.name.startswith(".env."):
        return "runtime state or environment file"
    if path.suffix.lower() in BINARY_SUFFIXES | {".srt", ".ass", ".vtt"}:
        return "media, model, generated subtitle, or compiled artifact"
    return None


def audit_content(name, content):
    problem = path_problem(name)
    if problem:
        return [finding(name, problem)]
    if len(content) > MAX_FILE_BYTES:
        return [finding(name, "file exceeds public-source size limit")]
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return [finding(name, "unexpected binary content")]
    if PurePosixPath(name).suffix.lower() == ".json":
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        if isinstance(value, dict) and "job_id" in value:
            return [finding(name, "runtime job or translation record")]
        if isinstance(value, list) and any(
                isinstance(row, dict) and {"start", "end", "text"} <= row.keys() for row in value):
            return [finding(name, "generated subtitle records")]
    return [finding(name, description) for description, pattern in PATTERNS.items() if pattern.search(text)]


def audit_tree(root):
    root = Path(root).resolve()
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True, check=True,
    )
    findings = []
    for name in sorted(set(result.stdout.decode("utf-8").split("\0")) - {""}):
        path = root / name
        if path.is_symlink():
            findings.append(finding(name, "symbolic links are not allowed in the public source bundle"))
        elif not path.exists():
            findings.append(finding(name, "tracked path is missing; commit or stage its deletion"))
        elif not path.is_file():
            findings.append(finding(name, "unexpected non-file entry"))
        else:
            with path.open("rb") as stream:
                findings.extend(audit_content(name, stream.read(MAX_FILE_BYTES + 1)))
    return findings


def audit_archive(path):
    path = Path(path)
    findings = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    findings.append(finding(member.filename, "archive contains a symbolic link"))
                    continue
                with archive.open(member) as stream:
                    findings.extend(audit_content(member.filename, stream.read(MAX_FILE_BYTES + 1)))
    else:
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                if member.isdir():
                    continue
                if not member.isfile():
                    findings.append(finding(member.name, "archive contains a non-regular file"))
                    continue
                with archive.extractfile(member) as stream:
                    findings.extend(audit_content(member.name, stream.read(MAX_FILE_BYTES + 1)))
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", action="append", default=[], type=Path)
    args = parser.parse_args(argv)
    try:
        findings = audit_tree(args.root)
        for archive in args.archive:
            findings.extend({**item, "archive": archive.name} for item in audit_archive(archive))
        print(json.dumps({"clean": not findings, "findings": findings}, indent=2))
        return 1 if findings else 0
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(json.dumps({"clean": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
