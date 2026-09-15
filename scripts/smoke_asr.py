"""Run real ASR, contextual rechecks, and resume checks on synthetic speech."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from captionweave.backends import resolve_backend
from captionweave.core import atomic_json, read_json, wav_duration


PHRASES = ("There is a blue notebook beside the window.", "The next meeting begins tomorrow morning.")
START = 5.125
DURATION = 40


def run_command(arguments):
    result = subprocess.run([str(value) for value in arguments], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{arguments[0]} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def cli(*arguments):
    return json.loads(run_command([sys.executable, "-m", "captionweave", *arguments]))


def fingerprint(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_media(root, voice):
    for command in ["ffmpeg", "ffprobe"]:
        if not shutil.which(command):
            raise RuntimeError(f"Install {command} before running this smoke test")
    speech = []
    for index, phrase in enumerate(PHRASES):
        if sys.platform == "darwin":
            path = root / f"speech-{index}.aiff"
            run_command(["say", "-v", voice, "-o", path, phrase])
        else:
            path = root / f"speech-{index}.wav"
            run_command(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                         f"flite=text='{phrase}':voice=slt", "-ar", "16000", "-ac", "1", path])
        speech.extend(["-i", path])
    media = root / "synthetic.wav"
    run_command(["ffmpeg", "-v", "error", "-nostdin", *speech, "-filter_complex",
                 "[0:a]adelay=8125:all=1,volume=0.12[a];"
                 "[1:a]adelay=37125:all=1,volume=0.12[b];"
                 "[a][b]amix=inputs=2:normalize=0,apad,atrim=duration=50",
                 "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", media])
    return media


def word_error_rate(reference, actual):
    expected = re.findall(r"[a-z]+", reference.lower())
    recognized = re.findall(r"[a-z]+", actual.lower())
    previous = list(range(len(recognized) + 1))
    for i, word in enumerate(expected, 1):
        current = [i]
        for j, other in enumerate(recognized, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1] / len(expected)


def smoke(args):
    backend = resolve_backend(args.backend, args.device)
    parent = args.work_dir
    if parent:
        parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="captionweave-smoke-", dir=parent))
    print(f"Smoke artifacts: {root}", file=sys.stderr, flush=True)
    media = make_media(root, args.voice)
    media_digest = fingerprint(media)
    runtime = ["--backend", args.backend, "--device", args.device, "--model", args.model]
    if args.offline:
        runtime.append("--offline")
    run_args = ["run", media, *runtime, "--language", "en", "--start", str(START),
                "--duration", str(DURATION), "--work-dir", root / "jobs", "--output-dir", root / "outputs"]
    result = cli(*run_args)["results"][0]
    job = Path(result["job"])
    manifest = read_json(job / "manifest.json")
    transcript = read_json(job / "transcript.json")
    expected_device = "metal" if backend == "mlx-whisper" else args.device
    if expected_device != "auto" and manifest["actual_device"] != expected_device:
        raise RuntimeError(f"Expected {expected_device}, used {manifest['actual_device']}")
    if manifest["identity"]["backend"]["name"] != backend:
        raise RuntimeError("Recognition used a different backend")
    if not transcript["segments"] or not any(row.get("words") for row in transcript["segments"]):
        raise RuntimeError("Recognition produced no speech or word timing")
    if abs(wav_duration(job / "aligned.wav") - DURATION) > 0.001:
        raise RuntimeError("Aligned audio duration changed")
    if transcript["processed_intervals"] != [[START, START + DURATION]]:
        raise RuntimeError("Recognition did not preserve the selected playback interval")
    for row in transcript["segments"]:
        if not START <= row["start"] < row["end"] <= START + DURATION:
            raise RuntimeError("A recognized segment lies outside the playback interval")
        if any(not START <= word[0] <= word[1] <= START + DURATION for word in row.get("words", [])):
            raise RuntimeError("A word timestamp lies outside the playback interval")
    text = " ".join(row["text"] for row in transcript["segments"])
    error_rate = word_error_rate(" ".join(PHRASES), text)
    if error_rate > 0.35:
        raise RuntimeError(f"Synthetic speech WER {error_rate:.3f} exceeds 0.35; recognized: {text}")
    immutable = {path: fingerprint(job / path) for path in ["transcript.json", "aligned.wav"]}
    ids = [row["id"] for row in transcript["segments"]]
    check_args = ["recheck", job, *runtime, "--ids", *ids]
    first = cli(*check_args)
    repeat = cli(*check_args)
    if first["created_passes"] < 1 or repeat["created_passes"] or repeat["cached_passes"] != len(first["recheck_files"]):
        raise RuntimeError("Contextual recheck checkpoints did not resume")
    if not all(any(read_json(path)["segments"] for path in first["recheck_files"]
                   if read_json(path)["identity"]["method"] == method) for method in ["original", "gain"]):
        raise RuntimeError("Original/gain rechecks did not both produce speech evidence")
    review = cli("review", job, "--ids", *ids)
    if not review["recheck_context"]:
        raise RuntimeError("Contextual evidence was not exposed for review")
    resumed = cli(*run_args)["results"][0]
    if Path(resumed["job"]) != job:
        raise RuntimeError("Recognition did not resume the original job")
    if fingerprint(media) != media_digest or any(fingerprint(job / path) != value for path, value in immutable.items()):
        raise RuntimeError("Recognition or rechecks modified source audio or transcript identity")
    report = resumed["report"]
    if not report["structural_validation_passed"] or not all(Path(name).is_file() for name in report["files"]):
        raise RuntimeError("Rendered artifacts failed structural validation")
    result = {"status": "passed", "backend": backend, "device": manifest["actual_device"],
              "model": args.model, "processed_interval": [START, START + DURATION],
              "recognized_text": text, "synthetic_word_error_rate": round(error_rate, 4),
              "source_segments": len(ids), "recheck_passes": first["created_passes"],
              "cached_passes": repeat["cached_passes"], "source_and_transcript_unchanged": True,
              "structural_validation_passed": True, "manual_listening_verified": False,
              "artifacts": str(root), "note": "Synthetic operational check, not a general accuracy benchmark."}
    atomic_json(root / "smoke-report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["auto", "faster-whisper", "mlx-whisper"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "metal"], default="auto")
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--voice", default="Samantha", help="Installed macOS say voice for the English fixture")
    parser.add_argument("--work-dir", type=Path, help="Parent directory for retained synthetic smoke artifacts")
    try:
        result = smoke(parser.parse_args())
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
