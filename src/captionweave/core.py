"""Local, resumable speech recognition on the media's playback timeline."""
import contextlib
from array import array
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from captionweave.backends import backend_identity, create_backend


SCHEMA_VERSION = 1
SAMPLE_RATE = 16000
BLOCK_SECONDS = 600
MEDIA_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv",
                    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}


def timestamp(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("Timestamp must be finite and nonnegative")
    total = round(value * 1000)
    hours, total = divmod(total, 3600000)
    minutes, total = divmod(total, 60000)
    seconds, milliseconds = divmod(total, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextlib.contextmanager
def job_lock(job):
    job = Path(job)
    job.mkdir(parents=True, exist_ok=True)
    with (job / ".lock").open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another process is using job {job}") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)


def probe_media(source, audio_stream=0):
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError(f"Input is not a local file: {source}")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("Install FFmpeg and put ffmpeg/ffprobe on PATH")
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,start_time:stream=index,codec_type,codec_name,start_time,duration,sample_rate,channels:stream_tags=language",
        "-of", "json", str(source)], capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"Cannot probe {source.name}: {result.stderr.strip()}")
    info = json.loads(result.stdout)
    audio = [s for s in info.get("streams", []) if s["codec_type"] == "audio"]
    if audio_stream < 0 or audio_stream >= len(audio):
        raise ValueError(f"Audio stream {audio_stream} does not exist; found {len(audio)} audio streams")
    try:
        duration = float(info["format"]["duration"])
    except (KeyError, ValueError) as error:
        raise ValueError("The input must have a known, finite duration") from error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("The input duration must be positive and finite")
    return {"path": str(source), "duration": duration, "audio_stream": audio_stream,
            "audio": audio[audio_stream], "audio_stream_count": len(audio),
            "container_start": float(info.get("format", {}).get("start_time", 0))}


def selected_duration(info, start=0, duration=None):
    if not math.isfinite(start) or not 0 <= start < info["duration"]:
        raise ValueError("--start must be inside the media timeline")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("--duration must be positive and finite")
    return min(info["duration"] - start, duration if duration is not None else info["duration"])


def wav_duration(path):
    with wave.open(str(path)) as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError("Aligned audio must be mono 16 kHz PCM16")
        return audio.getnframes() / SAMPLE_RATE


def extract_audio(source, output, info, start=0, duration=None):
    span = selected_duration(info, start, duration)
    if span * SAMPLE_RATE * 2 >= 2**32 - 1024:
        raise ValueError("Select a range shorter than 37 hours with --start/--duration")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".aligned-", suffix=".wav", dir=output.parent)
    os.close(fd)
    try:
        command = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error"]
        if start:
            command += ["-ss", str(start)]
        command += ["-i", str(source), "-map", f"0:a:{info['audio_stream']}", "-vn",
                    "-af", f"aresample={SAMPLE_RATE}:async=1000:first_pts=0,apad,atrim=duration={span:.9f}",
                    "-ac", "1", "-c:a", "pcm_s16le", "-y", temporary]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Audio extraction failed: {result.stderr.strip()}")
        if abs(wav_duration(temporary) - span) > 0.05:
            raise RuntimeError("Extracted audio does not match the selected playback timeline")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return span


def clip_windows(duration):
    result, start = [], 0.0
    while duration - start >= 0.1:
        end = min(start + 30, duration)
        result.append({"start": start, "end": end})
        start = end
    return result


def read_audio(path, start, duration):
    with wave.open(str(path)) as audio:
        audio.setpos(min(round(start * SAMPLE_RATE), audio.getnframes()))
        data = audio.readframes(round(duration * SAMPLE_RATE))
    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    return array("f", (sample / 32768 for sample in samples))


def transcribe_media(source, work_root, options, recognizer=None, restart=False):
    source = Path(source).resolve()
    before = source.stat()
    info = probe_media(source, options["audio_stream"])
    info["stat"] = {"size": before.st_size, "mtime_ns": before.st_mtime_ns}
    span = selected_duration(info, options["start"], options["duration"])
    engine = recognizer if recognizer is not None else create_backend(options)
    backend = backend_identity(engine)
    print(f"Fingerprinting {source.name}", flush=True)
    with source.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    if (source.stat().st_size, source.stat().st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise ValueError("Input changed while it was being inspected; retry after the download/edit finishes")
    identity = {"schema_version": SCHEMA_VERSION, "asr_revision": 2, "source_sha256": fingerprint,
                "source_path": str(source), "options": options, "backend": backend}
    if restart:
        identity["restart"] = time.time_ns()
    job_id = digest(identity)[:20]
    stem = re.sub(r"[^\w.-]+", "_", source.stem).encode()[:160].decode("utf-8", errors="ignore") or "media"
    job = Path(work_root).resolve() / f"{stem}-{job_id}"
    with job_lock(job):
        manifest_path = job / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["identity"] != identity:
                raise ValueError("Job identity mismatch; use --restart")
            if (job / "transcript.json").exists() and manifest.get("asr_complete"):
                if digest(read_json(job / "transcript.json")) != manifest.get("transcript_digest"):
                    raise ValueError("Cached transcript was changed or corrupted; use --restart")
                if manifest["media"]["stat"] != info["stat"]:
                    manifest["media"]["stat"] = info["stat"]
                    atomic_json(manifest_path, manifest)
                print(f"Resuming completed ASR: {job}", flush=True)
                return job
            if manifest["media"]["stat"] != info["stat"]:
                manifest["media"]["stat"] = info["stat"]
                atomic_json(manifest_path, manifest)
        else:
            manifest = {"schema_version": SCHEMA_VERSION, "job_id": job_id, "identity": identity,
                        "media": info, "processed_start": options["start"], "processed_duration": span,
                        "asr_complete": False}
            atomic_json(manifest_path, manifest)
        audio_path = job / "aligned.wav"
        if not audio_path.exists():
            print(f"Aligning audio: {span:.2f}s at playback offset {options['start']:.2f}s", flush=True)
            extract_audio(source, audio_path, info, options["start"], span)
        if abs(wav_duration(audio_path) - span) > 0.05:
            raise ValueError("Cached audio is incomplete or corrupt; use --restart")
        engine.language = None if options["language"] == "auto" else options["language"]
        rows, alternatives, processed = [], [], []
        for index, left in enumerate(range(0, math.ceil(span), BLOCK_SECONDS)):
            right = min(left + BLOCK_SECONDS, span)
            checkpoint = job / "blocks" / f"{index:06}.json"
            if checkpoint.exists():
                block = read_json(checkpoint)
                if block.get("range") != [left, right] or block.get("job_id") != job_id:
                    raise ValueError("Invalid block checkpoint; use --restart")
            else:
                block = engine.block(read_audio(audio_path, left, right - left), options["start"] + left)
                block.update(range=[left, right], job_id=job_id)
                atomic_json(checkpoint, block)
            engine.language = block.get("language") or engine.language
            rows.extend(block["segments"])
            alternatives.extend(block["alternatives"])
            processed.append([options["start"] + left, options["start"] + right])
            print(f"{source.name}: {right:.0f}/{span:.0f}s; {len(rows)} segments", flush=True)
        rows.sort(key=lambda r: (r["start"], r["end"]))
        if (source.stat().st_size, source.stat().st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise ValueError("Input changed during transcription; rerun with the completed input")
        for index, row in enumerate(rows, 1):
            row.update(id=f"s{index:06}", i=index)
        transcript = {"schema_version": SCHEMA_VERSION, "job_id": job_id, "source": str(source),
                      "source_sha256": fingerprint, "duration": info["duration"],
                      "language": engine.language or ("und" if options["language"] == "auto" else options["language"]), "processed_intervals": processed,
                      "segments": rows, "alternatives": alternatives}
        atomic_json(job / "transcript.json", transcript)
        atomic_json(job / "review.json", {"job_id": job_id, "segments": [r for r in rows if r["flags"]],
                                          "alternatives": alternatives,
                                          "note": "ASR confidence is a review signal, not proof of accuracy."})
        manifest.update(asr_complete=True, segment_count=len(rows), actual_device=engine.actual_device,
                        transcript_digest=digest(transcript))
        atomic_json(manifest_path, manifest)
        return job
