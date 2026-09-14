"""Bounded local rechecks that preserve source IDs and accepted translations."""
import hashlib
from array import array
import math
import wave
from pathlib import Path

from captionweave.backends import backend_identity, create_backend
from captionweave.core import atomic_json, digest, read_audio, read_json, wav_duration


def context_windows(rows, start, duration, context=5):
    if not math.isfinite(context) or not 0 <= context <= 15:
        raise ValueError("Recheck context must be between 0 and 15 seconds")
    finish = start + duration
    ranges = []
    for row in rows:
        left, right = row["start"], row["end"]
        if not all(math.isfinite(v) for v in [left, right]) or right < start or left > finish:
            raise ValueError(f"Segment {row.get('id')} is outside the saved audio timeline")
        left, right = max(start, left - context), min(finish, max(left + 0.1, right) + context)
        if right <= left:
            raise ValueError("No saved audio for this recheck")
        ranges.append((left, right))
    merged = []
    for left, right in sorted(ranges):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    windows = []
    for left, right in merged:
        while right - left > 30:
            windows.append((left, left + 30))
            left += 20
        windows.append((left, right))
    return windows


def _validate_gain_db(gain_db):
    if not math.isfinite(gain_db) or not 0 <= gain_db <= 18:
        raise ValueError("Recheck gain must be between 0 and 18 dB")


def bounded_gain(audio, gain_db):
    _validate_gain_db(gain_db)
    peak = max(map(abs, audio), default=0)
    if peak < 1e-5:
        return audio, 0.0
    factor = max(1.0, min(10 ** (gain_db / 20), 0.98 / peak))
    if factor == 1:
        return audio, 0.0
    return array("f", (sample * factor for sample in audio)), 20 * math.log10(factor)


def _read_checkpoint(path):
    value = read_json(path)
    if (not isinstance(value, dict) or not isinstance(value.get("identity"), dict)
            or not isinstance(value.get("segments"), list)):
        raise ValueError(f"Recheck evidence is corrupt: {path}")
    signed = {key: item for key, item in value.items() if key != "evidence_digest"}
    if (value.get("schema_version") != 1 or digest(signed) != value.get("evidence_digest")
            or digest(value.get("identity"))[:24] != path.stem):
        raise ValueError(f"Recheck evidence is corrupt: {path}")
    return value


def load_recheck_evidence(job, transcript):
    revision = digest(transcript)
    paths = sorted((Path(job) / "rechecks").glob("*.json"))
    if not paths:
        return []
    with (Path(job) / "aligned.wav").open("rb") as stream:
        audio_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    rows = []
    for path in paths:
        value = _read_checkpoint(path)
        identity = value["identity"]
        if (identity["transcript_digest"] != revision or identity["job_id"] != transcript["job_id"]
                or identity["audio_sha256"] != audio_digest):
            continue
        for row in value["segments"]:
            rows.append({**row, "recheck_id": path.stem, "method": identity["method"],
                         "language": identity["options"]["language"], "model": identity["options"]["model"],
                         "window": identity["window"], "applied_gain_db": value["applied_gain_db"]})
    return rows


def window_evidence(evidence, ranges):
    return [row for row in evidence if any(
        min(max(left + 0.001, right), row["window"][1]) > max(left, row["window"][0])
        for left, right in ranges)]


def recheck_audio(job, transcript, rows, options, context=5, gain_db=9, recognizer=None):
    job = Path(job)
    manifest = read_json(job / "manifest.json")
    revision = digest(transcript)
    if (not manifest.get("asr_complete") or manifest.get("transcript_digest") != revision
            or manifest.get("job_id") != transcript["job_id"]):
        raise ValueError("Completed source transcript is missing or changed; run ASR again")
    audio_path = job / "aligned.wav"
    start, duration = manifest["processed_start"], manifest["processed_duration"]
    if abs(wav_duration(audio_path) - duration) > 0.05:
        raise ValueError("Saved audio is incomplete; run ASR again")
    with wave.open(str(audio_path)) as audio:
        if audio.getnframes():
            audio.setpos(audio.getnframes() - 1)
        if len(audio.readframes(1)) != 2:
            raise ValueError("Saved audio is incomplete; run ASR again")
    windows = context_windows(rows, start, duration, context)
    _validate_gain_db(gain_db)
    with audio_path.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    engine = recognizer if recognizer is not None else create_backend(options)
    if not callable(getattr(engine, "recheck", None)):
        raise ValueError("This ASR backend does not support contextual rechecks")
    backend = backend_identity(engine)
    paths, cached, created, skipped = [], 0, 0, 0
    for left, right in windows:
        audio = read_audio(audio_path, left - start, right - left)
        gained, applied_gain = bounded_gain(audio, gain_db)
        variants = [("original", audio, 0.0)]
        if applied_gain >= 1 - 1e-9:
            variants.append(("gain", gained, applied_gain))
        elif gain_db:
            skipped += 1
        for method, samples, applied in variants:
            identity = {"job_id": transcript["job_id"], "transcript_digest": revision,
                        "audio_sha256": fingerprint, "recheck_revision": 1,
                        "options": options, "backend": backend, "window": [left, right],
                        "method": method, "gain_db": gain_db if method == "gain" else 0}
            path = job / "rechecks" / f"{digest(identity)[:24]}.json"
            if path.exists():
                if _read_checkpoint(path)["identity"] != identity:
                    raise ValueError("Recheck checkpoint identity mismatch")
                cached += 1
            else:
                print(f"Rechecking {left:.2f}-{right:.2f}s ({method}, +{applied:.1f} dB)", flush=True)
                segments = engine.recheck(samples, left)
                value = {"schema_version": 1, "identity": identity, "applied_gain_db": round(applied, 4),
                         "segments": segments}
                value["evidence_digest"] = digest(value)
                atomic_json(path, value)
                created += 1
            paths.append(str(path))
    return {"recheck_files": paths, "created_passes": created, "cached_passes": cached,
            "gain_skipped_windows": skipped, "windows": windows}
