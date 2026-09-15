"""Shared diagnostics for normalized recognition evidence."""
import importlib.metadata
import re


def package_versions(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def quality_flags(row):
    flags = []
    if row.get("avg_logprob", 0) < -0.8:
        flags.append("low_confidence")
    if row.get("no_speech_prob", 0) > 0.6:
        flags.append("possible_non_speech")
    if row.get("compression_ratio", 0) > 2.4:
        flags.append("repetition")
    if row["end"] <= row["start"] or row["end"] - row["start"] > 12:
        flags.append("timing")
    words = row.get("words") or []
    if any(right[0] - left[1] > 2 for left, right in zip(words, words[1:])):
        flags.append("word_timing_gap")
    if re.search(r"ご視聴ありがとうございました|字幕.*作成|thanks for watching|subscribe to", row["text"], re.I):
        flags.append("possible_boilerplate")
    return flags
