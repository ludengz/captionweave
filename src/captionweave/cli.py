"""CLI for local ASR and translation by the assistant in the current session."""
import argparse
import contextlib
import glob
import json
import sys
from pathlib import Path

from captionweave.subtitles import export_requests, import_response, language_tag, load_transcript, render_job
from captionweave.backends import create_backend, diagnostics, prepare_runtime
from captionweave.core import (MEDIA_EXTENSIONS, atomic_json, digest, job_lock, probe_media,
                             read_json, transcribe_media)


def expand_inputs(values, recursive=False, work_dir=None):
    found = []
    for value in values:
        matches = [value] if Path(value).exists() or not glob.has_magic(value) else glob.glob(value, recursive=recursive)
        if not matches:
            raise ValueError(f"No inputs match: {value}")
        for match in matches:
            path = Path(match).resolve()
            if path.is_dir():
                candidates = path.rglob("*") if recursive else path.iterdir()
                selected = [p for p in candidates if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
                            and not any(part.startswith(".") for part in p.relative_to(path).parts)
                            and not (work_dir and p.resolve().is_relative_to(Path(work_dir).resolve()))]
                found.extend(sorted(selected))
            elif path.is_file():
                found.append(path)
            else:
                raise ValueError(f"Input does not exist: {value}")
    found = list(dict.fromkeys(p.resolve() for p in found))
    if not found:
        raise ValueError("No audio/video files found")
    return found


def _bounded_stem(stem, source, suffix=""):
    """Keep a collision suffix intact when a long source name is shortened."""
    encoded_suffix = suffix.encode()
    if len(stem.encode()) + len(encoded_suffix) <= 180:
        return stem + suffix
    source_suffix = f"-{digest(str(source))[:8]}"
    protected = suffix if source_suffix in suffix else source_suffix + suffix
    maximum = 180 - len(protected.encode())
    if maximum < 1:
        raise ValueError("Output path suffix is too long")
    return stem.encode()[:maximum].decode("utf-8", errors="ignore") + protected


def _output_stem(source, inputs, output_dir, start, duration, disambiguator=""):
    stem = source.stem
    peers = [p for p in inputs if p.stem == stem and (output_dir or p.parent == source.parent)]
    if len(peers) > 1:
        stem += f".{source.suffix.lstrip('.')}"
        if len({p.suffix for p in peers}) < len(peers):
            stem += f"-{digest(str(source))[:8]}"
    clip_suffix = ""
    if start or duration is not None:
        clip_suffix = f".clip-{start:g}-{start + duration:g}" if duration is not None else f".from-{start:g}"
    return _bounded_stem(stem, source, f"{disambiguator}{clip_suffix}")


def default_outputs(inputs, language, output_dir=None, start=0, duration=None):
    """Allocate distinct final SRT paths for an entire batch before ASR starts."""
    sources = list(dict.fromkeys(Path(value).resolve() for value in inputs))
    output_dir = Path(output_dir).resolve() if output_dir else Path.cwd() / "outputs"
    disambiguators = {source: "" for source in sources}
    stems = {source: _output_stem(source, sources, output_dir, start, duration) for source in sources}
    for _ in range(len(sources) + 1):
        paths = {source: output_dir / f"{stem}.{language}.srt" for source, stem in stems.items()}
        groups = {}
        for source, path in paths.items():
            groups.setdefault(str(path).casefold(), []).append(source)
        collisions = [group for group in groups.values() if len(group) > 1]
        if not collisions:
            return paths
        for group in collisions:
            for source in group:
                disambiguators[source] += f"-{digest(str(source))[:8]}"
                stems[source] = _output_stem(source, sources, output_dir, start, duration, disambiguators[source])
    raise ValueError("Could not allocate unique output paths for this batch")


def default_output(source, inputs, language, output_dir=None, start=0, duration=None):
    source = Path(source).resolve()
    sources = list(inputs)
    if source not in {Path(value).resolve() for value in sources}:
        sources.append(source)
    return default_outputs(sources, language, output_dir, start, duration)[source]


def output_for_target(output, saved_target, requested_target):
    """Derive a target-specific path without replacing a previously rendered language."""
    output = Path(output)
    root = output.with_suffix("")
    suffix = f".{saved_target}" if saved_target else ""
    if suffix and root.name.endswith(suffix):
        root = root.with_name(root.name[:-len(saved_target)] + requested_target)
    else:
        root = root.with_name(f"{root.name}.{requested_target}")
    return root.parent / f"{root.name}.srt"


def build_parser():
    parser = argparse.ArgumentParser(prog="captionweave", description="Timeline-aligned transcription and assistant-driven translation.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Recognize local files, then export translation batches or source subtitles")
    run.add_argument("inputs", nargs="+")
    run.add_argument("-o", "--output", help="Explicit SRT output path for one input")
    run.add_argument("--output-dir")
    run.add_argument("--work-dir", default=str(Path.cwd() / ".captionweave" / "jobs"))
    run.add_argument("--recursive", action="store_true")
    run.add_argument("--model", default="large-v3")
    run.add_argument("--backend", choices=["faster-whisper"], default="faster-whisper")
    run.add_argument("--language", default="auto", help="Source language code, or auto (default: auto)")
    run.add_argument("--target-language", "--target", dest="target", type=language_tag)
    run.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    run.add_argument("--compute-type", default=None)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--mode", choices=["accurate", "balanced"], default="accurate",
                     help="Accurate also rechecks flagged audio windows; balanced runs one full-coverage pass")
    run.add_argument("--audio-stream", type=int, default=0, help="Zero-based index among audio streams")
    run.add_argument("--start", type=float, default=0)
    run.add_argument("--duration", type=float)
    run.add_argument("--offline", action="store_true", help="Use cached ASR models only")
    run.add_argument("--restart", action="store_true", help="Create a fresh job while preserving previous results")
    run.add_argument("--dry-run", action="store_true", help="Inspect inputs without recognition or writes")
    export = commands.add_parser("export", help="Export missing translation batches for the current assistant")
    export.add_argument("job", type=Path)
    export.add_argument("--target-language", "--target", dest="target", type=language_tag, required=True)
    export.add_argument("--batch-size", type=int, default=60)
    export.add_argument("--max-chars", type=int, default=6000)
    ingest = commands.add_parser("import", help="Validate and merge a model-written JSON response")
    ingest.add_argument("job", type=Path)
    ingest.add_argument("responses", nargs="+", type=Path)
    ingest.add_argument("--replace", action="store_true", help="Allow deliberate corrections to accepted responses")
    render = commands.add_parser("render", help="Publish validated subtitles after every item has a disposition")
    render.add_argument("job", type=Path)
    render.add_argument("--target-language", "--target", dest="target", type=language_tag)
    render.add_argument("-o", "--output", type=Path)
    render.add_argument("--font", default="sans-serif")
    render.add_argument("--unclear-text", default="[?]", help="Text for unresolved speech (default: [?])")
    render.add_argument("--size", type=int, default=58)
    render.add_argument("--width", type=int, help="Display columns per line; wide characters count as two (default: 48)")
    review = commands.add_parser("review", help="Inspect uncertain segments, word timing and alternate ASR")
    review.add_argument("job", type=Path)
    review.add_argument("--ids", nargs="*")
    review.add_argument("--offset", type=int, default=0)
    review.add_argument("--limit", type=int, default=20)
    commands.add_parser("doctor", help="Show tools and ASR package versions without loading the model runtime")
    return parser


def doctor():
    import shutil
    result = {"python": sys.executable, **diagnostics(),
              "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe")}
    return result


def run_inputs(args):
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    inputs = expand_inputs(args.inputs, args.recursive, args.work_dir)
    if args.output and (len(inputs) != 1 or args.output_dir):
        raise ValueError("-o is for one input and cannot be combined with --output-dir")
    if args.output and Path(args.output).suffix.lower() != ".srt":
        raise ValueError("-o must name an .srt file")
    if args.dry_run:
        return {"status": "dry_run", "inputs": [probe_media(p, args.audio_stream) for p in inputs]}, 0
    options = {name: getattr(args, name) for name in ["backend", "model", "language", "device", "compute_type", "batch_size",
                                                       "mode", "audio_stream", "start", "duration", "offline"]}
    planned_language = args.target or (args.language if args.language != "auto" else None)
    outputs = default_outputs(inputs, planned_language, args.output_dir, args.start) if (
        planned_language and not args.output and args.duration is None) else None
    recognizer = create_backend(options)
    results, code = [], 0
    for source in inputs:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                job = transcribe_media(source, args.work_dir, options, recognizer, args.restart)
            with job_lock(job):
                manifest = read_json(job / "manifest.json")
                transcript = load_transcript(job)
                language = args.target or transcript["language"]
                output = Path(args.output).resolve() if args.output else (
                    outputs[source] if outputs and language == planned_language else default_output(
                        source, inputs, language, args.output_dir, args.start,
                        manifest["processed_duration"] if args.duration is not None else None))
                manifest.update(default_output=str(output), target_language=args.target)
                atomic_json(job / "manifest.json", manifest)
                requests = export_requests(job, args.target) if args.target else []
                if requests:
                    results.append({"status": "translation_required", "source": str(source), "job": str(job),
                                    "output": str(output), "requests": [str(p) for p in requests]})
                    code = max(code, 3) if code != 1 else 1
                else:
                    report = render_job(job, output, args.target)
                    results.append({"status": report["status"], "source": str(source), "job": str(job), "report": report})
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            results.append({"status": "error", "source": str(source), "error": str(error)})
            code = 1
    return {"results": results}, code


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    commands = {"run", "export", "import", "render", "review", "doctor", "-h", "--help"}
    if values and values[0] not in commands:
        values.insert(0, "run")
    args = build_parser().parse_args(values)
    try:
        code = 0
        if args.command == "run":
            if argv is None and not args.dry_run:
                prepare_runtime(args.backend, args.device, values)
            result, code = run_inputs(args)
        elif args.command == "doctor":
            result = doctor()
        else:
            if not (args.job / "transcript.json").is_file():
                raise ValueError(f"No completed transcript in {args.job}")
            with job_lock(args.job):
                if args.command == "export":
                    paths = export_requests(args.job, args.target, args.batch_size, args.max_chars)
                    result = {"status": "translation_required" if paths else "ready_to_render", "requests": [str(p) for p in paths]}
                elif args.command == "import":
                    remaining = None
                    for response in args.responses:
                        remaining = import_response(args.job, response, args.replace)
                    result = {"status": "ready_to_render" if remaining == 0 else "translation_required", "remaining": remaining}
                elif args.command == "render":
                    manifest = read_json(args.job / "manifest.json")
                    target = args.target or manifest.get("target_language")
                    output = args.output or manifest.get("default_output")
                    if not output:
                        transcript = load_transcript(args.job)
                        start = manifest.get("processed_start", 0)
                        duration = manifest.get("processed_duration")
                        if duration == transcript["duration"] - start:
                            duration = None
                        output = default_output(transcript["source"], [transcript["source"]],
                                                target or transcript["language"], start=start, duration=duration)
                    elif args.target and not args.output and args.target != manifest.get("target_language"):
                        output = output_for_target(output, manifest.get("target_language"), args.target)
                    result = render_job(args.job, output, target, args.font, args.size, args.width,
                                        unclear_text=args.unclear_text)
                else:
                    transcript = load_transcript(args.job)
                    rows = [r for r in transcript["segments"] if r["id"] in args.ids] if args.ids else [r for r in transcript["segments"] if r.get("flags")]
                    if args.offset < 0 or args.limit < 1:
                        raise ValueError("Review offset must be nonnegative and limit positive")
                    chosen = rows[args.offset:args.offset + args.limit]
                    result = {"total": len(rows), "segments": chosen, "alternatives": [r for r in transcript.get("alternatives", [])
                        if any(min(r["end"], other["end"]) > max(r["start"], other["start"]) for other in chosen)]}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return code
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; completed audio blocks are saved. Repeat the same command to resume.", file=sys.stderr)
        return 130
