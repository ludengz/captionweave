# Contributing to CaptionWeave

Keep changes focused, preserve the documented CLI and translation exchange, and describe the resulting behavior with evidence. Public code, comments, identifiers, and documentation use English.

## Development environment

Use Python 3.11 or newer and a virtual environment. For work that does not execute recognition:

```sh
python -m pip install -e '.[test]'
```

For ASR development, also install the optional recognition dependencies:

```sh
python -m pip install -e '.[asr,test]'
captionweave doctor
```

Install FFmpeg and ffprobe on `PATH` when exercising media probing or extraction. Model downloads and a compatible runtime may be needed for a real recognition smoke test; they are not required by the unit test suite.

## Verification

Run the unit tests from the repository root with the package installed:

```sh
python -m unittest discover -s tests -v
```

Build and check release artifacts:

```sh
python -m build
python scripts/check_release.py
```

Tests must be deterministic and must not require credentials, a model service, model downloads, or a GPU. Use injected recognition backends for shared timing, checkpoint, and exchange behavior. Use small synthetic media and short invented multilingual dialogue in fixtures; do not copy private recordings or real transcript excerpts.

For changes to recognition or media alignment, add a small real CPU or GPU smoke test when the relevant runtime is available. Compare the selected playback interval, extracted duration, recognition timing, and output report. State exactly what was exercised and what remains unverified. A dependency check or mock-backed test does not establish model inference quality.

`python scripts/smoke_asr.py --device cpu --model small` runs that workflow on synthetic speech; on supported Apple Silicon use `--device metal --model tiny`, followed by the intended model. Use `--offline` once its weights are cached. The script uses macOS `say` or FFmpeg's `flite` filter, preserves its artifacts under a new temporary directory, and verifies word timing, original/gain evidence, checkpoint reuse, and source immutability. It is separate from the deterministic unit suite and may download model weights. See [Metal verification](docs/apple-silicon.md#verify-metal-on-your-mac).

## Change boundaries

- Preserve playback timestamps, stable source identity, validated checkpoints, and complete translation dispositions.
- Keep model-specific execution behind the backend interface. Read [docs/architecture.md](docs/architecture.md) before adding an adapter.
- Keep translations and media-specific options in job data or command arguments. Avoid per-recording scripts or hard-coded language defaults.
- Preserve existing outputs through backups and the publication recovery mechanism. Do not describe per-file replacement as a filesystem-wide atomic transaction.
- Update the [usage guide](docs/usage.md) and the [translation exchange documentation](docs/translation-exchange.md) when the public contract changes.

## Data hygiene and reviews

Do not commit media, recognition jobs, extracted audio, real transcripts, subtitle output, logs containing dialogue, credentials, personal paths, or machine-specific details. The default `.captionweave/` and `outputs/` directories are ignored, but custom paths and manually added files still require review. Inspect both the proposed changes and built artifacts; passing a release scan is useful evidence, not a complete privacy audit.

A contribution description should state the problem, resulting behavior, meaningful verification, and any unresolved limitations. Report failed or unavailable checks accurately. Do not claim full listening verification from structural checks or confidence scores.

For a suspected vulnerability, follow [SECURITY.md](SECURITY.md) before posting sensitive details.
