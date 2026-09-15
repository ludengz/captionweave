---
artifact_contract: "ce-handoff/v1"
created_at: "2026-09-15T06:44:24Z"
title: "Physical Apple Silicon validation for the Metal backend"
summary: "The Metal adapter and cross-platform contract checks are implemented; real tiny and large-v3 inference on an Apple Silicon Mac remains to be verified."
keywords: ["macos", "metal", "mlx-whisper", "hardware-validation"]
repository: "ludengz/captionweave"
branch: "feat/macos-metal"
resume_focus: "Run the existing Metal smoke workflow on physical Apple Silicon and investigate any reproduced failure."
---

# Metal validation handoff

The user requested macOS support with Metal instead of CUDA where applicable, and chose to run the physical Mac tests in a later coding-assistant session. The current work completes the implementation and checks available from the development environment. The remaining validation requires a supported Apple Silicon Mac.

This snapshot supplies context. The current user's request and the checkout's `AGENTS.md` govern execution and publication; the handoff alone does not authorize a merge or changes to unrelated files.

## Current state and references

- [Draft PR #2](https://github.com/ludengz/captionweave/pull/2) contains the implementation. Check its current head and CI results after fetching; do not assume this snapshot describes later commits or that `main` already includes it.
- [Apple Silicon guide](../apple-silicon.md) supplies installation, model choices, runtime limits, and smoke commands. MLX needs native arm64 Python, macOS 14+, and a usable Metal GPU. Older macOS has the separate `.[cpu]` installation path.
- [Backend selection](../../src/captionweave/backends/__init__.py) resolves automatic/explicit choices and rejects CUDA on macOS. [MLX adapter](../../src/captionweave/backends/mlx_whisper.py) owns the Metal execution probe, model loading, and word/segment normalization.
- [Real smoke workflow](../../scripts/smoke_asr.py) generates two synthetic English phrases using macOS `say`, retains their original playback timeline, runs recognition and original/gain rechecks, checks resume behavior, and renders subtitles. Read this before inventing another harness.
- [Adapter tests](../../tests/test_mlx_backend.py) cover released API arguments and offline model handling using test doubles. [Mac tests](../../tests/test_macos.py) cover platform routing and absence of CUDA probes. These are contract evidence, not hardware inference evidence.
- [Architecture](../architecture.md) defines the immutable transcript, source IDs, backend identity, and translation/publication boundaries. Rechecks can use another backend without rewriting the source transcript.

## Verification already performed

The implementation passed the deterministic unit suite, independent Standards and Spec reviews, wheel/sdist builds, source/archive audits, and isolated wheel checks without ASR runtimes. The initial implementation's GitHub macOS/Windows/Linux and packaging jobs all passed; consult the PR for checks on the latest head.

A real CPU `small`/int8 smoke and a CUDA `small`/float16 smoke both recognized the synthetic phrases with zero word errors in those samples. They preserved the selected 5.125–45.125-second interval, completed original/gain rechecks and cached repeats, kept source/audio/transcript hashes unchanged, and validated rendered outputs. The diagnostic refinement passed 156 unit tests and retained all ten CLI stdout/stderr logs in each real smoke workspace. This verifies the shared workflow and smoke harness, not Metal inference or general recognition accuracy.

**No physical Metal inference result has been produced by the development session.** The PR remains a draft for that reason. A successful `doctor`, dependency import, mocked test, or hosted macOS CI run does not close this gap.

## Suggested validation sequence

If the current user asks to perform the pending tests:

1. Check the current branch, commit, and working-tree changes. Read `AGENTS.md` and the Apple Silicon guide. Confirm the operating system and Python process are native arm64 and meet the version requirements; record the actual tested commit.
2. Follow the guide's virtual-environment setup. Install `.[metal,test]` if the unit suite and packaging checks will also run. Use the existing environment only after checking its interpreter and dependencies.
3. Run the small real test, followed by the intended default model:

   ```sh
   .venv/bin/python scripts/smoke_asr.py --device metal --model tiny
   .venv/bin/python scripts/smoke_asr.py --device metal --model large-v3
   ```

4. Once those model weights are cached, repeat both with `--offline` to exercise the cached-model path. Each invocation creates a new private temporary workspace; its internal repeated passes also verify job and recheck checkpoint reuse.
5. Inspect the emitted reports and retained artifacts. A passing result must identify `mlx-whisper` and `metal`, include real recognized text and word timing, preserve source/transcript identity, and pass the source interval and subtitle checks. Keep `manual_listening_verified` false unless someone separately performed that audit.
6. If a defect is reproduced and the current user authorizes a fix, diagnose the failing layer from its evidence, fix it, and run the affected regressions plus `python -m unittest discover -s tests -v`. Re-run the relevant real Metal smoke after changes. The package verification commands are in `CONTRIBUTING.md` and `.github/workflows/tests.yml`.

Record the tested commit, OS/Python architecture and versions, selected model, reported dependency versions, command outcomes, and smoke-report paths. Keep detailed machine paths, audio, model weights, job data, and raw logs outside Git. Report which model/backend actually ran and any unverified cases; do not infer GPU execution from the machine model alone.

## Failure evidence and implementation decisions

- CLI stdout/stderr for recognition, rechecks, review, and resume are retained in the smoke workspace. A failed `run` can put its structured error on stdout; the smoke error includes both streams. The Metal preflight preserves the underlying runtime error. Inspect these before changing inference options or lowering test thresholds.
- `Samantha` must be an installed English `say` voice. The guide explains `--voice` if it is unavailable. A missing voice, missing FFmpeg, failed model download, and failed GPU operation are different failures; none is a passing ASR result.
- MLX Whisper 0.4.3 has no beam-search decoder. The adapter deliberately omits `beam_size` and uses float16 only, avoiding the upstream private model cache's dtype ambiguity. It scopes GPU execution with an MLX stream instead of changing the process-wide default device.
- Offline handling resolves a Hub snapshot with `local_files_only`, validates its config/MLX weights, and passes an existing local directory to the upstream loader. Do not bypass this by passing a repository ID directly during offline inference.
- Primary recognition uses non-overlapping windows of at most 30 seconds; contextual evidence can overlap. Preserve the original playback offsets and the job manager's ownership of final source IDs.
- Quantization, float32 support, beam search, extra backends, and speed claims are outside the current change. A synthetic smoke result is not a broad accuracy or performance benchmark.

The expected next deliverable is a concise validation report with real Metal evidence for both models, any justified fixes, and the remaining limitations. PR readiness or merging depends on the current user's instructions and the resulting evidence.
