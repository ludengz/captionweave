# Architecture and backend boundaries

CaptionWeave separates recognition from the timestamp, job, translation, and publication rules that must remain consistent across recognition engines.

```text
Local media
  -> probe and align the playback timeline
  -> recognize resumable audio blocks through a backend
  -> immutable transcript with stable segment IDs
  -> target-specific translation requests and validated responses
  -> timed captions, artifact bundle, and quality report
```

## Media and job state

`core.py` probes media with ffprobe and extracts mono 16 kHz PCM audio with FFmpeg. Timestamp-aware resampling, padding, and trimming preserve the selected playback interval. Recognition receives blocks from this aligned audio; source-media timestamps remain the reference even for a selected range.

Job identity includes a media digest, source location, recognition options, schema/revision information, and backend cache identity. Validated completed blocks are checkpoints. Resume checks their job and time range before using them, and checks the completed transcript digest before returning cached recognition. `--restart` creates a fresh identity instead of replacing a prior job.

The backend returns recognition evidence; job orchestration owns checkpoint storage, stable IDs, source integrity checks, and the final transcript. Runtime state belongs in the work directory, not the installed package or source tree.

## Recognition adapter

`captionweave.backends.create_backend(options)` accepts `auto`, `faster-whisper`, and `mlx-whisper`. Automatic selection uses MLX on native arm64 macOS 14+ unless CPU is requested; other Macs use faster-whisper CPU, while Linux/Windows retain faster-whisper CPU/CUDA. Explicit Metal selects MLX and validates the platform; macOS rejects CUDA without probing its runtime. The CLI records the concrete backend in job options before computing job identity. Existing Linux/Windows recognition options and faster-whisper cache identities stay compatible.

There is no configurable plugin registry. Recognition dependencies and models are loaded lazily. Base translation exchange and rendering do not require them. `doctor` reports candidate backends and installed versions without importing GPU runtimes; model loading verifies the selected runtime separately.

The internal adapter interface is:

| Member | Contract |
| --- | --- |
| `block(audio, offset)` | Recognize normalized mono 16 kHz samples supplied as a standard-library `array('f')`, returning a normalized block record; `offset` is seconds on the original playback timeline |
| `recheck(audio, offset)` | Optional contextual recognition of at most 30 seconds of the same normalized samples; returns alternate segment records with playback offsets and word timing |
| `language` | Mutable source-language state; initialized from the requested language or reset for automatic detection at the start of each input |
| `actual_device` | Device actually used, recorded as execution metadata when available |
| `identity()` | Optional JSON dictionary containing at least `name`, used to distinguish implementations and relevant versions in the recognition cache |

A block record contains `segments`, `alternatives`, `digital_silence`, and `language`. Segments provide `start`, `end`, `text`, `words`, and `flags`; available confidence diagnostics may accompany them. Times in segments and words already include `offset`. Word records use `[start, end, text, probability]`. Alternate recognition uses the same timeline and segment representation, providing review evidence rather than silently replacing the primary transcript.

Each adapter's identity includes `name`, `revision`, and relevant dependency versions. MLX also records the resolved model source so changes to short-name mapping affect cache identity. Model repository names and local directory names do not fingerprint mutable weights; use stable snapshots or `--restart` after replacing a model in place. For an injected backend without `identity()`, the fallback uses its module-qualified class name, revision information, and empty dependency versions.

Adapters must not publish files, assign final source IDs, translate text, or manage target ledgers. Changes that can alter recognition results need an appropriate cache-identity change. Injected test backends can implement this interface without loading models or requiring a GPU. This is an internal extension seam, not a versioned third-party plugin API.

The faster-whisper adapter processes full-coverage windows without VAD-based removal of the playback timeline. Accurate mode gathers an additional recognition pass for flagged windows. Automatic language detection establishes language state for a recording; it does not promise independent language detection for every utterance.

`recheck.py` orchestrates bounded contextual recognition from a completed job's saved aligned audio. It merges nearby review ranges, splits spans into overlapping windows of at most 30 seconds, and compares original samples with an optional gain variant capped to avoid clipping. Window selection, gain, checkpoint storage, and evidence validation use the standard library; model execution remains in the adapter's optional `recheck` method. A backend without that method still supports ordinary recognition and fails explicitly if contextual rechecks are requested.

Recheck evidence has its own revision and binds the complete transcript digest, saved audio digest, options, backend identity, and window. Individual completed passes resume without inference. Export and review only use evidence matching the current transcript and audio, and reject corrupt checkpoints. Rechecks do not rewrite the primary transcript, assign new source IDs, or mutate translations. The existing primary recognition algorithm and cache identity are unchanged.

## Apple Silicon execution

The MLX adapter consumes the same aligned audio arrays and processes non-overlapping windows of at most 30 seconds. It converts each slice to NumPy float32, requests word timestamps, and applies the original playback offset to both words and segments. Shared quality flags remain independent of the engine. Accurate mode collects additional readings for flagged windows, and contextual rechecks use the same adapter without rewriting the original source IDs.

MLX inference runs inside `mlx.core.stream(mlx.core.gpu)`, including model loading. Startup evaluates a small GPU operation because the availability flag alone reports compiled Metal support, not usable hardware. The adapter fixes precision to float16, avoiding the upstream path-only global model cache's dtype ambiguity. It does not set the process-wide default device or use the upstream private cache directly. The pinned decoder has no beam search; it uses greedy decoding for primary windows and a bounded temperature fallback for rechecks.

Hub models are resolved with `snapshot_download(local_files_only=offline)` and a config/weights allowlist. Existing local model directories need no Hub request. The adapter requires `config.json` and `weights.safetensors` or `weights.npz`, then passes an existing directory to the upstream loader so offline recognition cannot fall through to an implicit download. Known short names map to verified MLX repositories; other models require an explicit repository or MLX directory. CTranslate2 model directories are incompatible with this format.

See [Apple Silicon setup and validation](apple-silicon.md). Ordinary macOS CI checks platform contracts and packaging; it is not evidence of real Metal ASR. WhisperKit and whisper.cpp adapters are not implemented.

## Translation exchange

`subtitles.py` exports pending stable IDs with source text, nearby context, and available alternate recognition, including overlapping alternatives for unflagged speech. Selected-ID export can revisit accepted items with their previous translation and contextual recheck evidence. Target tags are canonicalized for export and storage. Each response must echo the exported identity metadata exactly.

The source digest binds the language and ordered source IDs/text. The request digest binds the packet. Import requires exact batch coverage and accepts translated, partial, unclear, or omitted dispositions. Partial translations retain supported content with explicit uncertainty markers and reasons. The target ledger stores accepted responses and evidence-based corrections without rewriting recognition data; deliberate replacements preserve the previous ledger before updating it. See the [exchange contract](translation-exchange.md).

No model-provider client, credential lookup, translation API, or nested coding-agent process is part of this translation path. The assistant in the current conversation performs the translation and uses the CLI for validation.

## Rendering and publication

Rendering selects source text or a complete target ledger, applies accepted corrections, lays out timed captions, and produces SRT, ASS, text, JSON, and quality artifacts. Bilingual SRT and ASS are included when the target differs from the source language. Validation checks ordered, positive-duration captions and complete translation dispositions.

Publication validates the output bundle before writing: it must remain outside the job directory and must not overwrite source media. Existing artifacts receive backups. New files are staged and replaced individually with a journal supporting rollback and recovery. A publication lock coordinates writers to the same output directory. Atomicity applies to each file replacement, not simultaneous visibility of the complete bundle; unrelated readers can observe an intermediate mix of generations.

The quality report records structural checks, separate partial/unclear source counts, and review signals. Merged or split captions retain per-source translation statuses and reasons. It makes no claim of a manual listening audit. Unknown speech, hallucinations, missed dialogue, language changes, and timing ambiguity remain possible even when validation succeeds.

## Public source and private runtime data

The distributable package contains implementation and public documentation. Runtime jobs can contain extracted audio, source paths, media hashes, transcript text, alternate recognition, and translated dialogue. Published artifacts and their backups can also contain private text or metadata. Neither ignored directories nor local recognition guarantee safe sharing; review any artifact before committing or distributing it.
