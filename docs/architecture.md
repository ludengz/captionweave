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

`captionweave.backends.create_backend(options)` selects the recognition backend. The only accepted backend name is currently `faster-whisper`, which is also the default when `options["backend"]` is omitted; unknown names fail explicitly. There is no configurable plugin registry. The implementation uses CPU or CUDA and loads recognition dependencies and the model only when recognition needs them. Base translation exchange and rendering do not require these optional dependencies.

The internal adapter interface is:

| Member | Contract |
| --- | --- |
| `block(audio, offset)` | Recognize normalized mono 16 kHz samples supplied as a standard-library `array('f')`, returning a normalized block record; `offset` is seconds on the original playback timeline |
| `language` | Mutable source-language state; initialized from the requested language or reset for automatic detection at the start of each input |
| `actual_device` | Device actually used, recorded as execution metadata when available |
| `identity()` | Optional JSON dictionary containing at least `name`, used to distinguish implementations and relevant versions in the recognition cache |

A block record contains `segments`, `alternatives`, `digital_silence`, and `language`. Segments provide `start`, `end`, `text`, `words`, and `flags`; available confidence diagnostics may accompany them. Times in segments and words already include `offset`. Word records use `[start, end, text, probability]`. Alternate recognition uses the same timeline and segment representation, providing review evidence rather than silently replacing the primary transcript.

The real adapter's identity includes `name`, `revision`, and versions of faster-whisper, CTranslate2, and NumPy. For an injected backend without `identity()`, the fallback uses its module-qualified class name, revision information, and empty dependency versions.

Adapters must not publish files, assign final source IDs, translate text, or manage target ledgers. Changes that can alter recognition results need an appropriate cache-identity change. Injected test backends can implement this interface without loading models or requiring a GPU. This is an internal extension seam, not a versioned third-party plugin API.

The faster-whisper adapter processes full-coverage windows without VAD-based removal of the playback timeline. Accurate mode gathers an additional recognition pass for flagged windows. Automatic language detection establishes language state for a recording; it does not promise independent language detection for every utterance.

## Future Apple acceleration

MLX, WhisperKit, and whisper.cpp are possible future adapter implementations. None is implemented or selected automatically today. Adding one requires more than replacing a model call: it must normalize timing and word evidence, expose device and cache identity, honor source-language and offline choices, preserve resume behavior, and pass shared contract tests plus a real runtime smoke test. Platform-specific execution should stay within the adapter.

## Translation exchange

`subtitles.py` exports pending stable IDs with source text, nearby context, and available alternate recognition. Target tags are canonicalized for export and storage. Each response must echo the exported identity metadata exactly.

The source digest binds the language and ordered source IDs/text. The request digest binds the packet. Import requires exact batch coverage and accepts only translated, unclear, or omitted dispositions. The target ledger stores accepted responses and evidence-based corrections without rewriting recognition data. See the [exchange contract](translation-exchange.md).

No model-provider client, credential lookup, translation API, or nested coding-agent process is part of this translation path. The assistant in the current conversation performs the translation and uses the CLI for validation.

## Rendering and publication

Rendering selects source text or a complete target ledger, applies accepted corrections, lays out timed captions, and produces SRT, ASS, text, JSON, and quality artifacts. Bilingual SRT and ASS are included when the target differs from the source language. Validation checks ordered, positive-duration captions and complete translation dispositions.

Publication validates the output bundle before writing: it must remain outside the job directory and must not overwrite source media. Existing artifacts receive backups. New files are staged and replaced individually with a journal supporting rollback and recovery. A publication lock coordinates writers to the same output directory. Atomicity applies to each file replacement, not simultaneous visibility of the complete bundle; unrelated readers can observe an intermediate mix of generations.

The quality report records structural checks and review signals. It makes no claim of a manual listening audit. Unknown speech, hallucinations, missed dialogue, language changes, and timing ambiguity remain possible even when validation succeeds.

## Public source and private runtime data

The distributable package contains implementation and public documentation. Runtime jobs can contain extracted audio, source paths, media hashes, transcript text, alternate recognition, and translated dialogue. Published artifacts and their backups can also contain private text or metadata. Neither ignored directories nor local recognition guarantee safe sharing; review any artifact before committing or distributing it.
