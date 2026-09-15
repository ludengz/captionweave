# Apple Silicon and Metal

CaptionWeave includes an MLX Whisper backend for Apple GPU recognition. It requires macOS 14 or newer, native arm64 Python 3.11+, and an accessible Metal GPU. Intel Macs and older macOS installations use faster-whisper on CPU. macOS never selects CUDA.

## Install

From a source checkout, with Homebrew available:

```sh
brew install python@3.13 ffmpeg
python3.13 -m venv .venv
.venv/bin/python -m pip install -e '.[metal]'
.venv/bin/captionweave doctor
```

Use an arm64 terminal and Python, rather than a Python running under Rosetta. `doctor` should select `mlx-whisper` and list the installed MLX packages. It deliberately leaves the Metal runtime unloaded; inference performs a GPU operation before using the model.

`.[asr]` installs both MLX and faster-whisper on Apple Silicon if CPU fallback is also needed. With that extra, `--device cpu` selects CPU explicitly. If Metal is unavailable, the CLI reports the error and recommends CPU; it does not silently change engines.

For an older macOS installation, use the CPU-only extra so the installer does not request incompatible MLX wheels:

```sh
.venv/bin/python -m pip install -e '.[cpu]'
.venv/bin/captionweave run lecture.mp4 --device cpu
```

Do not use `.[asr]` on an Apple Silicon Mac running macOS older than 14: that extra includes MLX. Intel Macs can use either `.[cpu]` or `.[asr]`.

## Recognize and review

```sh
.venv/bin/captionweave run lecture.mp4 --device metal --language en --target zh
.venv/bin/captionweave recheck JOB --backend mlx-whisper --device metal --target zh
```

Automatic backend selection on a supported Mac has the same Metal route, so `--device metal` can be omitted. Translation still uses the current assistant through request/response files. It does not run inside MLX Whisper.

The default model is `large-v3`, resolved to `mlx-community/whisper-large-v3-mlx`. Supported short names are `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v2`, `large-v3`, `large`, `large-v3-turbo`, and `turbo`. Known names use explicit repository mappings; an unknown short name is rejected. A full Hub repository ID or existing local MLX model directory is also accepted.

Weights may download on first use. `--offline` uses only cached weights; an incomplete cached directory fails before inference. A local model needs `config.json` and either `weights.safetensors` or `weights.npz`. Faster-whisper/CTranslate2 weights cannot be reused as MLX weights. Use a stable local snapshot, or start a new job with `--restart` after replacing model files in place.

MLX currently supports float16 only. `--batch-size` affects faster-whisper; MLX runs windows serially. The pinned MLX Whisper decoder has no beam search, so recognition can differ from the CPU/CUDA backend even with the corresponding Whisper model. Timing preservation, stable source IDs, review evidence, and translation validation share the same job workflow.

## Verify Metal on your Mac

Run a small real inference test first:

```sh
.venv/bin/python scripts/smoke_asr.py --device metal --model tiny
```

Then verify the model you intend to use:

```sh
.venv/bin/python scripts/smoke_asr.py --device metal --model large-v3
```

The test uses the installed macOS `Samantha` voice to generate two short English sentences. If that voice is unavailable, select an installed English voice with `--voice` (`say -v '?'` lists voices). It creates a new temporary directory and retains its synthetic audio, jobs, subtitles, CLI stdout/stderr logs, and `smoke-report.json` for inspection. Errors retain the underlying Metal initialization or CLI failure message. Reports include Python and recognition dependency versions.

A passing JSON result reports `backend: "mlx-whisper"` and `device: "metal"`. The test checks actual inference with word timestamps, a nonzero playback offset spanning two recognition windows, original/gain contextual evidence, cached rechecks, unchanged source/transcript hashes, and rendered artifacts. The synthetic word-error rate is a small smoke-test guard, not a general accuracy benchmark or manual listening audit.

Normal macOS CI validates deterministic backend contracts, dependency installation, and packaging. It does not establish that a physical Metal GPU can execute the model. A real result from this script is required before reporting hardware inference as verified.

For a fresh coding-assistant session on the Mac, the [validation handoff](handoffs/macos-metal-validation.md) records the implementation decisions, checks already performed, remaining hardware evidence, and locations to inspect if a test fails.

## Implementation references

- [MLX installation requirements](https://ml-explore.github.io/mlx/build/html/install.html) and [GPU stream API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.stream.html).
- [MLX Whisper 0.4.3 release metadata](https://pypi.org/project/mlx-whisper/0.4.3/) and [official implementation](https://github.com/ml-explore/mlx-examples/tree/main/whisper).
- [CTranslate2 platform support](https://opennmt.net/CTranslate2/installation.html): macOS uses CPU; documented CUDA wheels target Linux and Windows.
- [MLX's own CI](https://github.com/ml-explore/mlx/blob/v0.32.2/.github/workflows/build_and_test.yml) separates standard macOS CPU checks from hardware Metal tests.
