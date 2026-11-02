# Installation and inference

[All releases](../README.md) · [中文](inference_zh.md) · [Deployment](deployment.md) · [SFT](training.md)

## Installation

Run from the repository root. Python 3.10+ and a CUDA GPU are recommended for 0.6B; the 0.1B model card recommends Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Choose a release

Choose one model ID below and set MODEL. Every following command uses that release. A complete local checkpoint directory is also accepted.

| Release | MODEL |
|---|---|
| Realistic Air | `Edge0/Audio8-TTS-Realistic-Air` |
| Preview 0.6B | `Edge0/Audio8-TTS-Preview-0.6b` |
| Preview 0.1B | `Edge0/Audio8-TTS-Preview-0.1b` |

```bash
export MODEL=Edge0/Audio8-TTS-Realistic-Air
# Preview 0.6B: export MODEL=Edge0/Audio8-TTS-Preview-0.6b
# Preview 0.1B: export MODEL=Edge0/Audio8-TTS-Preview-0.1b
```

If you arrived from a Preview guide, select its Preview ID. Without --model, the scripts still default to model/audio8_tts_0_6B_preview/; this guide always passes the model explicitly.

## Inference

For best synthesis quality, keep each input within 150 characters. Longer text
may reduce generation quality; split it into shorter segments when needed.

### Zero-shot voice cloning

Use your own recording at `reference.wav`. Its transcript should match the spoken content exactly.

```bash
python audio8_tts_infer.py \
  --model "${MODEL}" \
  --text "Welcome to audio8_tts." \
  --reference-audio reference.wav \
  --reference-text "Transcript of the reference recording." \
  --output outputs/clone.wav
```

### Generation without a reference

```bash
python audio8_tts_infer.py \
  --model "${MODEL}" \
  --text "This utterance does not use a reference voice." \
  --output outputs/no_reference.wav
```

### Text files and sentence iteration

Both `--text` and `--reference-text` accept an existing file path as an alternative
to inline text. Quote paths that contain spaces. For example, in PowerShell:

```powershell
python audio8_tts_infer.py `
  --model "Edge0/Audio8-TTS-Realistic-Air" `
  --text "C:\texts\my story.txt" `
  --reference-audio "reference.wav" `
  --reference-text "C:\texts\reference.txt" `
  --sentence-iteration
```

Without `--output`, this writes `my story.wav` in the current working directory.
Inline text defaults to `output_YYYYMMDD_HHMMSS_microseconds.wav`. An explicit
`--output` overrides either default. Existing outputs are skipped unless you pass
`--overwrite`.

Files are decoded into Unicode without changing the original files. UTF-8
(including a BOM), BOM-marked UTF-16/32, and Windows-1252 are supported
automatically. For other encodings, use `--text-encoding` and/or
`--reference-text-encoding`, for example `--text-encoding cp1251` or
`--reference-text-encoding utf-16-le` for a BOM-less UTF-16 file. Legacy encodings
cannot always be identified reliably, so use an explicit encoding when known.

`--sentence-iteration` splits at `.`, `!`, `?`, `。`, `！`, `？`, and `…`, retaining
punctuation and closing quotes. Each sentence is generated separately, in order,
with the same reference voice and the existing retry settings. The model loads
once, and the decoded audio is concatenated into one file without added silence
or intermediate sentence files. `--save-codes` also concatenates the sentence
codes into one `.npy` file. This is a simple punctuation splitter: abbreviations
and decimals also split, and long sentences are not subdivided automatically.
For JSONL inputs, this mode produces one concatenated audio file per row and
processes sentences individually regardless of `--batch-size`.

The same behavior is available from Python:

```python
from pathlib import Path
from audio8_tts_infer import synthesize_speech

output = synthesize_speech(
    Path(r"C:\texts\my story.txt"),
    model="Edge0/Audio8-TTS-Realistic-Air",
    reference_audio="reference.wav",
    reference_text=Path(r"C:\texts\reference.txt"),
    sentence_iteration=True,
)
print(output)
```

Strings are interpreted as files when they name an existing file; otherwise they
are inline text. In Python, pass a `Path` to require the file to exist. Additional
generation options use CLI names with underscores, such as `device="cuda"`,
`max_new_tokens=1024`, `save_codes=True`, or `overwrite=True`.

### Batch inference

Each line in the input manifest is an independent JSON object. Relative audio
paths are resolved from the manifest directory.

```json
{"id":"sample_001","text":"Target text","reference_audio":"audio/ref.wav","reference_text":"Reference transcript"}
{"id":"sample_002","text":"Text without a reference voice"}
```

```bash
python audio8_tts_infer.py \
  --model "${MODEL}" \
  --input-jsonl data/prompts.jsonl \
  --output-dir outputs/batch \
  --batch-size 2
```

The batch command writes `manifest.jsonl` and `failures.jsonl`. Existing WAV
files are skipped unless `--overwrite` is passed. See
`python audio8_tts_infer.py --help` for sampling and code-saving options.
