#!/usr/bin/env python3
"""Command-line inference for audio8_tts Preview."""

from __future__ import annotations

import argparse
import codecs
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from audio8_tts_data import (
    chunks,
    clean_text,
    json_line,
    read_jsonl,
    resolve_manifest_path,
    validate_sample_id,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "model" / "audio8_tts_0_6B_preview"


@dataclass(frozen=True)
class InferenceItem:
    sample_id: str
    text: str
    output: Path
    reference_audio: Path | None = None
    reference_text: str | None = None


@dataclass(frozen=True)
class GeneratedAudio:
    waveform: np.ndarray
    codes: np.ndarray
    finished: bool


def load_text_input(
    value: str | Path, *, field_name: str = "text", encoding: str | None = None
) -> tuple[str, Path | None]:
    """Read an existing file or inline text without modifying the source file.

    BOMs identify UTF-8/16/32 files. Otherwise try UTF-8, then Windows-1252.
    Use ``encoding`` for other legacy encodings or BOM-less UTF-16/32 files.
    A Path object always means a file; strings are treated as files if they exist.
    """
    path = Path(value).expanduser()
    try:
        is_file = path.is_file()
    except (OSError, ValueError):
        # Long inline text and punctuation may not be legal filesystem paths.
        is_file = False
    if not is_file and not isinstance(value, Path):
        return clean_text(value, field_name=field_name), None

    data = path.read_bytes()
    if encoding is None:
        if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
            encoding = "utf-32"
        elif data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            encoding = "utf-16"
        else:
            try:
                return clean_text(data.decode("utf-8-sig"), field_name=field_name), path
            except UnicodeDecodeError:
                encoding = "cp1252"
    try:
        text = data.decode(encoding)
    except (UnicodeError, LookupError) as exc:
        option = field_name.replace("_", "-")
        raise ValueError(
            f"cannot decode {path} as {encoding}; set --{option}-encoding explicitly"
        ) from exc
    return clean_text(text, field_name=field_name), path


def split_sentences(text: str) -> list[str]:
    """Split at sentence punctuation, retaining punctuation and closing quotes.

    This deliberately simple splitter also splits abbreviations and decimals.
    """
    return [
        part.strip()
        for part in re.findall(r".+?(?:[.!?。！？…]+[\"'”’»」』）)\]]*|$)", clean_text(text))
        if part.strip()
    ]


def synthesize_sentences(
    item: InferenceItem,
    generate: Callable[[list[InferenceItem]], list[GeneratedAudio]],
) -> GeneratedAudio:
    """Generate each sentence independently and join only the unpadded samples."""
    parts = [
        generate([replace(item, text=sentence)])[0]
        for sentence in tqdm(split_sentences(item.text), desc="sentences", leave=False)
    ]
    return GeneratedAudio(
        waveform=np.concatenate([part.waveform for part in parts], axis=0),
        codes=np.concatenate([part.codes for part in parts], axis=1),
        finished=all(part.finished for part in parts),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize speech with audio8_tts Preview.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Inline text or an existing text file path.")
    source.add_argument("--input-jsonl", type=Path, help="Batch input manifest.")
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--reference-text", help="Inline transcript or a text file path.")
    parser.add_argument("--text-encoding", help="Override input text file encoding, e.g. cp1252.")
    parser.add_argument("--reference-text-encoding", help="Override reference transcript encoding.")
    parser.add_argument(
        "--sentence-iteration",
        action="store_true",
        help="Generate sentences one at a time and concatenate them into one audio file per input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path; defaults to <text-file-stem>.wav or output_<timestamp>.wav in cwd.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--retry-max-new-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--save-codes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    dtype = getattr(torch, value)
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU inference requires --dtype float32")
    return dtype


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.retry_max_new_tokens < args.max_new_tokens:
        raise ValueError("--retry-max-new-tokens must be >= --max-new-tokens")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if (args.reference_audio is None) != (args.reference_text is None):
        raise ValueError("--reference-audio and --reference-text must be provided together")
    if args.input_jsonl is not None and (
        args.reference_audio is not None or args.reference_text is not None
    ):
        raise ValueError("reference options belong in each batch JSONL row")
    if args.input_jsonl is not None and (
        args.text_encoding is not None or args.reference_text_encoding is not None
    ):
        raise ValueError("file encoding options apply to --text and --reference-text only")


def load_batch_items(manifest: Path, output_dir: Path) -> list[InferenceItem]:
    manifest = manifest.resolve()
    items: list[InferenceItem] = []
    seen: set[str] = set()
    for line_number, row in read_jsonl(manifest):
        try:
            sample_id = validate_sample_id(row.get("id"), fallback=f"row_{line_number:06d}")
            text = clean_text(row.get("text"), field_name="text")
            reference_value = row.get("reference_audio")
            reference_text_value = row.get("reference_text")
            if bool(reference_value) != bool(reference_text_value):
                raise ValueError("reference_audio and reference_text must be provided together")
            reference_audio = (
                resolve_manifest_path(reference_value, manifest, field_name="reference_audio")
                if reference_value
                else None
            )
            reference_text = (
                clean_text(reference_text_value, field_name="reference_text")
                if reference_text_value
                else None
            )
        except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
            raise ValueError(f"{manifest}:{line_number}: invalid inference row: {exc}") from exc
        if sample_id in seen:
            raise ValueError(f"{manifest}:{line_number}: duplicate id {sample_id!r}")
        seen.add(sample_id)
        items.append(
            InferenceItem(
                sample_id=sample_id,
                text=text,
                output=output_dir / f"{sample_id}.wav",
                reference_audio=reference_audio,
                reference_text=reference_text,
            )
        )
    if not items:
        raise ValueError(f"no inference rows found in {manifest}")
    return items


def load_items(args: argparse.Namespace) -> list[InferenceItem]:
    if args.input_jsonl is not None:
        return load_batch_items(args.input_jsonl, args.output_dir)
    reference_audio = args.reference_audio.resolve() if args.reference_audio else None
    if reference_audio is not None and not reference_audio.is_file():
        raise FileNotFoundError(f"reference audio does not exist: {reference_audio}")
    text, text_path = load_text_input(args.text, encoding=args.text_encoding)
    reference_text, reference_text_path = None, None
    if args.reference_text is not None:
        reference_text, reference_text_path = load_text_input(
            args.reference_text, field_name="reference_text", encoding=args.reference_text_encoding
        )
    output = args.output
    if output is None:
        text_file_stem = text_path.stem if text_path else f"output_{datetime.now():%Y%m%d_%H%M%S_%f}"
        ref_file_stem = reference_text_path.stem if reference_text_path else None
        output = Path(f"{ref_file_stem}_{text_file_stem}.wav") if ref_file_stem else Path(f"{text_file_stem}.wav")
    output_paths = [output, output.with_suffix(".npy")] if args.save_codes else [output]
    for input_path in (text_path, reference_text_path, reference_audio):
        if input_path is not None and any(
            target.resolve() == input_path.resolve() for target in output_paths
        ):
            raise ValueError(f"output must not overwrite an input file: {input_path}")
    return [
        InferenceItem(
            sample_id=validate_sample_id(re.sub(r"[^\w.-]", "_", output.stem), fallback="output"),
            text=text,
            output=output,
            reference_audio=reference_audio,
            reference_text=reference_text,
        )
    ]


def run_inference(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run parsed inference options, loading the model once for all inputs."""
    validate_args(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    items = load_items(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    for item in items:
        item.output.parent.mkdir(parents=True, exist_ok=True)
    batch_mode = args.input_jsonl is not None
    manifest_path = args.manifest or (args.output_dir / "manifest.jsonl")
    failures_path = args.failures or (args.output_dir / "failures.jsonl")
    if batch_mode:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        failures_path.parent.mkdir(parents=True, exist_ok=True)

    skipped = [
        item
        for item in items
        if not args.overwrite and item.output.is_file() and item.output.stat().st_size > 0
    ]
    pending = [item for item in items if item not in skipped]
    print(
        f"[audio8_tts] model={args.model} device={device} dtype={dtype} "
        f"items={len(items)} pending={len(pending)} skipped={len(skipped)}"
    )

    if not pending:
        if batch_mode:
            with manifest_path.open("w", encoding="utf-8") as handle:
                for item in skipped:
                    handle.write(
                        json_line(
                            {
                                "id": item.sample_id,
                                "status": "SKIP",
                                "output_audio": str(item.output),
                                "reference_audio": (
                                    str(item.reference_audio) if item.reference_audio else None
                                ),
                            }
                        )
                    )
            failures_path.write_text("", encoding="utf-8")
            print(f"[audio8_tts] manifest={manifest_path} failures={failures_path}")
        else:
            print(f"[audio8_tts] output={skipped[0].output} (already exists)")
        return [
            {"id": item.sample_id, "status": "SKIP", "output_audio": str(item.output)}
            for item in skipped
        ]

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=dtype).eval().to(device)
    )
    sample_rate = int(model.config.codec_sample_rate)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def synthesize(batch: list[InferenceItem], max_new_tokens: int) -> dict[str, Any]:
        processor_kwargs: dict[str, Any] = {
            "text": [item.text for item in batch],
            "return_tensors": "pt",
        }
        if batch[0].reference_audio is not None:
            processor_kwargs.update(
                reference_audio=[item.reference_audio for item in batch],
                reference_text=[item.reference_text for item in batch],
            )
        inputs = processor(**processor_kwargs)
        inputs = {name: value.to(device) for name, value in inputs.items()}
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=not args.greedy,
            generator=generator,
            return_dict_in_generate=True,
        )
        waveforms, waveform_lengths = model.decode_audio(output.codes)
        return {
            "output": output,
            "waveforms": waveforms,
            "waveform_lengths": waveform_lengths,
        }

    def generate_batch(batch: list[InferenceItem]) -> list[GeneratedAudio]:
        result = synthesize(batch, args.max_new_tokens)
        output = result["output"]
        results: list[GeneratedAudio] = []
        for index, item in enumerate(batch):
            finished = bool(output.finished[index])
            code_length = int(output.code_lengths[index])
            codes = output.codes[index, :, :code_length]
            waveform_length = int(result["waveform_lengths"][index])
            waveform = result["waveforms"][index, :waveform_length]
            if not finished and args.retry_max_new_tokens > args.max_new_tokens:
                retry = synthesize([item], args.retry_max_new_tokens)
                retry_output = retry["output"]
                finished = bool(retry_output.finished[0])
                code_length = int(retry_output.code_lengths[0])
                codes = retry_output.codes[0, :, :code_length]
                waveform_length = int(retry["waveform_lengths"][0])
                waveform = retry["waveforms"][0, :waveform_length]

            results.append(
                GeneratedAudio(
                    waveform=waveform.float().cpu().numpy(),
                    codes=codes.cpu().numpy(),
                    finished=finished,
                )
            )
        return results

    def run_batch(batch: list[InferenceItem]) -> list[dict[str, Any]]:
        results = (
            [synthesize_sentences(item, generate_batch) for item in batch]
            if args.sentence_iteration
            else generate_batch(batch)
        )
        records: list[dict[str, Any]] = []
        for item, result in zip(batch, results, strict=True):
            sf.write(item.output, result.waveform, sample_rate)
            if args.save_codes:
                np.save(item.output.with_suffix(".npy"), result.codes)
            records.append(
                {
                    "id": item.sample_id,
                    "status": "OK" if result.finished else "NO_EOS",
                    "output_audio": str(item.output),
                    "reference_audio": (
                        str(item.reference_audio) if item.reference_audio is not None else None
                    ),
                    "code_frames": int(result.codes.shape[1]),
                    "waveform_samples": int(result.waveform.shape[0]),
                    "sample_rate": sample_rate,
                }
            )
        return records

    records = [
        {
            "id": item.sample_id,
            "status": "SKIP",
            "output_audio": str(item.output),
            "reference_audio": str(item.reference_audio) if item.reference_audio else None,
        }
        for item in skipped
    ]
    failures: list[dict[str, str]] = []

    def record_failure(item: InferenceItem, exc: Exception) -> None:
        failures.append({"id": item.sample_id, "error_type": type(exc).__name__, "error": str(exc)})
        print(
            f"[audio8_tts] FAILED {item.sample_id}: {type(exc).__name__}: {exc}",
            flush=True,
        )

    # The processor treats reference conditioning as a batch-wide mode, so the
    # two groups must not be mixed in one call.
    groups = [
        [item for item in pending if item.reference_audio is None],
        [item for item in pending if item.reference_audio is not None],
    ]
    batch_size = 1 if args.sentence_iteration else args.batch_size
    total_batches = sum((len(group) + batch_size - 1) // batch_size for group in groups)
    progress = tqdm(total=total_batches, desc="audio8_tts")
    for group in groups:
        for batch_values in chunks(group, batch_size):
            batch = list(batch_values)
            try:
                records.extend(run_batch(batch))
            except Exception as batch_exc:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if args.sentence_iteration:
                    # Sentences already run individually; batch fallback would
                    # regenerate every preceding sentence after a failure.
                    record_failure(batch[0], batch_exc)
                else:
                    print(
                        f"[audio8_tts] batch fallback after "
                        f"{type(batch_exc).__name__}: {batch_exc}",
                        flush=True,
                    )
                    for item in batch:
                        try:
                            records.extend(run_batch([item]))
                        except Exception as exc:
                            record_failure(item, exc)
            progress.update(1)
    progress.close()

    if batch_mode:
        with manifest_path.open("w", encoding="utf-8") as handle:
            handle.writelines(json_line(record) for record in records)
        with failures_path.open("w", encoding="utf-8") as handle:
            handle.writelines(json_line(record) for record in failures)
        print(f"[audio8_tts] manifest={manifest_path} failures={failures_path}")
    else:
        if records:
            print(f"[audio8_tts] output={records[-1]['output_audio']}")
    if failures:
        raise RuntimeError(f"inference failed for {len(failures)} sample(s)")
    return records


def synthesize_speech(
    text: str | Path,
    *,
    reference_audio: str | Path | None = None,
    reference_text: str | Path | None = None,
    output: str | Path | None = None,
    sentence_iteration: bool = False,
    text_encoding: str | None = None,
    reference_text_encoding: str | None = None,
    **generation_options: Any,
) -> Path:
    """Convenience wrapper for one text or text file; return the output path.

    Additional inference settings use CLI names with underscores, for example
    ``model=...``, ``device="cuda"``, ``seed=42`` or ``overwrite=True``.
    Pass a Path object to require a file to exist instead of treating it as text.
    """
    args = parse_args(["--text", "placeholder"])
    allowed = {
        "model",
        "device",
        "dtype",
        "max_new_tokens",
        "retry_max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "greedy",
        "save_codes",
        "overwrite",
    }
    unknown = generation_options.keys() - allowed
    if unknown:
        raise TypeError(f"unknown inference option(s): {', '.join(sorted(unknown))}")
    vars(args).update(generation_options)
    args.text = text
    args.reference_audio = Path(reference_audio) if reference_audio is not None else None
    args.reference_text = reference_text
    args.output = Path(output) if output is not None else None
    args.sentence_iteration = sentence_iteration
    args.text_encoding = text_encoding
    args.reference_text_encoding = reference_text_encoding
    records = run_inference(args)
    return Path(records[0]["output_audio"])


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
