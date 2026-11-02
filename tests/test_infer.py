"""Exercise inference orchestration with real WAV I/O and no model download.

Run with ``python -m unittest discover -s tests -p test_infer.py``.
"""

import codecs
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def load_inference_module():
    # Isolate optional ML imports, including the shared data module, so this
    # suite can run without torch/transformers and cannot affect other suites.
    with patch.dict(
        "sys.modules",
        {
            "torch": MagicMock(),
            "torch.nn": MagicMock(),
            "torch.nn.functional": MagicMock(),
            "transformers": MagicMock(),
        },
    ) as modules:
        for name, filename in (
            ("audio8_tts_data", "audio8_tts_data.py"),
            ("_audio8_tts_infer_test", "audio8_tts_infer.py"),
        ):
            spec = importlib.util.spec_from_file_location(name, ROOT / filename)
            module = importlib.util.module_from_spec(spec)
            modules[name] = module
            spec.loader.exec_module(module)
        return module


infer = load_inference_module()


class FakeTensor(np.ndarray):
    def to(self, device):
        return self

    def cpu(self):
        return self

    def float(self):
        return self.astype(np.float32)

    def numpy(self):
        return np.asarray(self)


class FakeBackend:
    """Return distinctive, padded audio/codes so ordering and trimming matter."""

    def __init__(self, finished=()):
        self.finished = list(finished)
        self.calls = []
        self.processor = MagicMock(side_effect=self.process)
        self.model = MagicMock()
        self.model.eval.return_value = self.model
        self.model.to.return_value = self.model
        self.model.config.codec_sample_rate = 24000
        self.model.generate.side_effect = self.generate
        self.model.decode_audio.side_effect = lambda codes: self.decoded

    def process(self, **kwargs):
        self.latest = kwargs
        return {"input_ids": np.zeros((len(kwargs["text"]), 1)).view(FakeTensor)}

    def generate(self, **kwargs):
        index = len(self.calls)
        self.calls.append({**self.latest, "max_new_tokens": kwargs["max_new_tokens"]})
        count = len(self.latest["text"])
        samples, frames = index + 2, index + 1
        waveforms = np.full((count, samples + 5), 0.99, dtype=np.float32)
        waveforms[:, :samples] = (index + 1) / 10
        codes = np.full((count, 2, frames + 5), -999, dtype=np.int64)
        codes[:, :, :frames] = index + 1
        self.decoded = waveforms.view(FakeTensor), [samples] * count
        finished = self.finished[index] if index < len(self.finished) else True
        return SimpleNamespace(
            codes=codes.view(FakeTensor),
            code_lengths=[frames] * count,
            finished=[finished] * count,
        )


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        torch.device.side_effect = lambda value: SimpleNamespace(type=value.split(":")[0])
        self.stack.enter_context(patch.object(infer, "torch", torch))

    def backend(self, finished=()):
        backend = FakeBackend(finished)
        self.model_loader = self.stack.enter_context(
            patch.object(infer.AutoModel, "from_pretrained", return_value=backend.model)
        )
        self.stack.enter_context(
            patch.object(infer.AutoProcessor, "from_pretrained", return_value=backend.processor)
        )
        return backend

    def test_file_encodings_and_source_preservation(self):
        text = "Grüße – café.\r\nNext sentence!"
        for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-32", "cp1252"):
            with self.subTest(encoding=encoding):
                path = self.root / f"input-{encoding}.txt"
                original = text.encode(encoding)
                path.write_bytes(original)
                decoded, source = infer.load_text_input(str(path))
                self.assertEqual(decoded, "Grüße – café. Next sentence!")
                self.assertEqual(source, path)
                self.assertEqual(path.read_bytes(), original)
        for bom, encoding in (
            (codecs.BOM_UTF16_BE, "utf-16-be"),
            (codecs.BOM_UTF32_BE, "utf-32-be"),
        ):
            path.write_bytes(bom + "你好。".encode(encoding))
            self.assertEqual(infer.load_text_input(path)[0], "你好。")

    def test_explicit_encoding_and_decode_errors(self):
        path = self.root / "input.txt"
        path.write_bytes("Привет!".encode("cp1251"))
        self.assertEqual(infer.load_text_input(path, encoding="cp1251")[0], "Привет!")
        path.write_bytes("Grüße!".encode("utf-16-le"))
        self.assertEqual(infer.load_text_input(path, encoding="utf-16-le")[0], "Grüße!")
        with self.assertRaisesRegex(ValueError, "--reference-text-encoding"):
            infer.load_text_input(path, field_name="reference_text", encoding="not-an-encoding")
        path.write_bytes(b"\x81")
        with self.assertRaisesRegex(ValueError, "--text-encoding"):
            infer.load_text_input(path)

    def test_inline_long_text_empty_files_and_missing_path_objects(self):
        text = "Hello! " * 1000
        self.assertEqual(infer.load_text_input(text), (text.strip(), None))
        with self.assertRaises(FileNotFoundError):
            infer.load_text_input(self.root / "missing.txt")
        path = self.root / "empty.txt"
        path.write_text(" \n\t", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            infer.load_text_input(path)

    def test_default_names_and_explicit_output(self):
        path = self.root / "my story.txt"
        path.write_text("Hello.", encoding="utf-8")
        args = infer.parse_args(["--text", str(path)])
        item = infer.load_items(args)[0]
        self.assertEqual(item.output, Path("my story.wav"))
        self.assertEqual(item.text, "Hello.")
        item = infer.load_items(infer.parse_args(["--text", "Hello."]))[0]
        self.assertRegex(item.output.name, r"^output_\d{8}_\d{6}_\d{6}\.wav$")
        args.output = self.root / "custom name.wav"
        self.assertEqual(infer.load_items(args)[0].output, args.output)
        args.output = path
        with self.assertRaisesRegex(ValueError, "overwrite an input"):
            infer.load_items(args)
        sidecar_input = self.root / "codes.npy"
        sidecar_input.write_text("Text in a file with a custom extension.", encoding="utf-8")
        args.text, args.output, args.save_codes = str(sidecar_input), self.root / "codes.wav", True
        with self.assertRaisesRegex(ValueError, "overwrite an input"):
            infer.load_items(args)

    def test_sentence_splitting(self):
        cases = (
            ("First. Second?! Last", ["First.", "Second?!", "Last"]),
            ('She said "Hello!" Next.', ['She said "Hello!"', "Next."]),
            ("你好。世界！再见？", ["你好。", "世界！", "再见？"]),
            ("Wait… Really... Yes!", ["Wait…", "Really...", "Yes!"]),
            ("No punctuation\njust text", ["No punctuation just text"]),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(infer.split_sentences(text), expected)

    def test_wrapper_sentence_retry_reference_and_concatenation(self):
        backend = self.backend(finished=[False, True, True])
        text = self.root / "story.txt"
        text.write_bytes("First. Second!".encode("utf-16"))
        transcript = self.root / "reference.txt"
        transcript.write_bytes("Grüße vom Sprecher.".encode("cp1252"))
        reference = self.root / "reference.wav"
        sf.write(reference, np.zeros(4), 24000)
        output = self.root / "joined.wav"
        result = infer.synthesize_speech(
            text,
            reference_audio=reference,
            reference_text=transcript,
            output=output,
            sentence_iteration=True,
            max_new_tokens=10,
            retry_max_new_tokens=20,
            save_codes=True,
        )
        self.assertEqual(result, output)
        self.model_loader.assert_called_once()
        self.assertEqual([c["text"] for c in backend.calls], [["First."], ["First."], ["Second!"]])
        self.assertEqual([c["max_new_tokens"] for c in backend.calls], [10, 20, 10])
        for call in backend.calls:
            self.assertEqual(call["reference_audio"], [reference.resolve()])
            self.assertEqual(call["reference_text"], ["Grüße vom Sprecher."])
        audio, rate = sf.read(output)
        self.assertEqual(rate, 24000)
        np.testing.assert_allclose(audio, [0.2] * 3 + [0.3] * 4, atol=1 / 32768)
        np.testing.assert_array_equal(np.load(output.with_suffix(".npy")), [[2, 2, 3, 3, 3]] * 2)
        self.assertEqual(
            sorted(p.name for p in self.root.glob("joined*")), ["joined.npy", "joined.wav"]
        )

    def test_non_iteration_remains_one_call_and_existing_output_is_skipped(self):
        backend = self.backend()
        output = self.root / "whole.wav"
        infer.synthesize_speech("First. Second!", output=output)
        self.assertEqual([c["text"] for c in backend.calls], [["First. Second!"]])
        self.assertEqual(infer.synthesize_speech("Other text.", output=output), output)
        self.assertEqual(len(backend.calls), 1)
        infer.synthesize_speech("Other text.", output=output, overwrite=True)
        self.assertEqual(len(backend.calls), 2)

    def test_cli_file_inputs_and_encoding_overrides(self):
        backend = self.backend()
        text = self.root / "story.txt"
        text.write_bytes("Привет! Пока.".encode("cp1251"))
        transcript = self.root / "reference.txt"
        transcript.write_bytes("Grüße!".encode("utf-16-le"))
        reference = self.root / "reference.wav"
        sf.write(reference, np.zeros(4), 24000)
        output = self.root / "cli.wav"
        with patch(
            "sys.argv",
            [
                "audio8_tts_infer.py",
                "--text",
                str(text),
                "--text-encoding",
                "cp1251",
                "--reference-audio",
                str(reference),
                "--reference-text",
                str(transcript),
                "--reference-text-encoding",
                "utf-16-le",
                "--output",
                str(output),
                "--sentence-iteration",
            ],
        ):
            infer.main()
        self.assertTrue(output.is_file())
        self.assertEqual([c["text"] for c in backend.calls], [["Привет!"], ["Пока."]])
        self.assertEqual([c["reference_text"] for c in backend.calls], [["Grüße!"], ["Grüße!"]])

    def test_regular_batch_preserves_reference_groups_and_skip_manifest(self):
        backend = self.backend()
        reference = self.root / "reference.wav"
        sf.write(reference, np.zeros(4), 24000)
        rows = [
            {"id": "a", "text": "First. Second!"},
            {"id": "b", "text": "Hello."},
            {
                "id": "c",
                "text": "Goodbye.",
                "reference_audio": "reference.wav",
                "reference_text": "Transcript.",
            },
        ]
        manifest = self.root / "input.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        output_dir = self.root / "outputs"
        args = infer.parse_args(
            [
                "--input-jsonl",
                str(manifest),
                "--output-dir",
                str(output_dir),
                "--batch-size",
                "2",
            ]
        )
        records = infer.run_inference(args)
        self.assertEqual(
            [c["text"] for c in backend.calls], [["First. Second!", "Hello."], ["Goodbye."]]
        )
        self.assertNotIn("reference_audio", backend.calls[0])
        self.assertEqual(backend.calls[1]["reference_audio"], [reference.resolve()])
        self.assertEqual([r["id"] for r in records], ["a", "b", "c"])
        self.assertEqual([r["waveform_samples"] for r in records], [2, 2, 3])
        skipped = infer.run_inference(args)
        self.assertEqual([r["status"] for r in skipped], ["SKIP"] * 3)
        self.model_loader.assert_called_once()
        saved = [
            json.loads(line) for line in (output_dir / "manifest.jsonl").read_text().splitlines()
        ]
        self.assertEqual([r["status"] for r in saved], ["SKIP"] * 3)

    def test_batch_sentence_iteration_and_unfinished_status(self):
        backend = self.backend(finished=[True, False, True])
        manifest = self.root / "input.jsonl"
        manifest.write_text(
            json.dumps({"id": "a", "text": "First. Second!"})
            + "\n"
            + json.dumps({"id": "b", "text": "Third?"})
            + "\n",
            encoding="utf-8",
        )
        output_dir = self.root / "outputs"
        args = infer.parse_args(
            [
                "--input-jsonl",
                str(manifest),
                "--output-dir",
                str(output_dir),
                "--sentence-iteration",
                "--batch-size",
                "3",
                "--max-new-tokens",
                "10",
                "--retry-max-new-tokens",
                "10",
            ]
        )
        records = infer.run_inference(args)
        self.assertEqual([c["text"] for c in backend.calls], [["First."], ["Second!"], ["Third?"]])
        self.assertEqual([r["status"] for r in records], ["NO_EOS", "OK"])
        self.assertEqual([r["waveform_samples"] for r in records], [5, 4])
        self.assertEqual([r["code_frames"] for r in records], [3, 3])
        self.assertEqual(len(list(output_dir.glob("*.wav"))), 2)
        saved = [
            json.loads(line) for line in (output_dir / "manifest.jsonl").read_text().splitlines()
        ]
        self.assertEqual(saved, records)
        self.assertEqual((output_dir / "failures.jsonl").read_text(), "")

    def test_sentence_failure_does_not_write_partial_audio(self):
        backend = self.backend()
        generate = backend.model.generate.side_effect

        def fail_second_sentence(**kwargs):
            if backend.latest["text"] == ["Second!"]:
                raise RuntimeError("model failure")
            return generate(**kwargs)

        backend.model.generate.side_effect = fail_second_sentence
        output = self.root / "partial.wav"
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            infer.synthesize_speech("First. Second!", output=output, sentence_iteration=True)
        self.assertFalse(output.exists())
        self.assertEqual(
            [call.kwargs["text"] for call in backend.processor.call_args_list],
            [["First."], ["Second!"]],
        )

    def test_wrapper_rejects_unknown_options_and_unpaired_reference(self):
        with self.assertRaisesRegex(TypeError, "unknown inference option"):
            infer.synthesize_speech("Hello.", misspelled_option=42)
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            infer.synthesize_speech("Hello.", reference_text="Transcript.")


if __name__ == "__main__":
    unittest.main()
