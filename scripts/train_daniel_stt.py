#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "accelerate>=1.10,<2",
#   "datasets>=3,<5",
#   "huggingface-hub>=0.34,<2",
#   "numpy>=1.26,<3",
#   "peft>=0.17,<1",
#   "scipy>=1.14,<2",
#   "soundfile>=0.13,<1",
#   "torch>=2.6",
#   "transformers>=4.55,<5",
# ]
# ///
"""Fine-tune a browser-sized Whisper model on speaker-disjoint English speech."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from datasets import Dataset
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, get_peft_model
from scipy.signal import resample_poly
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-dir", type=Path)
    source.add_argument("--dataset-repo")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--model", default="openai/whisper-tiny.en")
    parser.add_argument("--output", type=Path, default=Path("artifacts/daniel-stt"))
    parser.add_argument("--epochs", type=float, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-test-wer", type=float, default=0.35)
    parser.add_argument("--max-worst-group-wer", type=float, default=0.65)
    parser.add_argument("--min-keyword-recall", type=float, default=0.75)
    parser.add_argument("--keywords", type=Path)
    parser.add_argument("--hub-repo", default="danelcsb/daniel-stt-tiny-en")
    parser.add_argument("--training-revision", default="local")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--public-model", action="store_true")
    parser.add_argument("--disable-augmentation", action="store_true")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9가-힣']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_counts(reference: str, prediction: str) -> dict[str, int]:
    ref = normalize_text(reference).split()
    hyp = normalize_text(prediction).split()
    previous = [(index, 0, 0, index) for index in range(len(hyp) + 1)]
    for ref_index, ref_word in enumerate(ref, 1):
        current = [(ref_index, 0, ref_index, 0)]
        for hyp_index, hyp_word in enumerate(hyp, 1):
            if ref_word == hyp_word:
                current.append(previous[hyp_index - 1])
                continue
            diagonal = previous[hyp_index - 1]
            deletion = previous[hyp_index]
            insertion = current[hyp_index - 1]
            current.append(
                min(
                    (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                    (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                    (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
                )
            )
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    return {
        "reference_words": len(ref),
        "edits": distance,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def aggregate(items: list[dict]) -> dict[str, float | int]:
    totals = {
        key: sum(item[key] for item in items)
        for key in ("reference_words", "edits", "substitutions", "deletions", "insertions")
    }
    totals["wer"] = totals["edits"] / totals["reference_words"] if totals["reference_words"] else 0.0
    return totals


def read_jsonl(path: Path, split: str) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"{path}: no records")
    for record in records:
        record["split"] = split
    return records


def resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.dataset_repo:
        return Path(
            snapshot_download(
                repo_id=args.dataset_repo,
                repo_type="dataset",
                revision=args.dataset_revision,
                token=os.environ.get("HF_TOKEN"),
            )
        )
    return (args.data_dir or Path("artifacts/daniel-stt-data")).resolve()


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim != 1:
        raise ValueError(f"{path}: expected mono audio")
    if sample_rate != 16_000:
        divisor = math.gcd(sample_rate, 16_000)
        audio = resample_poly(audio, 16_000 // divisor, sample_rate // divisor).astype(np.float32)
    if len(audio) > 30 * 16_000:
        audio = audio[: 30 * 16_000]
    return audio


def augment_audio(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    augmented = audio.copy()
    if rng.random() < 0.55:
        augmented *= 10 ** (rng.uniform(-4.0, 4.0) / 20.0)
    if rng.random() < 0.45:
        speed = rng.choice((0.94, 0.97, 1.03, 1.06))
        target_length = max(1, round(len(augmented) / speed))
        augmented = np.interp(
            np.linspace(0, len(augmented) - 1, target_length),
            np.arange(len(augmented)),
            augmented,
        ).astype(np.float32)
    if rng.random() < 0.45 and np.any(augmented):
        signal_rms = float(np.sqrt(np.mean(np.square(augmented))))
        snr_db = rng.uniform(14.0, 30.0)
        noise_rms = signal_rms / (10 ** (snr_db / 20.0))
        noise = np.random.default_rng(rng.randrange(2**32)).normal(
            0.0, noise_rms, len(augmented)
        )
        augmented = augmented + noise.astype(np.float32)
    return np.clip(augmented, -1.0, 1.0)


@dataclass
class SpeechCollator:
    processor: Any
    data_dir: Path
    seed: int
    augment: bool = True

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        waveforms = []
        transcripts = []
        for feature in features:
            audio_path = Path(feature["audio_path"])
            if not audio_path.is_absolute():
                audio_path = self.data_dir / audio_path
            audio = load_audio(audio_path)
            if self.augment and feature.get("split") == "train":
                audio = augment_audio(audio, self.rng)
            waveforms.append(audio)
            transcripts.append(feature["transcript"])
        inputs = self.processor.feature_extractor(
            waveforms, sampling_rate=16_000, return_tensors="pt"
        )
        labels = self.processor.tokenizer(
            transcripts, padding=True, return_tensors="pt"
        )
        label_ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        if (
            label_ids.shape[1] > 0
            and (label_ids[:, 0] == self.processor.tokenizer.bos_token_id).all().item()
        ):
            label_ids = label_ids[:, 1:]
        return {"input_features": inputs.input_features, "labels": label_ids}


def metric_function(processor: Any):
    def compute_metrics(prediction: Any) -> dict[str, float]:
        predicted_ids = prediction.predictions
        if isinstance(predicted_ids, tuple):
            predicted_ids = predicted_ids[0]
        label_ids = prediction.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predictions = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        references = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        totals = aggregate([edit_counts(ref, hyp) for ref, hyp in zip(references, predictions)])
        return {"wer": float(totals["wer"])}

    return compute_metrics


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def strict_evaluate(
    model: Any,
    processor: Any,
    records: list[dict],
    data_dir: Path,
    keywords: list[str],
    output_path: Path,
) -> dict:
    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    predictions = []
    for record in records:
        audio_path = Path(record["audio_path"])
        if not audio_path.is_absolute():
            audio_path = data_dir / audio_path
        audio = load_audio(audio_path)
        inputs = processor.feature_extractor(audio, sampling_rate=16_000, return_tensors="pt")
        input_features = inputs.input_features.to(device=device, dtype=model_dtype)
        synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(input_features=input_features, max_new_tokens=128)
        synchronize()
        elapsed = time.perf_counter() - started
        prediction = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        predictions.append({**record, "prediction": prediction, "inference_seconds": elapsed})

    scored = []
    groups: dict[str, dict[str, list[dict]]] = {
        key: defaultdict(list) for key in ("speaker_id", "domain", "environment", "accent_group")
    }
    keyword_hits = 0
    keyword_opportunities = 0
    for record in predictions:
        counts = edit_counts(record["transcript"], record["prediction"])
        scored.append(counts)
        for key, values in groups.items():
            values[str(record.get(key, "unspecified"))].append(counts)
        reference = normalize_text(record["transcript"])
        prediction = normalize_text(record["prediction"])
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", reference):
                keyword_opportunities += 1
                keyword_hits += int(bool(re.search(rf"\b{re.escape(keyword)}\b", prediction)))

    grouped_metrics = {
        key: {name: aggregate(items) for name, items in sorted(values.items())}
        for key, values in groups.items()
    }
    speaker_wers = [value["wer"] for value in grouped_metrics["speaker_id"].values()]
    comparison_wers = [
        value["wer"]
        for key in ("domain", "environment", "accent_group")
        for value in grouped_metrics[key].values()
    ]
    metrics = {
        "overall": aggregate(scored),
        "macro_speaker_wer": statistics.fmean(speaker_wers),
        "worst_speaker_wer": max(speaker_wers),
        "worst_group_wer": max(comparison_wers) if comparison_wers else 0.0,
        "keyword_recall": keyword_hits / keyword_opportunities if keyword_opportunities else None,
        "keyword_hits": keyword_hits,
        "keyword_opportunities": keyword_opportunities,
        "median_inference_seconds": statistics.median(
            record["inference_seconds"] for record in predictions
        ),
        "groups": grouped_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in predictions:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return metrics


def write_model_card(path: Path, args: argparse.Namespace, metrics: dict) -> None:
    keyword_recall = metrics["keyword_recall"]
    keyword_text = "not measured" if keyword_recall is None else f"{keyword_recall:.3f}"
    path.write_text(
        f"""---
language: en
license: mit
base_model: {args.model}
pipeline_tag: automatic-speech-recognition
library_name: transformers
---

# Daniel OS browser STT

LoRA adaptation of `{args.model}` for English browser speech recognition. The
training data is speaker-disjoint and remains private by default. This model is
not trained to recognize only Daniel's voice; it targets varied speakers,
devices, accents, environments, and portfolio-specific names.

## Held-out test

- WER: {metrics['overall']['wer']:.3f}
- Macro speaker WER: {metrics['macro_speaker_wer']:.3f}
- Worst non-speaker group WER: {metrics['worst_group_wer']:.3f}
- Portfolio keyword recall: {keyword_text}
- Training revision: `{args.training_revision}`
- Dataset revision: `{args.dataset_revision}`

The browser deployment still requires an ONNX export and a separate browser
latency, memory, and regression gate.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = resolve_data_dir(args)
    records = {split: read_jsonl(data_dir / f"{split}.jsonl", split) for split in SPLITS}

    processor = AutoProcessor.from_pretrained(args.model)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.config.apply_spec_augment = not args.disable_augmentation
    model.config.mask_time_prob = 0.05
    model.config.mask_feature_prob = 0.05
    model.config.use_cache = False
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    # The generic PEFT wrapper preserves Whisper's input_features signature.
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    args.output.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        predict_with_generate=True,
        generation_max_length=128,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(records["train"]),
        eval_dataset=Dataset.from_list(records["validation"]),
        data_collator=SpeechCollator(
            processor, data_dir, args.seed, augment=not args.disable_augmentation
        ),
        compute_metrics=metric_function(processor),
        processing_class=processor,
    )
    train_result = trainer.train()
    validation_metrics = trainer.evaluate()
    (args.output / "training_metrics.json").write_text(
        json.dumps(
            {
                "training_revision": args.training_revision,
                "dataset_revision": args.dataset_revision,
                "train": train_result.metrics,
                "validation": validation_metrics,
                "history": trainer.state.log_history,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    adapter_dir = args.output / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    merged = trainer.model.merge_and_unload()
    merged.config.use_cache = True
    merged_dir = args.output / "merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    processor.save_pretrained(merged_dir)

    keyword_candidates = [
        args.keywords,
        data_dir / "keywords.txt",
        Path(__file__).resolve().parents[1] / "assets/data/daniel-stt-keywords.txt",
    ]
    keyword_path = next(
        (path for path in keyword_candidates if path is not None and path.is_file()), None
    )
    if keyword_path is None:
        raise FileNotFoundError("Could not locate the STT keyword list")
    keywords = [
        normalize_text(line)
        for line in keyword_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merged.to(device)
    test_metrics = strict_evaluate(
        merged,
        processor,
        records["test"],
        data_dir,
        keywords,
        args.output / "test_predictions.jsonl",
    )
    (merged_dir / "evaluation.json").write_text(
        json.dumps(test_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_model_card(merged_dir / "README.md", args, test_metrics)

    failures = []
    if test_metrics["overall"]["wer"] > args.max_test_wer:
        failures.append(
            f"test WER {test_metrics['overall']['wer']:.3f} > {args.max_test_wer:.3f}"
        )
    if test_metrics["worst_group_wer"] > args.max_worst_group_wer:
        failures.append(
            f"worst group WER {test_metrics['worst_group_wer']:.3f} > "
            f"{args.max_worst_group_wer:.3f}"
        )
    if (
        test_metrics["keyword_recall"] is not None
        and test_metrics["keyword_recall"] < args.min_keyword_recall
    ):
        failures.append(
            f"keyword recall {test_metrics['keyword_recall']:.3f} < "
            f"{args.min_keyword_recall:.3f}"
        )
    if failures:
        raise RuntimeError("; ".join(failures))

    if args.push_to_hub:
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN is required to publish the model")
        api = HfApi()
        api.create_repo(
            args.hub_repo,
            repo_type="model",
            private=not args.public_model,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=args.hub_repo,
            repo_type="model",
            folder_path=merged_dir,
            commit_message=f"Train speaker-disjoint browser STT at {args.training_revision}",
        )
    print(json.dumps(test_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
