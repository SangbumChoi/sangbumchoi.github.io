#!/usr/bin/env python3
"""Fine-tune and merge a small Daniel OS LoRA adapter for LFM2-350M."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2-350M")
    parser.add_argument("--dataset", default="assets/data/daniel-lfm2-sft.jsonl")
    parser.add_argument("--output", default="artifacts/daniel-lfm2-350m")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def load_records(path: Path, seed: int) -> tuple[Dataset, Dataset]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) < 10:
        raise ValueError("At least 10 verified conversations are required.")
    for record in records:
        answer = record["messages"][-1]["content"]
        record["messages"][0]["content"] = (
            "You are Daniel OS, the browser-native portfolio assistant of Sangbum Daniel Choi. "
            "Never claim to be Daniel. Answer only from the verified facts and say when information is missing.\n"
            f"Verified facts for this question:\n{answer}"
        )
    random.Random(seed).shuffle(records)
    dataset = Dataset.from_list(records)
    return dataset, dataset.select(range(min(2, len(dataset))))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    adapter_dir = output / "adapter"
    merged_dir = output / "merged"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
    if not args.merge_only:
        train_data, eval_data = load_records(Path(args.dataset), args.seed)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        model.config.use_cache = False
        config = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        max_length=512,
        bf16=False,
        fp16=False,
        report_to="none",
        seed=args.seed,
        )
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=train_data,
            eval_dataset=eval_data,
            processing_class=tokenizer,
            peft_config=lora,
        )
        trainer.train()
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(adapter_dir)

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged checkpoint written to {merged_dir}")


if __name__ == "__main__":
    main()
