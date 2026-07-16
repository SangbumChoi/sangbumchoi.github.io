#!/usr/bin/env python3
"""Merge a trained Daniel OS adapter without loading the training stack."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2-350M")
    parser.add_argument("--adapter", default="artifacts/daniel-lfm2-350m/adapter")
    parser.add_argument("--output", default="artifacts/daniel-lfm2-350m/merged")
    args = parser.parse_args()

    dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    output = Path(args.output)
    merged.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(output)
    print(f"Merged checkpoint written to {output}")


if __name__ == "__main__":
    main()
