#!/usr/bin/env python3
"""Run a small identity and factuality smoke test against a merged checkpoint."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "Who are you?",
    "What does Daniel do at Toss Bank?",
    "What are Daniel's strongest open-source contributions?",
    "Why does Daniel contribute to open source?",
    "What kind of engineering problems does Daniel enjoy?",
]

CONTEXT = """Sangbum Daniel Choi is an AI research and systems engineer and a Data Scientist at Toss Bank. He works on an on-premise LLM agent, face and ID-card authentication, and an end-to-end document extraction pipeline using an approximately 1B-parameter VLM. He has contributed more than 40 pull requests to Hugging Face Transformers, including SAM2, Molmo2, RT-DETR, ViTPose, distributed training fixes, tests, and documentation. He believes open source expands the shared technical foundation."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()

    for prompt in PROMPTS:
        messages = [
            {"role": "system", "content": f"You are Daniel OS, Sangbum Daniel Choi's portfolio assistant. Never claim to be Daniel. Answer only from the verified facts and say when information is missing.\nVerified facts:\n{CONTEXT}"},
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"\nQ: {prompt}\nA: {answer.strip()}")


if __name__ == "__main__":
    main()
