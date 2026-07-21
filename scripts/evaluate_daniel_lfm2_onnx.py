#!/usr/bin/env python3
"""Replay a full-precision evaluation manifest against a quantized LFM2 ONNX graph.

This script runs inside the pinned LiquidONNX project environment. It deliberately
consumes the messages and generation settings emitted by
``evaluate_daniel_lfm2_test.py`` so quantization is the only intended variable.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from liquidonnx.session import initialize_cache, load_onnx_session, update_cache
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tokenizer_root(model_path: Path) -> Path:
    return model_path.parent.parent if model_path.suffix == ".onnx" else model_path


def token_list(value) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    while value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def eos_token_ids(tokenizer, root: Path) -> set[int]:
    values = tokenizer.eos_token_id
    config_path = root / "generation_config.json"
    if config_path.exists():
        values = json.loads(config_path.read_text(encoding="utf-8")).get(
            "eos_token_id", values
        )
    if values is None:
        return set()
    if not isinstance(values, list):
        values = [values]
    return {int(value) for value in values}


def penalize_repeated_tokens(
    logits: np.ndarray, token_ids: list[int], penalty: float
) -> np.ndarray:
    if penalty == 1.0:
        return logits
    adjusted = logits.copy()
    for token_id in set(token_ids):
        if token_id < 0 or token_id >= adjusted.shape[-1]:
            continue
        score = adjusted[token_id]
        adjusted[token_id] = score * penalty if score < 0 else score / penalty
    return adjusted


def generate(
    session,
    tokenizer,
    messages: list[dict],
    generation: dict,
    stop_ids: set[int],
) -> tuple[str, int, int, float]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=int(generation["max_input_tokens"]),
        return_dict=False,
    )
    input_ids = token_list(encoded)
    history = input_ids.copy()
    generated_ids: list[int] = []
    input_names = {item.name for item in session.get_inputs()}
    output_infos = session.get_outputs()
    cache = initialize_cache(session)

    started_at = time.perf_counter()
    for step in range(int(generation["max_new_tokens"])):
        current_length = len(history)
        if step == 0:
            ids = np.asarray([input_ids], dtype=np.int64)
            positions = np.arange(len(input_ids), dtype=np.int64).reshape(1, -1)
        else:
            ids = np.asarray([[generated_ids[-1]]], dtype=np.int64)
            positions = np.asarray([[current_length - 1]], dtype=np.int64)

        feed = {
            "input_ids": ids,
            "attention_mask": np.ones((1, current_length), dtype=np.int64),
        }
        if "position_ids" in input_names:
            feed["position_ids"] = positions
        feed.update(cache)

        outputs = session.run(None, feed)
        logits = outputs[0][0, -1]
        logits = penalize_repeated_tokens(
            logits,
            history,
            float(generation.get("repetition_penalty", 1.0)),
        )
        next_token = int(np.argmax(logits))
        generated_ids.append(next_token)
        history.append(next_token)
        update_cache(cache, outputs, output_infos)
        if next_token in stop_ids:
            break

    elapsed_seconds = time.perf_counter() - started_at
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer, len(input_ids), len(generated_ids), elapsed_seconds


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    generation = baseline.get("generation")
    if not generation:
        raise ValueError("Baseline report is missing deterministic generation settings.")
    if generation.get("do_sample") is not False:
        raise ValueError("Quantization parity requires greedy decoding (do_sample=false).")

    missing_messages = [item["id"] for item in baseline["results"] if not item.get("messages")]
    if missing_messages:
        raise ValueError(
            "Baseline report does not contain replayable messages for: "
            + ", ".join(missing_messages)
        )

    root = tokenizer_root(args.model)
    tokenizer = AutoTokenizer.from_pretrained(str(root), trust_remote_code=True)
    session = load_onnx_session(args.model, providers=["CPUExecutionProvider"])
    stop_ids = eos_token_ids(tokenizer, root)
    results = []

    for case in baseline["results"]:
        answer, input_count, output_count, elapsed_seconds = generate(
            session,
            tokenizer,
            case["messages"],
            generation,
            stop_ids,
        )
        result = {
            "id": case["id"],
            "prompt": case["prompt"],
            "messages_sha256": case["messages_sha256"],
            "answer": answer,
            "input_token_count": input_count,
            "generated_token_count": output_count,
            "generation_seconds": elapsed_seconds,
            "tokens_per_second": output_count / elapsed_seconds if elapsed_seconds else 0.0,
        }
        results.append(result)
        print(
            f"[{case['id']}] {elapsed_seconds:.3f}s, "
            f"{result['tokens_per_second']:.2f} token/s | {answer}",
            flush=True,
        )

    report = {
        "model": str(args.model),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": "onnxruntime-cpu",
        "precision": "q4-symmetric",
        "baseline_model": baseline["model"],
        "generation": generation,
        "case_count": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Quantized evaluation written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
