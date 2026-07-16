#!/usr/bin/env python3
"""Validate Daniel OS training and evaluation data before fine-tuning."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


BEHAVIORS = {"answer", "unknown", "refuse"}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def flatten(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def validate_training(records: list[dict], profile: dict) -> Counter:
    ids: set[str] = set()
    prompts: set[str] = set()
    counts: Counter = Counter()
    for record in records:
        record_id = record.get("id")
        if not record_id or record_id in ids:
            raise ValueError(f"Missing or duplicate training id: {record_id}")
        ids.add(record_id)
        behavior = record.get("behavior")
        if behavior not in BEHAVIORS:
            raise ValueError(f"{record_id}: invalid behavior {behavior}")
        counts[behavior] += 1
        context_keys = record.get("context_keys")
        if not context_keys or any(key not in profile for key in context_keys):
            raise ValueError(f"{record_id}: invalid context keys {context_keys}")
        messages = record.get("messages", [])
        if [message.get("role") for message in messages] != ["user", "assistant"]:
            raise ValueError(f"{record_id}: messages must be one user/assistant pair")
        prompt = messages[0]["content"].strip().lower()
        if prompt in prompts:
            raise ValueError(f"{record_id}: duplicate prompt")
        prompts.add(prompt)
        answer = messages[1]["content"].strip()
        if len(answer.split()) > 110:
            raise ValueError(f"{record_id}: answer is too long")
        for term in record.get("expected_terms", []):
            if term.lower() not in answer.lower():
                raise ValueError(f"{record_id}: expected term missing from answer: {term}")
        if behavior == "unknown" and "does not contain" not in answer.lower():
            raise ValueError(f"{record_id}: unknown answer must state that information is absent")
        if behavior == "refuse" and not any(
            marker in answer.lower() for marker in ("outside this portfolio's scope", "cannot pretend")
        ):
            raise ValueError(f"{record_id}: refusal boundary is missing")
        if behavior == "answer":
            context = flatten({key: profile[key] for key in context_keys})
            for number in re.findall(r"\b\d[\d.,]*[A-Za-z+]*\b", answer):
                if number.lower().rstrip(".,") not in context.lower():
                    raise ValueError(f"{record_id}: numeric claim is absent from context: {number}")
    if counts["answer"] < 30 or counts["unknown"] < 6 or counts["refuse"] < 12:
        raise ValueError(f"Insufficient behavior coverage: {dict(counts)}")
    return counts


def validate_eval(records: list[dict], profile: dict) -> Counter:
    ids: set[str] = set()
    counts: Counter = Counter()
    for record in records:
        record_id = record.get("id")
        if not record_id or record_id in ids:
            raise ValueError(f"Missing or duplicate evaluation id: {record_id}")
        ids.add(record_id)
        behavior = record.get("behavior")
        if behavior not in BEHAVIORS:
            raise ValueError(f"{record_id}: invalid behavior {behavior}")
        counts[behavior] += 1
        if any(key not in profile for key in record.get("context_keys", [])):
            raise ValueError(f"{record_id}: invalid context key")
        if not record.get("prompt") or not record.get("expected_groups"):
            raise ValueError(f"{record_id}: prompt and expected groups are required")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("assets/data/daniel-lfm2-sft.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("assets/data/daniel-lfm2-eval.jsonl"))
    parser.add_argument("--profile", type=Path, default=Path("assets/data/daniel-profile.json"))
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    training = load_jsonl(args.dataset)
    evaluation = load_jsonl(args.eval)
    summary = {
        "training_records": len(training),
        "training_behaviors": dict(validate_training(training, profile)),
        "evaluation_records": len(evaluation),
        "evaluation_behaviors": dict(validate_eval(evaluation, profile)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
