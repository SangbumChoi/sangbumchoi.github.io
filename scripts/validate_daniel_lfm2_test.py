#!/usr/bin/env python3
"""Validate the public strict test split and claim provenance."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


BEHAVIORS = {"answer", "unknown", "refuse"}
LANGUAGES = {"en", "ko"}
SOURCE_STATUSES = {"externally_verified", "public_self_report", "not_verified"}


def validate_messages(messages: list[dict], record_id: str) -> None:
    roles = [message.get("role") for message in messages]
    expected = ["user" if index % 2 == 0 else "assistant" for index in range(len(messages))]
    if not messages or roles != expected or roles[-1] != "user":
        raise ValueError(
            f"{record_id}: messages must alternate user/assistant and end with user"
        )
    if any(not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages):
        raise ValueError(f"{record_id}: every message needs non-empty content")


def final_prompt(record: dict) -> str:
    messages = record.get("messages")
    return messages[-1]["content"] if messages else record.get("prompt", "")


def read_jsonl(path: Path) -> list[dict]:
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
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


def validate_sources(path: Path) -> Counter:
    data = json.loads(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    counts: Counter = Counter()
    for claim in data.get("claims", []):
        claim_id = claim.get("id")
        if not claim_id or claim_id in ids:
            raise ValueError(f"Missing or duplicate provenance id: {claim_id}")
        ids.add(claim_id)
        status = claim.get("status")
        if status not in SOURCE_STATUSES:
            raise ValueError(f"{claim_id}: invalid provenance status {status}")
        counts[status] += 1
        sources = claim.get("sources")
        if not isinstance(sources, list):
            raise ValueError(f"{claim_id}: sources must be a list")
        if status == "externally_verified" and not sources:
            raise ValueError(f"{claim_id}: externally verified claims require a source")
        if status == "not_verified" and sources:
            raise ValueError(f"{claim_id}: unverified claims cannot cite a confirming source")
        if any(not source.startswith("https://") for source in sources):
            raise ValueError(f"{claim_id}: only HTTPS provenance URLs are allowed")
    if not counts["not_verified"]:
        raise ValueError("At least one explicit not-verified claim is required")
    return counts


def validate_test(
    records: list[dict],
    profile: dict,
    excluded_prompts: set[str],
) -> tuple[Counter, Counter]:
    ids: set[str] = set()
    prompts: set[str] = set()
    behavior_counts: Counter = Counter()
    language_counts: Counter = Counter()
    for record in records:
        record_id = record.get("id")
        if not record_id or record_id in ids:
            raise ValueError(f"Missing or duplicate test id: {record_id}")
        ids.add(record_id)
        behavior = record.get("behavior")
        language = record.get("language")
        if behavior not in BEHAVIORS:
            raise ValueError(f"{record_id}: invalid behavior {behavior}")
        if language not in LANGUAGES:
            raise ValueError(f"{record_id}: invalid language {language}")
        behavior_counts[behavior] += 1
        language_counts[language] += 1
        context_keys = record.get("context_keys", [])
        if not context_keys or any(key not in profile for key in context_keys):
            raise ValueError(f"{record_id}: invalid context keys {context_keys}")
        messages = record.get("messages")
        if messages:
            validate_messages(messages, record_id)
        prompt = final_prompt(record).strip()
        normalized_prompt = prompt.lower()
        if not prompt or normalized_prompt in prompts or normalized_prompt in excluded_prompts:
            raise ValueError(f"{record_id}: prompt is empty, duplicated, or leaked from train/validation")
        prompts.add(normalized_prompt)
        if not record.get("difficulty"):
            raise ValueError(f"{record_id}: difficulty is required")
        groups = record.get("expected_groups")
        if not groups or any(not group or not all(isinstance(term, str) and term for term in group) for group in groups):
            raise ValueError(f"{record_id}: expected_groups must contain non-empty string groups")
        forbidden = record.get("forbidden_terms", [])
        if not isinstance(forbidden, list) or not all(isinstance(term, str) for term in forbidden):
            raise ValueError(f"{record_id}: forbidden_terms must be a string list")
        source_urls = record.get("source_urls")
        if not isinstance(source_urls, list) or any(not url.startswith("https://") for url in source_urls):
            raise ValueError(f"{record_id}: source_urls must contain only HTTPS URLs")
        if behavior == "answer" and not source_urls:
            raise ValueError(f"{record_id}: grounded answers require at least one public source")
        if behavior == "answer":
            context = flatten({key: profile[key] for key in context_keys})
            for group in groups:
                if not any(term.lower() in context for term in group):
                    raise ValueError(f"{record_id}: expected group is absent from context: {group}")

    minimums = {"answer": 20, "unknown": 6, "refuse": 8}
    for behavior, minimum in minimums.items():
        if behavior_counts[behavior] < minimum:
            raise ValueError(f"Strict test needs at least {minimum} {behavior} cases")
    if language_counts["ko"] < 8:
        raise ValueError("Strict test needs at least eight Korean cases")
    return behavior_counts, language_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("assets/data/daniel-lfm2-sft.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("assets/data/daniel-lfm2-eval.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("assets/data/daniel-lfm2-test.jsonl"))
    parser.add_argument("--profile", type=Path, default=Path("assets/data/daniel-profile.json"))
    parser.add_argument(
        "--sources", type=Path, default=Path("assets/data/daniel-profile-sources.json")
    )
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    train = read_jsonl(args.train)
    validation = read_jsonl(args.validation)
    excluded_prompts = {
        message["content"].strip().lower()
        for record in train
        for message in record["messages"]
        if message["role"] == "user"
    } | {
        final_prompt(record).strip().lower() for record in validation
    }
    behaviors, languages = validate_test(read_jsonl(args.test), profile, excluded_prompts)
    summary = {
        "provenance_statuses": dict(validate_sources(args.sources)),
        "test_behaviors": dict(behaviors),
        "test_languages": dict(languages),
        "test_records": sum(behaviors.values()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
