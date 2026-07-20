#!/usr/bin/env python3
"""Diagnose Daniel OS SFT balance, leakage risk, and loss behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


BEHAVIORS = ("answer", "ground_external", "retrieve", "unknown", "refuse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training",
        nargs="+",
        type=Path,
        default=[
            Path("assets/data/daniel-lfm2-sft.jsonl"),
            Path("assets/data/daniel-lfm2-routing-sft.jsonl"),
        ],
    )
    parser.add_argument(
        "--evaluation",
        nargs="+",
        type=Path,
        default=[
            Path("assets/data/daniel-lfm2-eval.jsonl"),
            Path("assets/data/daniel-lfm2-routing-eval.jsonl"),
            Path("assets/data/daniel-lfm2-test.jsonl"),
        ],
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("assets/data/daniel-lfm2-training-metrics.json"),
    )
    parser.add_argument("--legacy-balance-floor", type=int, default=64)
    parser.add_argument("--legacy-holdout-per-behavior", type=int, default=2)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.72)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/daniel-lfm2-data-diagnostics.json")
    )
    return parser.parse_args()


def read_jsonl(paths: list[Path]) -> list[dict]:
    return [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def final_prompt(record: dict) -> str:
    messages = record.get("messages") or []
    if messages:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message["content"]
    return record.get("prompt", "")


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9가-힣]+", text.lower()))


def token_set(text: str) -> set[str]:
    return set(normalize(text).split())


def jaccard(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:12]


def behavior_counts(records: list[dict]) -> Counter:
    return Counter(record["behavior"] for record in records)


def language_counts(records: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        counts[record["behavior"]][record.get("language", "en")] += 1
    return {behavior: dict(counts[behavior]) for behavior in BEHAVIORS}


def length_summary(records: list[dict]) -> dict[str, dict[str, float]]:
    by_behavior: dict[str, list[int]] = defaultdict(list)
    for record in records:
        answer = (record.get("messages") or [{"content": ""}])[-1]["content"]
        by_behavior[record["behavior"]].append(len(answer.split()))
    return {
        behavior: {
            "minimum_words": min(lengths),
            "median_words": statistics.median(lengths),
            "mean_words": round(statistics.mean(lengths), 2),
            "maximum_words": max(lengths),
        }
        for behavior, lengths in by_behavior.items()
    }


def legacy_sampling(
    counts: Counter, balance_floor: int, holdout_per_behavior: int
) -> dict:
    details = {}
    source_total = sum(counts.values())
    training_total = 0
    holdout_total = 0
    unique_training_total = 0
    for behavior in BEHAVIORS:
        source = counts[behavior]
        holdout = min(holdout_per_behavior, source)
        unique_train = source - holdout
        effective_train = max(balance_floor, unique_train)
        training_total += effective_train
        holdout_total += holdout
        unique_training_total += unique_train
        details[behavior] = {
            "source": source,
            "loss_holdout": holdout,
            "unique_train": unique_train,
            "effective_train_per_epoch": effective_train,
            "repeat_factor": round(effective_train / unique_train, 2),
        }
    return {
        "source_total": source_total,
        "loss_holdout_total": holdout_total,
        "unique_training_total": unique_training_total,
        "effective_training_total_per_epoch": training_total,
        "repeated_slots_per_epoch": training_total - unique_training_total,
        "repeated_slot_fraction": round(
            (training_total - unique_training_total) / training_total, 4
        ),
        "by_behavior": details,
    }


def prompt_overlap(training: list[dict], evaluation: list[dict], threshold: float) -> dict:
    train_prompts = [(record["id"], final_prompt(record)) for record in training]
    exact = {normalize(prompt): record_id for record_id, prompt in train_prompts}
    exact_matches = []
    near_matches = []
    for record in evaluation:
        prompt = final_prompt(record)
        normalized = normalize(prompt)
        if normalized in exact:
            exact_matches.append(
                {"evaluation_id": record["id"], "training_id": exact[normalized]}
            )
            continue
        best_id = None
        best_score = 0.0
        for train_id, train_prompt in train_prompts:
            score = jaccard(prompt, train_prompt)
            if score > best_score:
                best_score = score
                best_id = train_id
        if best_score >= threshold:
            near_matches.append(
                {
                    "evaluation_id": record["id"],
                    "training_id": best_id,
                    "token_jaccard": round(best_score, 4),
                    "evaluation_prompt_fingerprint": fingerprint(prompt),
                }
            )
    return {
        "exact_prompt_matches": exact_matches,
        "near_prompt_matches": sorted(
            near_matches, key=lambda item: item["token_jaccard"], reverse=True
        ),
        "threshold": threshold,
    }


def loss_diagnostics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {"available": False}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    train = metrics.get("train", [])
    validation = [point for point in metrics.get("validation", []) if point.get("loss")]
    result = {
        "available": bool(validation),
        "train_points": len(train),
        "validation_points": len(validation),
        "validation": validation,
    }
    if not validation:
        return result
    best_index = min(range(len(validation)), key=lambda index: validation[index]["loss"])
    best = validation[best_index]
    result.update(
        {
            "best_epoch": best.get("epoch"),
            "best_loss": best["loss"],
            "last_loss": validation[-1]["loss"],
            "post_best_relative_increase": round(
                (validation[-1]["loss"] - best["loss"]) / best["loss"], 4
            ),
            "interpretation": (
                "Validation improved before rising, which is consistent with overfitting, "
                "but the legacy ten-example loss holdout is too small to separate model "
                "movement from sampling variance."
                if best_index < len(validation) - 1
                else "Validation did not rise after the selected checkpoint."
            ),
        }
    )
    if train:
        early = [point["loss"] for point in train[: max(1, len(train) // 10)]]
        late = [point["loss"] for point in train[-max(1, len(train) // 10) :]]
        result["early_train_loss_mean"] = round(statistics.mean(early), 4)
        result["late_train_loss_mean"] = round(statistics.mean(late), 4)
        result["train_loss_reduction"] = round(
            result["early_train_loss_mean"] - result["late_train_loss_mean"], 4
        )
    return result


def recommendations(legacy: dict, loss: dict, overlap: dict) -> list[str]:
    items = [
        "Replace cyclic minority oversampling with genuinely distinct, grounded prompts.",
        "Keep every paraphrase from one seed in the same split so template families cannot leak.",
        "Use a fixed validation set with enough examples per behavior and report per-behavior loss.",
        "Evaluate every 25-50 optimizer steps and stop after two non-improving evaluations.",
        "Sweep learning rates near 5e-5, 1e-4, and 2e-4 instead of interpreting one run.",
        "Select checkpoints by strict behavior gates first and macro validation loss second.",
    ]
    if legacy["repeated_slot_fraction"] > 0.2:
        items.append(
            f"The legacy stream repeats {legacy['repeated_slot_fraction']:.1%} of its slots; "
            "generate new coverage before another full run."
        )
    if loss.get("post_best_relative_increase", 0) > 0:
        items.append(
            f"The final validation loss is {loss['post_best_relative_increase']:.1%} above the "
            "best checkpoint; retain early stopping and test lower learning rates."
        )
    if overlap["near_prompt_matches"]:
        items.append(
            "Review near-duplicate evaluation prompts and group related scenarios before splitting."
        )
    return items


def main() -> None:
    args = parse_args()
    training = read_jsonl(args.training)
    evaluation = read_jsonl(args.evaluation)
    counts = behavior_counts(training)
    legacy = legacy_sampling(
        counts, args.legacy_balance_floor, args.legacy_holdout_per_behavior
    )
    overlap = prompt_overlap(training, evaluation, args.near_duplicate_threshold)
    loss = loss_diagnostics(args.metrics)
    payload = {
        "training_records": len(training),
        "evaluation_records": len(evaluation),
        "behavior_counts": dict(counts),
        "language_counts": language_counts(training),
        "answer_lengths": length_summary(training),
        "legacy_sampling": legacy,
        "prompt_overlap": overlap,
        "loss": loss,
        "recommendations": recommendations(legacy, loss, overlap),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
