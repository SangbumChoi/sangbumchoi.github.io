#!/usr/bin/env python3
"""Compare source-checkpoint and Q4 answers, then enforce the publish gate."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-strict-score-drop", type=float, default=0.0)
    parser.add_argument("--maximum-new-strict-regressions", type=int, default=0)
    parser.add_argument("--maximum-new-forbidden-leaks", type=int, default=0)
    parser.add_argument("--minimum-key-fact-retention", type=float, default=1.0)
    parser.add_argument("--minimum-mean-answer-similarity", type=float, default=0.70)
    parser.add_argument("--minimum-throughput-ratio", type=float, default=0.75)
    return parser.parse_args()


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def answer_score(case: dict, answer: str) -> dict:
    answer_lower = answer.casefold()
    matched_groups = [
        any(term.casefold() in answer_lower for term in group)
        for group in case["expected_groups"]
    ]
    forbidden_matches = [
        term for term in case.get("forbidden_terms", []) if term.casefold() in answer_lower
    ]
    expected_pass = all(matched_groups)
    forbidden_pass = not forbidden_matches
    language_pass = case["language"] != "ko" or bool(re.search(r"[가-힣]", answer))
    behavior_pass = expected_pass and forbidden_pass
    return {
        "matched_groups": matched_groups,
        "forbidden_matches": forbidden_matches,
        "expected_pass": expected_pass,
        "forbidden_pass": forbidden_pass,
        "language_pass": language_pass,
        "behavior_pass": behavior_pass,
        "strict_pass": behavior_pass and language_pass,
    }


def indexed(results: list[dict], name: str) -> dict[str, dict]:
    output = {item["id"]: item for item in results}
    if len(output) != len(results):
        raise ValueError(f"{name} contains duplicate case IDs.")
    return output


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def build_report(baseline: dict, quantized: dict, thresholds: dict) -> dict:
    baseline_cases = indexed(baseline["results"], "baseline")
    quantized_cases = indexed(quantized["results"], "quantized evaluation")
    if set(baseline_cases) != set(quantized_cases):
        missing = sorted(set(baseline_cases) - set(quantized_cases))
        extra = sorted(set(quantized_cases) - set(baseline_cases))
        raise ValueError(f"Case IDs differ; missing={missing}, extra={extra}")
    if baseline.get("generation") != quantized.get("generation"):
        raise ValueError("Generation settings differ between source and Q4 evaluation.")

    comparisons = []
    baseline_strict = 0
    quantized_strict = 0
    new_strict_regressions = 0
    new_forbidden_leaks = 0
    exact_matches = 0
    status_matches = 0
    tokenization_mismatches = 0
    retained_fact_groups = 0
    baseline_fact_groups = 0
    similarities = []

    for case_id, source in baseline_cases.items():
        q4 = quantized_cases[case_id]
        if source.get("messages_sha256") != q4.get("messages_sha256"):
            raise ValueError(f"Message digest differs for {case_id}.")

        q4_score = answer_score(source, q4["answer"])
        source_strict = bool(source["strict_pass"])
        q4_strict = bool(q4_score["strict_pass"])
        exact_match = normalized(source["answer"]) == normalized(q4["answer"])
        similarity = SequenceMatcher(
            None, normalized(source["answer"]), normalized(q4["answer"])
        ).ratio()
        new_regression = source_strict and not q4_strict
        new_leak = bool(source["forbidden_pass"]) and not q4_score["forbidden_pass"]
        input_tokens_match = source.get("input_token_count") == q4.get("input_token_count")

        source_group_matches = source["matched_groups"]
        if len(source_group_matches) != len(q4_score["matched_groups"]):
            raise ValueError(f"Expected-group count differs for {case_id}.")
        for was_matched, remains_matched in zip(
            source_group_matches, q4_score["matched_groups"]
        ):
            if was_matched:
                baseline_fact_groups += 1
                retained_fact_groups += int(remains_matched)

        baseline_strict += int(source_strict)
        quantized_strict += int(q4_strict)
        new_strict_regressions += int(new_regression)
        new_forbidden_leaks += int(new_leak)
        exact_matches += int(exact_match)
        status_matches += int(source_strict == q4_strict)
        tokenization_mismatches += int(not input_tokens_match)
        similarities.append(similarity)
        source_seconds = source.get("generation_seconds", 0.0)
        q4_seconds = q4.get("generation_seconds", 0.0)
        source_tps = source.get("tokens_per_second", 0.0)
        q4_tps = q4.get("tokens_per_second", 0.0)
        comparisons.append(
            {
                "id": case_id,
                "behavior": source["behavior"],
                "prompt": source["prompt"],
                "source_answer": source["answer"],
                "quantized_answer": q4["answer"],
                "exact_answer_match": exact_match,
                "answer_similarity": similarity,
                "source_strict_pass": source_strict,
                "quantized_strict_pass": q4_strict,
                "new_strict_regression": new_regression,
                "new_forbidden_leak": new_leak,
                "quantized_matched_groups": q4_score["matched_groups"],
                "quantized_forbidden_matches": q4_score["forbidden_matches"],
                "input_tokens_match": input_tokens_match,
                "source_generated_token_count": source.get("generated_token_count", 0),
                "quantized_generated_token_count": q4.get("generated_token_count", 0),
                "source_generation_seconds": source_seconds,
                "quantized_generation_seconds": q4_seconds,
                "latency_ratio": q4_seconds / source_seconds if source_seconds else 0.0,
                "source_tokens_per_second": source_tps,
                "quantized_tokens_per_second": q4_tps,
                "throughput_ratio": q4_tps / source_tps if source_tps else 0.0,
            }
        )

    case_count = len(comparisons)
    source_strict_rate = rate(baseline_strict, case_count)
    q4_strict_rate = rate(quantized_strict, case_count)
    strict_drop = source_strict_rate - q4_strict_rate
    source_latencies = [item["source_generation_seconds"] for item in comparisons]
    q4_latencies = [item["quantized_generation_seconds"] for item in comparisons]
    source_throughputs = [item["source_tokens_per_second"] for item in comparisons]
    q4_throughputs = [item["quantized_tokens_per_second"] for item in comparisons]
    source_median_tps = median(source_throughputs)
    q4_median_tps = median(q4_throughputs)
    throughput_ratio = q4_median_tps / source_median_tps if source_median_tps else 0.0
    source_median_latency = median(source_latencies)
    q4_median_latency = median(q4_latencies)
    same_device_class = str(baseline.get("runtime", "")).endswith("-cpu") and str(
        quantized.get("runtime", "")
    ).endswith("-cpu")

    quality = {
        "source_strict_pass_rate": source_strict_rate,
        "quantized_strict_pass_rate": q4_strict_rate,
        "strict_score_drop": strict_drop,
        "exact_answer_match_rate": rate(exact_matches, case_count),
        "answer_change_rate": 1.0 - rate(exact_matches, case_count),
        "mean_answer_similarity": statistics.fmean(similarities) if similarities else 0.0,
        "strict_status_parity_rate": rate(status_matches, case_count),
        "key_fact_retention_rate": rate(retained_fact_groups, baseline_fact_groups),
        "new_strict_regression_count": new_strict_regressions,
        "new_forbidden_leak_count": new_forbidden_leaks,
        "input_tokenization_mismatch_count": tokenization_mismatches,
    }
    performance = {
        "source_runtime": baseline.get("runtime"),
        "quantized_runtime": quantized.get("runtime"),
        "source_median_generation_seconds": source_median_latency,
        "quantized_median_generation_seconds": q4_median_latency,
        "median_latency_ratio": (
            q4_median_latency / source_median_latency if source_median_latency else 0.0
        ),
        "median_latency_change_rate": (
            q4_median_latency / source_median_latency - 1.0
            if source_median_latency
            else 0.0
        ),
        "source_median_tokens_per_second": source_median_tps,
        "quantized_median_tokens_per_second": q4_median_tps,
        "throughput_ratio": throughput_ratio,
        "throughput_change_rate": throughput_ratio - 1.0,
        "same_device_class": same_device_class,
    }
    gates = {
        "comparable_performance_runtime": same_device_class,
        "same_input_tokenization": tokenization_mismatches == 0,
        "strict_score_drop": strict_drop <= thresholds["maximum_strict_score_drop"],
        "new_strict_regressions": (
            new_strict_regressions <= thresholds["maximum_new_strict_regressions"]
        ),
        "new_forbidden_leaks": (
            new_forbidden_leaks <= thresholds["maximum_new_forbidden_leaks"]
        ),
        "key_fact_retention": (
            quality["key_fact_retention_rate"] >= thresholds["minimum_key_fact_retention"]
        ),
        "answer_similarity": (
            quality["mean_answer_similarity"]
            >= thresholds["minimum_mean_answer_similarity"]
        ),
        "generation_throughput": (
            throughput_ratio >= thresholds["minimum_throughput_ratio"]
        ),
    }
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "source_model": baseline["model"],
        "quantized_model": quantized["model"],
        "case_count": case_count,
        "generation": baseline["generation"],
        "thresholds": thresholds,
        "quality": quality,
        "performance": performance,
        "gates": gates,
        "publication_allowed": all(gates.values()),
        "changed_answers": [
            item for item in comparisons if not item["exact_answer_match"]
        ],
        "comparisons": comparisons,
    }


def main() -> None:
    args = parse_args()
    thresholds = {
        "maximum_strict_score_drop": args.maximum_strict_score_drop,
        "maximum_new_strict_regressions": args.maximum_new_strict_regressions,
        "maximum_new_forbidden_leaks": args.maximum_new_forbidden_leaks,
        "minimum_key_fact_retention": args.minimum_key_fact_retention,
        "minimum_mean_answer_similarity": args.minimum_mean_answer_similarity,
        "minimum_throughput_ratio": args.minimum_throughput_ratio,
    }
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    quantized = json.loads(args.quantized.read_text(encoding="utf-8"))
    report = build_report(baseline, quantized, thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console_summary = {
        "quality": report["quality"],
        "performance": report["performance"],
        "gates": report["gates"],
    }
    print(json.dumps(console_summary, indent=2, ensure_ascii=False))
    if not report["publication_allowed"]:
        failed = [name for name, passed in report["gates"].items() if not passed]
        raise SystemExit("Quantization parity gate failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
