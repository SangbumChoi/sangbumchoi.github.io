#!/usr/bin/env python3
"""Score STT predictions globally and by speaker, domain, and environment."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--keywords", type=Path, default=Path("assets/data/daniel-stt-keywords.txt")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-wer", type=float)
    parser.add_argument("--max-worst-group-wer", type=float)
    parser.add_argument("--min-keyword-recall", type=float)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9가-힣']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_counts(reference: str, prediction: str) -> dict[str, int]:
    ref = normalize_text(reference).split()
    hyp = normalize_text(prediction).split()
    rows: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, index) for index in range(len(hyp) + 1)]
    ]
    for ref_index in range(1, len(ref) + 1):
        row = [(ref_index, 0, ref_index, 0)]
        for hyp_index in range(1, len(hyp) + 1):
            if ref[ref_index - 1] == hyp[hyp_index - 1]:
                row.append(rows[ref_index - 1][hyp_index - 1])
                continue
            diagonal = rows[ref_index - 1][hyp_index - 1]
            deletion = rows[ref_index - 1][hyp_index]
            insertion = row[hyp_index - 1]
            candidates = [
                (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            ]
            row.append(min(candidates))
        rows.append(row)
    distance, substitutions, deletions, insertions = rows[-1][-1]
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
    words = totals["reference_words"]
    return {**totals, "wer": totals["edits"] / words if words else 0.0}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, math.ceil(len(values) * fraction) - 1)
    return values[index]


def load_records(path: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"{path}: no prediction records")
    required = {"utterance_id", "transcript", "prediction", "speaker_id"}
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"{record.get('utterance_id')}: missing fields {sorted(missing)}")
    return records


def main() -> None:
    args = parse_args()
    records = load_records(args.predictions)
    keywords = [
        normalize_text(line)
        for line in args.keywords.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    scored = []
    grouped: dict[str, dict[str, list[dict]]] = {
        key: defaultdict(list) for key in ("speaker_id", "domain", "environment", "accent_group")
    }
    keyword_hits = 0
    keyword_opportunities = 0
    inference_times = []
    real_time_factors = []
    for record in records:
        counts = edit_counts(record["transcript"], record["prediction"])
        scored.append(counts)
        for key, values in grouped.items():
            values[str(record.get(key, "unspecified"))].append(counts)
        normalized_ref = normalize_text(record["transcript"])
        normalized_prediction = normalize_text(record["prediction"])
        for keyword in keywords:
            if keyword and re.search(rf"\b{re.escape(keyword)}\b", normalized_ref):
                keyword_opportunities += 1
                keyword_hits += int(
                    bool(re.search(rf"\b{re.escape(keyword)}\b", normalized_prediction))
                )
        if record.get("inference_seconds") is not None:
            inference_time = float(record["inference_seconds"])
            inference_times.append(inference_time)
            if record.get("duration_seconds"):
                real_time_factors.append(inference_time / float(record["duration_seconds"]))

    by_group = {
        key: {name: aggregate(values) for name, values in sorted(groups.items())}
        for key, groups in grouped.items()
    }
    speaker_wers = [value["wer"] for value in by_group["speaker_id"].values()]
    non_speaker_wers = [
        value["wer"]
        for key in ("domain", "environment", "accent_group")
        for value in by_group[key].values()
        if value["reference_words"]
    ]
    metrics = {
        "records": len(records),
        "overall": aggregate(scored),
        "macro_speaker_wer": statistics.fmean(speaker_wers),
        "worst_speaker_wer": max(speaker_wers),
        "worst_group_wer": max(non_speaker_wers) if non_speaker_wers else 0.0,
        "keyword_recall": keyword_hits / keyword_opportunities if keyword_opportunities else None,
        "keyword_hits": keyword_hits,
        "keyword_opportunities": keyword_opportunities,
        "latency_seconds_p50": statistics.median(inference_times) if inference_times else None,
        "latency_seconds_p95": percentile(inference_times, 0.95),
        "real_time_factor_p95": percentile(real_time_factors, 0.95),
        "groups": by_group,
    }
    failures = []
    if args.max_wer is not None and metrics["overall"]["wer"] > args.max_wer:
        failures.append(f"WER {metrics['overall']['wer']:.3f} > {args.max_wer:.3f}")
    if (
        args.max_worst_group_wer is not None
        and metrics["worst_group_wer"] > args.max_worst_group_wer
    ):
        failures.append(
            f"worst group WER {metrics['worst_group_wer']:.3f} > "
            f"{args.max_worst_group_wer:.3f}"
        )
    if (
        args.min_keyword_recall is not None
        and metrics["keyword_recall"] is not None
        and metrics["keyword_recall"] < args.min_keyword_recall
    ):
        failures.append(
            f"keyword recall {metrics['keyword_recall']:.3f} < {args.min_keyword_recall:.3f}"
        )
    metrics["passed"] = not failures
    metrics["failures"] = failures
    rendered = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
