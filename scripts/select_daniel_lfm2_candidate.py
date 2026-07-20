#!/usr/bin/env python3
"""Rank LFM2 ablations and allow publication only when baseline gates hold."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("assets/data/daniel-lfm2-strict-evaluation.json"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def metric(metrics: dict, key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    return float(value) if value is not None and math.isfinite(float(value)) else default


def best_macro_loss(training: dict) -> float:
    losses = [
        point["loss"]
        for point in training.get("validation", [])
        if point.get("dataset") in {"macro", "legacy"} and point.get("loss") is not None
    ]
    return min(losses) if losses else math.inf


def load_run(directory: Path, baseline_metrics: dict) -> dict | None:
    training_path = directory / "training_metrics.json"
    behavior_path = directory / "merged" / "evaluation.json"
    strict_path = directory / "strict-evaluation.json"
    if not all(path.exists() for path in (training_path, behavior_path, strict_path)):
        return None
    training = json.loads(training_path.read_text(encoding="utf-8"))
    behavior = json.loads(behavior_path.read_text(encoding="utf-8"))
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    strict_metrics = strict["metrics"]
    by_behavior = strict_metrics["by_behavior"]
    baseline_behavior = baseline_metrics["by_behavior"]
    gates = {
        "strict_not_below_baseline": metric(strict_metrics, "strict_pass_rate")
        >= metric(baseline_metrics, "strict_pass_rate"),
        "macro_not_below_baseline": metric(strict_metrics, "macro_behavior_pass_rate")
        >= metric(baseline_metrics, "macro_behavior_pass_rate"),
        "answers_not_below_baseline": metric(by_behavior, "answer")
        >= metric(baseline_behavior, "answer"),
        "unknown_not_below_baseline": metric(by_behavior, "unknown")
        >= metric(baseline_behavior, "unknown"),
        "retrieval_not_below_baseline": metric(by_behavior, "retrieve")
        >= metric(baseline_behavior, "retrieve"),
        "refusal_not_below_baseline": metric(by_behavior, "refuse")
        >= metric(baseline_behavior, "refuse"),
        "hallucination_guard_perfect": metric(strict_metrics, "hallucination_guard_rate") == 1.0,
        "korean_not_below_baseline": metric(strict_metrics, "korean_response_rate")
        >= metric(baseline_metrics, "korean_response_rate"),
    }
    loss = best_macro_loss(training)
    return {
        "run": directory.name,
        "path": str(directory),
        "publishable": all(gates.values()),
        "gates": gates,
        "strict_pass_rate": metric(strict_metrics, "strict_pass_rate"),
        "macro_behavior_pass_rate": metric(strict_metrics, "macro_behavior_pass_rate"),
        "hallucination_guard_rate": metric(strict_metrics, "hallucination_guard_rate"),
        "behavior_eval_overall": metric(behavior, "overall"),
        "best_macro_validation_loss": loss if math.isfinite(loss) else None,
        "by_behavior": by_behavior,
    }


def ranking_key(run: dict) -> tuple:
    loss = run["best_macro_validation_loss"]
    return (
        run["publishable"],
        run["strict_pass_rate"],
        run["macro_behavior_pass_rate"],
        -(loss if loss is not None else math.inf),
        run["hallucination_guard_rate"],
    )


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["metrics"]
    runs = [
        run
        for directory in sorted(args.sweep_root.iterdir())
        if directory.is_dir()
        for run in [load_run(directory, baseline)]
        if run is not None
    ]
    runs.sort(key=ranking_key, reverse=True)
    selected = runs[0] if runs and runs[0]["publishable"] else None
    payload = {
        "baseline": {
            "strict_pass_rate": baseline["strict_pass_rate"],
            "macro_behavior_pass_rate": baseline["macro_behavior_pass_rate"],
            "by_behavior": baseline["by_behavior"],
        },
        "selected": selected,
        "publication_allowed": selected is not None,
        "runs": runs,
    }
    output = args.output or args.sweep_root / "selection.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
