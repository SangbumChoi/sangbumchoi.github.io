#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9,<4"]
# ///
"""Plot committed STT training and validation loss without inventing points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("assets/images/daniel-stt-loss.png"))
    args = parser.parse_args()
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    history = payload.get("history", [])
    train = [(row["epoch"], row["loss"]) for row in history if "epoch" in row and "loss" in row]
    validation = [
        (row["epoch"], row["eval_loss"])
        for row in history
        if "epoch" in row and "eval_loss" in row
    ]
    if not train or not validation:
        raise ValueError("Metrics must contain real train and validation loss points")

    plt.style.use("dark_background")
    figure, axis = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    axis.plot(*zip(*train), color="#b7f24a", linewidth=2, label="Train loss")
    axis.plot(*zip(*validation), color="#ff6b57", linewidth=2, marker="o", label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Sequence cross-entropy")
    axis.grid(alpha=0.14)
    axis.legend(frameon=False)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", facecolor=figure.get_facecolor())


if __name__ == "__main__":
    main()
