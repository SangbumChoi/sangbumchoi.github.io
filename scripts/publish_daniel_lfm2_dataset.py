#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=0.34,<2"]
# ///
"""Publish Daniel OS train, validation, test, provenance, and metrics to the Hub."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="danelcsb/daniel-os-profile-sft")
    parser.add_argument("--evaluation", type=Path)
    return parser.parse_args()


def copy(source: str | Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required to publish the dataset")
    with tempfile.TemporaryDirectory(prefix="daniel-os-dataset-") as directory:
        root = Path(directory)
        copy("assets/data/daniel-lfm2-dataset-card.md", root / "README.md")
        copy("assets/data/daniel-lfm2-sft.jsonl", root / "sft/train.jsonl")
        copy("assets/data/daniel-lfm2-eval.jsonl", root / "behavior_eval/validation.jsonl")
        copy("assets/data/daniel-lfm2-test.jsonl", root / "strict_test/test.jsonl")
        copy("assets/data/daniel-profile.json", root / "profile/profile.json")
        copy("assets/data/daniel-profile-sources.json", root / "profile/profile-sources.json")
        copy("assets/data/daniel-lfm2-training-metrics.json", root / "metrics/training.json")
        if args.evaluation and args.evaluation.exists():
            copy(args.evaluation, root / "metrics/strict-evaluation.json")

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True)
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=root,
            commit_message="Publish sourced profile SFT and strict behavior benchmark",
        )
        print(f"Uploaded {args.repo_id}: {commit.oid}")


if __name__ == "__main__":
    main()
