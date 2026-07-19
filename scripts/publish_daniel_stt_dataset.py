#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=0.34,<2"]
# ///
"""Publish a prepared STT dataset privately unless public release is explicit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/daniel-stt-data"))
    parser.add_argument("--repo-id", default="danelcsb/daniel-os-stt")
    parser.add_argument("--public", action="store_true")
    return parser.parse_args()


def records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required to publish the dataset")
    manifest = args.data_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    dataset_records = records(manifest)
    if args.public:
        blocked = [
            record["utterance_id"]
            for record in dataset_records
            if record.get("allow_publication") is not True
        ]
        if blocked:
            raise ValueError(
                "Public publication requires allow_publication=true on every record: "
                + ", ".join(blocked[:10])
            )

    root_dir = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="daniel-stt-dataset-") as directory:
        staging = Path(directory)
        shutil.copytree(args.data_dir, staging, dirs_exist_ok=True)
        shutil.copy2(root_dir / "assets/data/daniel-stt-dataset-card.md", staging / "README.md")
        shutil.copy2(root_dir / "assets/data/daniel-stt-keywords.txt", staging / "keywords.txt")
        api = HfApi()
        api.create_repo(
            args.repo_id,
            repo_type="dataset",
            private=not args.public,
            exist_ok=True,
        )
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=staging,
            commit_message="Publish speaker-disjoint Daniel OS STT dataset",
        )
        print(f"Uploaded {args.repo_id}: {commit.oid}")


if __name__ == "__main__":
    main()
