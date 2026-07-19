#!/usr/bin/env python3
"""Validate consented STT audio and create speaker-disjoint dataset splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import unicodedata
import wave
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_FIELDS = {
    "utterance_id",
    "audio_path",
    "transcript",
    "speaker_id",
    "session_id",
    "language",
    "source",
    "domain",
    "environment",
    "consent",
}
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/daniel-stt-data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--minimum-speakers", type=int, default=3)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--no-copy-audio", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9가-힣']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wav(path: Path) -> tuple[int, float]:
    try:
        with wave.open(str(path), "rb") as stream:
            if stream.getcomptype() != "NONE":
                raise ValueError("must be uncompressed PCM WAV")
            if stream.getnchannels() != 1:
                raise ValueError("must contain one mono channel")
            sample_rate = stream.getframerate()
            duration = stream.getnframes() / sample_rate
    except wave.Error as error:
        raise ValueError(f"invalid WAV file: {error}") from error
    if sample_rate != 16_000:
        raise ValueError(f"must be 16 kHz, found {sample_rate} Hz")
    if not 0.4 <= duration <= 30.0:
        raise ValueError(f"duration must be 0.4-30 seconds, found {duration:.2f}")
    return sample_rate, duration


def validate_metadata(record: dict, known_ids: set[str]) -> None:
    missing = REQUIRED_FIELDS - record.keys()
    record_id = record.get("utterance_id", "<missing-id>")
    if missing:
        raise ValueError(f"{record_id}: missing fields {sorted(missing)}")
    if record_id in known_ids:
        raise ValueError(f"{record_id}: duplicate utterance_id")
    known_ids.add(record_id)
    for field in REQUIRED_FIELDS:
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{record_id}: {field} must be a non-empty string")
    if record["language"] != "en":
        raise ValueError(f"{record_id}: the current browser target is English-only")
    if record["consent"].lower() in {"none", "false", "unknown"}:
        raise ValueError(f"{record_id}: explicit recording consent is required")
    if record["source"] == "public" and not record.get("license"):
        raise ValueError(f"{record_id}: public audio must include its license")
    if not normalize_text(record["transcript"]):
        raise ValueError(f"{record_id}: transcript is empty after normalization")
    split = record.get("split")
    if split is not None and split not in SPLITS:
        raise ValueError(f"{record_id}: invalid split {split}")


def assign_speaker_splits(
    records: list[dict], seed: int, validation_ratio: float, test_ratio: float
) -> None:
    provided = ["split" in record for record in records]
    if any(provided) and not all(provided):
        raise ValueError("Either every record must define split or none may define it")
    if all(provided):
        return

    speakers = sorted({record["speaker_id"] for record in records})
    if len(speakers) < 3:
        raise ValueError("At least three speakers are needed for speaker-disjoint splits")
    rng = random.Random(seed)
    rng.shuffle(speakers)
    validation_count = max(1, round(len(speakers) * validation_ratio))
    test_count = max(1, round(len(speakers) * test_ratio))
    if validation_count + test_count >= len(speakers):
        validation_count = 1
        test_count = 1
    speaker_splits = {
        **{speaker: "validation" for speaker in speakers[:validation_count]},
        **{
            speaker: "test"
            for speaker in speakers[validation_count : validation_count + test_count]
        },
        **{
            speaker: "train"
            for speaker in speakers[validation_count + test_count :]
        },
    }
    for record in records:
        record["split"] = speaker_splits[record["speaker_id"]]


def validate_disjoint_splits(records: list[dict]) -> None:
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    session_splits: dict[str, set[str]] = defaultdict(set)
    digest_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        speaker_splits[record["speaker_id"]].add(record["split"])
        session_key = f'{record["speaker_id"]}:{record["session_id"]}'
        session_splits[session_key].add(record["split"])
        if record.get("sha256"):
            digest_splits[record["sha256"]].add(record["split"])
    for label, mapping in (
        ("speaker", speaker_splits),
        ("session", session_splits),
        ("audio digest", digest_splits),
    ):
        leaked = {key: sorted(value) for key, value in mapping.items() if len(value) > 1}
        if leaked:
            raise ValueError(f"Cross-split {label} leakage: {leaked}")
    missing = [split for split in SPLITS if not any(r["split"] == split for r in records)]
    if missing:
        raise ValueError(f"Dataset is missing splits: {missing}")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.validation_ratio <= 0 or args.test_ratio <= 0:
        raise ValueError("Validation and test ratios must be positive")
    if args.validation_ratio + args.test_ratio >= 1:
        raise ValueError("Validation and test ratios must sum to less than one")

    records = read_jsonl(args.manifest)
    known_ids: set[str] = set()
    for record in records:
        validate_metadata(record, known_ids)
    speakers = {record["speaker_id"] for record in records}
    if len(speakers) < args.minimum_speakers:
        raise ValueError(
            f"Found {len(speakers)} speakers; require at least {args.minimum_speakers}"
        )

    output_dir = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    for record in records:
        source = (args.manifest.parent / record["audio_path"]).resolve()
        if args.schema_only:
            record.setdefault("sample_rate_hz", 16_000)
            record.setdefault("duration_seconds", 1.0)
            record.setdefault("sha256", f"schema-only:{record['utterance_id']}")
            continue
        if not source.is_file():
            raise FileNotFoundError(f"{record['utterance_id']}: {source}")
        sample_rate, duration = inspect_wav(source)
        digest = file_sha256(source)
        if record.get("sha256") and record["sha256"] != digest:
            raise ValueError(f"{record['utterance_id']}: sha256 does not match audio")
        record["sample_rate_hz"] = sample_rate
        record["duration_seconds"] = round(duration, 4)
        record["sha256"] = digest
        if args.no_copy_audio:
            record["audio_path"] = str(source)
        else:
            audio_dir.mkdir(parents=True, exist_ok=True)
            destination = audio_dir / f"{record['utterance_id']}.wav"
            shutil.copy2(source, destination)
            record["audio_path"] = str(destination.relative_to(output_dir))

    for record in records:
        record["normalized_text"] = normalize_text(record["transcript"])
    assign_speaker_splits(records, args.seed, args.validation_ratio, args.test_ratio)
    validate_disjoint_splits(records)

    for split in SPLITS:
        split_records = [record for record in records if record["split"] == split]
        write_jsonl(output_dir / f"{split}.jsonl", split_records)
    write_jsonl(output_dir / "manifest.jsonl", records)

    split_counts = Counter(record["split"] for record in records)
    split_hours = {
        split: round(
            sum(record["duration_seconds"] for record in records if record["split"] == split)
            / 3600,
            3,
        )
        for split in SPLITS
    }
    summary = {
        "records": len(records),
        "speakers": len(speakers),
        "split_records": dict(split_counts),
        "split_hours": split_hours,
        "domains": dict(Counter(record["domain"] for record in records)),
        "environments": dict(Counter(record["environment"] for record in records)),
        "schema_only": args.schema_only,
        "speaker_disjoint": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
