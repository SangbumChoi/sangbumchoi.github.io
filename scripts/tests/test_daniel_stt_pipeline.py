#!/usr/bin/env python3
"""Offline smoke tests for STT split preparation and metric scoring."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Command failed:\n{result.stdout}{result.stderr}")
    return result


def write_wav(path: Path, sample_value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        sample = sample_value.to_bytes(2, byteorder="little", signed=True)
        stream.writeframes(sample * 8_000)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="daniel-stt-test-") as directory:
        root = Path(directory)
        audio = root / "audio"
        manifest = root / "raw.jsonl"
        records = []
        for speaker_index in range(6):
            for utterance_index in range(2):
                utterance_id = f"speaker_{speaker_index}_{utterance_index}"
                write_wav(audio / f"{utterance_id}.wav", 100 + speaker_index * 10 + utterance_index)
                records.append(
                    {
                        "utterance_id": utterance_id,
                        "audio_path": f"audio/{utterance_id}.wav",
                        "transcript": f"Speaker {speaker_index} asks about Toss Bank",
                        "speaker_id": f"speaker_{speaker_index}",
                        "session_id": f"session_{speaker_index}",
                        "language": "en",
                        "source": "consented",
                        "domain": "portfolio",
                        "environment": "quiet_mobile",
                        "accent_group": "test_group",
                        "consent": "test-v1",
                    }
                )
        manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
        prepared = root / "prepared"
        run_checked(
            [
                sys.executable,
                str(ROOT / "scripts/prepare_daniel_stt_dataset.py"),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(prepared),
            ],
        )
        speaker_splits: dict[str, set[str]] = {}
        for split in ("train", "validation", "test"):
            split_records = read_jsonl(prepared / f"{split}.jsonl")
            assert split_records
            for record in split_records:
                speaker_splits.setdefault(record["speaker_id"], set()).add(split)
                assert (prepared / record["audio_path"]).is_file()
        assert all(len(splits) == 1 for splits in speaker_splits.values())

        predictions = root / "predictions.jsonl"
        prediction_records = [
            {
                "utterance_id": "one",
                "transcript": "Tell me about Toss Bank",
                "prediction": "Tell me about Toss Bank",
                "speaker_id": "speaker_test_a",
                "domain": "portfolio",
                "environment": "quiet",
                "accent_group": "group_a",
            },
            {
                "utterance_id": "two",
                "transcript": "Explain Hugging Face Transformers",
                "prediction": "Explain Hugging Phase Transformers",
                "speaker_id": "speaker_test_b",
                "domain": "open_source",
                "environment": "office",
                "accent_group": "group_b",
            },
        ]
        predictions.write_text(
            "".join(json.dumps(record) + "\n" for record in prediction_records)
        )
        result = run_checked(
            [
                sys.executable,
                str(ROOT / "scripts/score_daniel_stt_predictions.py"),
                "--predictions",
                str(predictions),
                "--keywords",
                str(ROOT / "assets/data/daniel-stt-keywords.txt"),
            ],
        )
        metrics = json.loads(result.stdout)
        assert 0 < metrics["overall"]["wer"] < 1
        assert metrics["keyword_recall"] < 1
        assert metrics["groups"]["domain"]["portfolio"]["wer"] == 0
    print("Daniel STT pipeline smoke test passed")


if __name__ == "__main__":
    main()
