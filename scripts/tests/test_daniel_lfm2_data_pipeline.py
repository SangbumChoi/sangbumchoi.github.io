#!/usr/bin/env python3
"""Tests for the leakage-resistant Daniel OS SFT preparation pipeline."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ANALYZE = load_module("analyze_daniel_lfm2_data", ROOT / "scripts/analyze_daniel_lfm2_data.py")
GENERATE = load_module(
    "generate_daniel_lfm2_synthetic",
    ROOT / "scripts/generate_daniel_lfm2_synthetic.py",
)


class DanielLfm2DataPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = GENERATE.read_jsonl(
            [
                ROOT / "assets/data/daniel-lfm2-sft.jsonl",
                ROOT / "assets/data/daniel-lfm2-routing-sft.jsonl",
            ]
        )
        cls.plan = json.loads(
            (ROOT / "assets/data/daniel-lfm2-generation-plan.json").read_text()
        )

    def test_legacy_sampler_reconstructs_repeat_inflation(self) -> None:
        counts = Counter(record["behavior"] for record in self.records)
        sampling = ANALYZE.legacy_sampling(counts, 64, 2)
        self.assertEqual(sampling["loss_holdout_total"], 10)
        self.assertEqual(sampling["effective_training_total_per_epoch"], 431)
        self.assertEqual(sampling["repeated_slots_per_epoch"], 145)
        self.assertEqual(sampling["by_behavior"]["ground_external"]["repeat_factor"], 6.4)

    def test_scenario_families_do_not_cross_splits(self) -> None:
        records = GENERATE.enrich_entity_seeds(
            self.records, ROOT / "assets/data/daniel-entity-knowledge.json"
        )
        records = GENERATE.enrich_retrieval_seeds(
            records, ROOT / "assets/data/daniel-lfm2-public-topic-seeds.json"
        )
        assignments = GENERATE.assign_seed_splits(records, self.plan, 42)
        families = {}
        for record in records:
            family = record.get("generation", {}).get(
                "scenario_family", f"seed:{record['id']}"
            )
            split = assignments[record["id"]]
            if family in families:
                self.assertEqual(families[family], split)
            families[family] = split
        self.assertEqual(set(assignments.values()), {"train", "validation"})

    def test_public_topics_train_retrieval_not_world_facts(self) -> None:
        records = GENERATE.enrich_retrieval_seeds(
            [], ROOT / "assets/data/daniel-lfm2-public-topic-seeds.json"
        )
        self.assertGreaterEqual(len(records), 50)
        for record in records:
            answer = record["messages"][-1]["content"]
            self.assertEqual(record["behavior"], "retrieve")
            self.assertRegex(
                answer,
                r"^<search_public_knowledge>[^<>]+</search_public_knowledge>$",
            )

    def test_candidate_filter_rejects_duplicates_and_wrong_language(self) -> None:
        seed = self.records[0]
        prompt = seed["messages"][GENERATE.final_user_index(seed["messages"])]["content"]
        accepted, reason = GENERATE.validate_candidate(
            {"prompt": prompt}, seed, [prompt], self.plan["maximum_prompt_token_jaccard"]
        )
        self.assertFalse(accepted)
        self.assertIn(reason, {"exact_duplicate", "too_similar_to_seed"})
        accepted, reason = GENERATE.validate_candidate(
            {"prompt": "최상범은 누구야?"},
            seed,
            [prompt],
            self.plan["maximum_prompt_token_jaccard"],
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "language")

    def test_notebook_never_contains_a_hugging_face_token(self) -> None:
        notebook = (ROOT / "notebooks/daniel_lfm2_gpu_retraining.ipynb").read_text()
        self.assertNotIn("hf_ke", notebook)
        self.assertIn("userdata.get('HF_TOKEN')", notebook)


if __name__ == "__main__":
    unittest.main()
