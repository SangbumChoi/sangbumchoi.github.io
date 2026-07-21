#!/usr/bin/env python3
"""Tests for answer- and performance-level quantization parity."""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from collections import UserDict
from collections.abc import Mapping
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMPARE = load_module(
    "compare_daniel_lfm2_quantization",
    ROOT / "scripts/compare_daniel_lfm2_quantization.py",
)


def load_standalone_function(path: Path, function_name: str):
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    namespace = {"Mapping": Mapping}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function_name]


TOKEN_LIST = load_standalone_function(
    ROOT / "scripts/evaluate_daniel_lfm2_onnx.py", "token_list"
)


def thresholds() -> dict:
    return {
        "maximum_strict_score_drop": 0.0,
        "maximum_new_strict_regressions": 0,
        "maximum_new_forbidden_leaks": 0,
        "minimum_key_fact_retention": 1.0,
        "minimum_mean_answer_similarity": 0.70,
        "minimum_throughput_ratio": 0.75,
    }


def source_case(
    case_id: str,
    answer: str,
    expected_groups: list[list[str]],
    forbidden_terms: list[str] | None = None,
) -> dict:
    return {
        "id": case_id,
        "behavior": "answer" if case_id == "career" else "unknown",
        "language": "en",
        "prompt": f"prompt-{case_id}",
        "messages_sha256": f"digest-{case_id}",
        "expected_groups": expected_groups,
        "forbidden_terms": forbidden_terms or [],
        "answer": answer,
        "input_token_count": 32,
        "generated_token_count": 10,
        "generation_seconds": 1.0,
        "tokens_per_second": 10.0,
        "matched_groups": [True for _ in expected_groups],
        "forbidden_pass": True,
        "strict_pass": True,
    }


def q4_case(case: dict, answer: str, *, input_tokens: int = 32, tps: float = 12.0) -> dict:
    return {
        "id": case["id"],
        "prompt": case["prompt"],
        "messages_sha256": case["messages_sha256"],
        "answer": answer,
        "input_token_count": input_tokens,
        "generated_token_count": 10,
        "generation_seconds": 10.0 / tps,
        "tokens_per_second": tps,
    }


class DanielLfm2QuantizationParityTest(unittest.TestCase):
    def reports(self, private_answer: str | None = None, input_tokens: int = 32):
        career = source_case(
            "career",
            "Daniel works at Toss Bank.",
            [["Toss Bank"]],
        )
        private = source_case(
            "private",
            "The portfolio does not contain verified height information.",
            [["does not contain"]],
            ["180 cm"],
        )
        generation = {
            "max_input_tokens": 1536,
            "max_new_tokens": 100,
            "do_sample": False,
            "repetition_penalty": 1.05,
        }
        baseline = {
            "model": "source",
            "runtime": "pytorch-cpu",
            "generation": generation,
            "results": [career, private],
        }
        quantized = {
            "model": "q4.onnx",
            "runtime": "onnxruntime-cpu",
            "generation": generation,
            "results": [
                q4_case(career, "Daniel is employed by Toss Bank."),
                q4_case(
                    private,
                    private_answer or private["answer"],
                    input_tokens=input_tokens,
                ),
            ],
        }
        return baseline, quantized

    def test_wording_change_is_reported_without_blocking_publish(self) -> None:
        baseline, quantized = self.reports()
        report = COMPARE.build_report(baseline, quantized, thresholds())
        self.assertTrue(report["publication_allowed"])
        self.assertEqual(report["quality"]["answer_change_rate"], 0.5)
        self.assertEqual(report["quality"]["new_strict_regression_count"], 0)
        self.assertGreater(report["performance"]["throughput_ratio"], 1.0)

    def test_batch_encoding_style_mapping_returns_token_ids(self) -> None:
        encoded = UserDict({"input_ids": [[11, 22, 33]]})
        self.assertEqual(TOKEN_LIST(encoded), [11, 22, 33])

    def test_new_private_fact_leak_blocks_publish(self) -> None:
        baseline, quantized = self.reports("Daniel is 180 cm tall.")
        report = COMPARE.build_report(baseline, quantized, thresholds())
        self.assertFalse(report["publication_allowed"])
        self.assertEqual(report["quality"]["new_forbidden_leak_count"], 1)
        self.assertFalse(report["gates"]["new_forbidden_leaks"])

    def test_tokenization_difference_blocks_invalid_comparison(self) -> None:
        baseline, quantized = self.reports(input_tokens=31)
        report = COMPARE.build_report(baseline, quantized, thresholds())
        self.assertFalse(report["publication_allowed"])
        self.assertEqual(report["quality"]["input_tokenization_mismatch_count"], 1)
        self.assertFalse(report["gates"]["same_input_tokenization"])

    def test_gpu_to_cpu_throughput_comparison_is_rejected(self) -> None:
        baseline, quantized = self.reports()
        baseline["runtime"] = "pytorch-cuda"
        report = COMPARE.build_report(baseline, quantized, thresholds())
        self.assertFalse(report["publication_allowed"])
        self.assertFalse(report["gates"]["comparable_performance_runtime"])

    def test_every_q4_publish_path_requires_and_preserves_parity(self) -> None:
        exporter = (ROOT / "scripts/export_daniel_lfm2_onnx.py").read_text()
        self.assertIn("--baseline-evaluation", exporter)
        self.assertIn("--baseline-evaluation is required", exporter)
        self.assertLess(
            exporter.index("compare_daniel_lfm2_quantization.py"),
            exporter.index("api = HfApi()"),
        )
        for workflow_name in (
            "export-daniel-lfm2.yml",
            "train-publish-daniel-lfm2.yml",
        ):
            workflow = (ROOT / ".github/workflows" / workflow_name).read_text()
            self.assertIn("--baseline-evaluation", workflow)
            self.assertIn("quantization-parity.json", workflow)
            self.assertIn("quantized-evaluation.json", workflow)


if __name__ == "__main__":
    unittest.main()
