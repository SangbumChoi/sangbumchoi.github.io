#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "torch>=2.6",
#   "transformers>=4.55,<5",
#   "accelerate>=1.2,<2",
#   "bitsandbytes>=0.45,<1; platform_system == 'Linux'",
# ]
# ///
"""Generate grounded prompt diversity without asking a teacher to invent facts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable


BEHAVIORS = ("answer", "ground_external", "retrieve", "unknown", "refuse")
PARAPHRASE_SYSTEM = """You create training prompts for a small portfolio assistant.
Return valid JSON only. Preserve the original intent, requested facts, behavior, language, names, and numbers.
Do not answer the prompt. Do not add facts, people, employers, achievements, dates, metrics, or private details.
Produce natural phrasings with genuinely different syntax: direct questions, concise requests, indirect requests,
and realistic follow-ups. Do not use meta phrases such as 'verified context', 'training data', or 'according to the portfolio'."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training",
        nargs="+",
        type=Path,
        default=[
            Path("assets/data/daniel-lfm2-sft.jsonl"),
            Path("assets/data/daniel-lfm2-routing-sft.jsonl"),
        ],
    )
    parser.add_argument(
        "--entity-knowledge",
        type=Path,
        default=Path("assets/data/daniel-entity-knowledge.json"),
    )
    parser.add_argument(
        "--public-topics",
        type=Path,
        default=Path("assets/data/daniel-lfm2-public-topic-seeds.json"),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("assets/data/daniel-lfm2-generation-plan.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/daniel-lfm2-v3"))
    parser.add_argument("--teacher-model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Build leakage-resistant seed splits without loading a teacher model.",
    )
    return parser.parse_args()


def read_jsonl(paths: list[Path]) -> list[dict]:
    return [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def final_user_index(messages: list[dict]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "user":
            return index
    raise ValueError("Conversation has no user message")


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9가-힣]+", text.lower()))


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize(left).split())
    right_tokens = set(normalize(right).split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def stable_score(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def enrich_entity_seeds(records: list[dict], entity_path: Path) -> list[dict]:
    entities = json.loads(entity_path.read_text(encoding="utf-8"))["entities"]
    enriched = list(records)
    existing_prompts = {
        normalize(record["messages"][final_user_index(record["messages"])]["content"])
        for record in records
    }
    patterns = {
        "en": {
            "definition": ["What is {name}?", "Explain {name} in one sentence."],
        },
        "ko": {
            "definition": ["{name}가 무엇인지 설명해 줘.", "{name}의 개념을 한 문장으로 알려줘."],
        },
    }
    for entity in entities:
        evidence = {
            "entity": entity["name"],
            "definition": entity["definition_en"],
            "definition_ko": entity["definition_ko"],
            "sources": [source["url"] for source in entity["sources"]],
        }
        for language in ("en", "ko"):
            mode = "definition"
            answer = entity[f"definition_{language}"]
            for pattern_index, pattern in enumerate(patterns[language][mode]):
                prompt = pattern.format(name=entity["name"])
                if normalize(prompt) in existing_prompts:
                    continue
                existing_prompts.add(normalize(prompt))
                enriched.append(
                    {
                        "id": f"entity_{entity['id']}_{mode}_{language}_{pattern_index}",
                        "behavior": "ground_external",
                        "language": language,
                        "context_keys": [],
                        "evidence": evidence,
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer},
                        ],
                        "expected_terms": [entity["name"]],
                        "generation": {
                            "origin": "curated_entity_index",
                            "scenario_family": f"entity:{entity['id']}:{mode}:{language}",
                        },
                    }
                )
    return enriched


def enrich_retrieval_seeds(records: list[dict], topic_path: Path) -> list[dict]:
    topics = json.loads(topic_path.read_text(encoding="utf-8"))["topics"]
    enriched = list(records)
    existing_prompts = {
        normalize(record["messages"][final_user_index(record["messages"])]["content"])
        for record in records
    }
    for topic_index, topic in enumerate(topics):
        for language in ("en", "ko"):
            prompt = topic[f"prompt_{language}"]
            if normalize(prompt) in existing_prompts:
                continue
            existing_prompts.add(normalize(prompt))
            enriched.append(
                {
                    "id": f"public_topic_{topic_index:03d}_{language}",
                    "behavior": "retrieve",
                    "language": language,
                    "context_keys": [],
                    "messages": [
                        {"role": "user", "content": prompt},
                        {
                            "role": "assistant",
                            "content": f"<search_public_knowledge>{topic['term']}</search_public_knowledge>",
                        },
                    ],
                    "expected_terms": [topic["term"]],
                    "generation": {
                        "origin": "curated_public_topic",
                        "scenario_family": f"public_topic:{topic_index}:{language}",
                    },
                }
            )
    return enriched


def assign_seed_splits(records: list[dict], plan: dict, seed: int) -> dict[str, str]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        family = record.get("generation", {}).get(
            "scenario_family", f"seed:{record['id']}"
        )
        grouped[record["behavior"]][family].append(record)
    assignments = {}
    fraction = plan["validation_seed_fraction"]
    minimum = plan["minimum_validation_seeds_per_behavior"]
    for behavior, families in grouped.items():
        ordered = sorted(families, key=lambda family: stable_score(family, seed))
        validation_count = min(
            len(ordered) - 1,
            max(minimum, math.ceil(len(ordered) * fraction)),
        )
        for index, family in enumerate(ordered):
            split = "validation" if index < validation_count else "train"
            for record in families[family]:
                assignments[record["id"]] = split
    return assignments


def parse_json_payload(text: str) -> list[dict]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    first = cleaned.find("[")
    last = cleaned.rfind("]")
    if first < 0 or last < first:
        raise ValueError("Teacher response did not contain a JSON array")
    payload = json.loads(cleaned[first : last + 1])
    if not isinstance(payload, list):
        raise ValueError("Teacher response is not a list")
    return payload


def generation_request(record: dict, count: int) -> str:
    messages = record["messages"]
    prompt = messages[final_user_index(messages)]["content"]
    history = [message for message in messages[:-1] if message["role"] == "user"]
    return json.dumps(
        {
            "task": f"Create {count} diverse paraphrases of the final user prompt.",
            "behavior": record["behavior"],
            "language": record.get("language", "en"),
            "prior_user_turns": [message["content"] for message in history[:-1]],
            "final_user_prompt": prompt,
            "output_schema": [
                {
                    "prompt": "string",
                    "difficulty": "direct|indirect|follow_up|adversarial",
                    "variation_type": "short description",
                }
            ],
        },
        ensure_ascii=False,
    )


def load_teacher(model_id: str, max_new_tokens: int, temperature: float) -> Callable:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quantization = None
    if torch.cuda.is_available():
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype,
        quantization_config=quantization,
    ).eval()

    def generate(requests: list[str]) -> list[str]:
        rendered = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": PARAPHRASE_SYSTEM},
                    {"role": "user", "content": request},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for request in requests
        ]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.05,
            )
        return [
            tokenizer.decode(output[inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            for output in outputs
        ]

    return generate


def language_matches(prompt: str, language: str) -> bool:
    has_korean = bool(re.search(r"[가-힣]", prompt))
    return has_korean if language == "ko" else not has_korean


def validate_candidate(
    candidate: dict,
    seed_record: dict,
    accepted_prompts: list[str],
    maximum_similarity: float,
) -> tuple[bool, str]:
    prompt = candidate.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return False, "missing_prompt"
    prompt = prompt.strip()
    words = prompt.split()
    if len(words) < 2 or len(words) > 80:
        return False, "length"
    if not language_matches(prompt, seed_record.get("language", "en")):
        return False, "language"
    if any(marker in prompt.lower() for marker in ("training data", "verified context", "assistant:")):
        return False, "meta_language"
    if any(normalize(prompt) == normalize(existing) for existing in accepted_prompts):
        return False, "exact_duplicate"
    seed_prompt = seed_record["messages"][final_user_index(seed_record["messages"])]["content"]
    if token_jaccard(prompt, seed_prompt) > maximum_similarity:
        return False, "too_similar_to_seed"
    if any(token_jaccard(prompt, existing) > 0.92 for existing in accepted_prompts[-500:]):
        return False, "near_duplicate"
    return True, "accepted"


def synthetic_record(seed_record: dict, candidate: dict, index: int, teacher_model: str) -> dict:
    messages = [dict(message) for message in seed_record["messages"]]
    messages[final_user_index(messages)]["content"] = candidate["prompt"].strip()
    record = {
        key: value
        for key, value in seed_record.items()
        if key not in {"id", "messages", "generation"}
    }
    record.update(
        {
            "id": f"synthetic_{seed_record['id']}_{index:03d}",
            "messages": messages,
            "generation": {
                "origin": "teacher_prompt_paraphrase",
                "teacher_model": teacher_model,
                "seed_id": seed_record["id"],
                "scenario_family": seed_record.get("generation", {}).get(
                    "scenario_family", f"seed:{seed_record['id']}"
                ),
                "difficulty": candidate.get("difficulty", "unspecified"),
                "variation_type": candidate.get("variation_type", "unspecified"),
                "answer_source": "curated_seed_answer",
            },
        }
    )
    return record


def generate_split(
    split_records: list[dict],
    targets: dict[str, int],
    generator: Callable,
    teacher_model: str,
    plan: dict,
    batch_size: int,
    seed: int,
    accepted_prompts: list[str],
) -> tuple[list[dict], Counter]:
    by_behavior: dict[str, list[dict]] = defaultdict(list)
    for record in split_records:
        by_behavior[record["behavior"]].append(record)
    needed_by_seed = {}
    for behavior, behavior_records in by_behavior.items():
        additional = max(0, targets[behavior] - len(behavior_records))
        per_seed = math.ceil(additional / len(behavior_records)) if additional else 0
        for record in behavior_records:
            needed_by_seed[record["id"]] = per_seed

    synthetic = []
    rejection_counts: Counter = Counter()
    accepted_by_seed: Counter = Counter()
    ordered_seeds = sorted(split_records, key=lambda item: stable_score(item["id"], seed))
    for _round in range(plan["generation_rounds"]):
        remaining_seeds = [
            record
            for record in ordered_seeds
            if accepted_by_seed[record["id"]] < needed_by_seed.get(record["id"], 0)
        ]
        if not remaining_seeds:
            break
        for batch_start in range(0, len(remaining_seeds), batch_size):
            batch = remaining_seeds[batch_start : batch_start + batch_size]
            requests = []
            request_records = []
            for record in batch:
                remaining = needed_by_seed[record["id"]] - accepted_by_seed[record["id"]]
                request_count = min(
                    remaining + 3,
                    plan["maximum_generation_attempts_per_seed"],
                )
                requests.append(generation_request(record, request_count))
                request_records.append(record)
            responses = generator(requests)
            for seed_record, response in zip(request_records, responses):
                needed = needed_by_seed[seed_record["id"]]
                try:
                    candidates = parse_json_payload(response)
                except (ValueError, json.JSONDecodeError):
                    rejection_counts["invalid_json"] += 1
                    continue
                for candidate in candidates:
                    valid, reason = validate_candidate(
                        candidate,
                        seed_record,
                        accepted_prompts,
                        plan["maximum_prompt_token_jaccard"],
                    )
                    if not valid:
                        rejection_counts[reason] += 1
                        continue
                    accepted_by_seed[seed_record["id"]] += 1
                    record = synthetic_record(
                        seed_record,
                        candidate,
                        accepted_by_seed[seed_record["id"]],
                        teacher_model,
                    )
                    synthetic.append(record)
                    accepted_prompts.append(candidate["prompt"])
                    if accepted_by_seed[seed_record["id"]] >= needed:
                        break
    return synthetic, rejection_counts


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    teacher_model = args.teacher_model or plan["teacher_model"]
    records = enrich_entity_seeds(read_jsonl(args.training), args.entity_knowledge)
    records = enrich_retrieval_seeds(records, args.public_topics)
    assignments = assign_seed_splits(records, plan, args.seed)
    seed_train = [record for record in records if assignments[record["id"]] == "train"]
    seed_validation = [record for record in records if assignments[record["id"]] == "validation"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "seed-train.jsonl", seed_train)
    write_jsonl(args.output_dir / "seed-validation.jsonl", seed_validation)

    if args.seed_only:
        report = {
            "mode": "seed_only",
            "seed_counts": {
                "train": dict(Counter(record["behavior"] for record in seed_train)),
                "validation": dict(Counter(record["behavior"] for record in seed_validation)),
            },
            "note": "No teacher model was loaded and no synthetic records were accepted.",
        }
        (args.output_dir / "generation-report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return

    generator = load_teacher(teacher_model, args.max_new_tokens, args.temperature)
    accepted_prompts = [
        record["messages"][final_user_index(record["messages"])]["content"] for record in records
    ]
    synthetic_train, train_rejections = generate_split(
        seed_train,
        plan["target_train_records"],
        generator,
        teacher_model,
        plan,
        args.batch_size,
        args.seed,
        accepted_prompts,
    )
    synthetic_validation, validation_rejections = generate_split(
        seed_validation,
        plan["target_validation_records"],
        generator,
        teacher_model,
        plan,
        args.batch_size,
        args.seed + 1,
        accepted_prompts,
    )

    train = seed_train + synthetic_train
    validation = seed_validation + synthetic_validation
    random.Random(args.seed).shuffle(train)
    random.Random(args.seed).shuffle(validation)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    report = {
        "mode": "teacher_prompt_paraphrase",
        "teacher_model": teacher_model,
        "policy": "The teacher writes prompts only; every target answer remains curated.",
        "split_policy": "All variants of one seed remain in the same split.",
        "counts": {
            "train": dict(Counter(record["behavior"] for record in train)),
            "validation": dict(Counter(record["behavior"] for record in validation)),
            "synthetic_train": dict(Counter(record["behavior"] for record in synthetic_train)),
            "synthetic_validation": dict(
                Counter(record["behavior"] for record in synthetic_validation)
            ),
        },
        "rejections": {
            "train": dict(train_rejections),
            "validation": dict(validation_rejections),
        },
    }
    (args.output_dir / "generation-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    final_train_counts = Counter(record["behavior"] for record in train)
    final_validation_counts = Counter(record["behavior"] for record in validation)
    shortfalls = [
        f"{split}:{behavior}={counts[behavior]}/{targets[behavior]}"
        for split, counts, targets in (
            ("train", final_train_counts, plan["target_train_records"]),
            ("validation", final_validation_counts, plan["target_validation_records"]),
        )
        for behavior in BEHAVIORS
        if counts[behavior] < targets[behavior] * plan["minimum_target_fraction"]
    ]
    if shortfalls:
        raise RuntimeError(
            "Synthetic generation did not reach the minimum coverage after retries: "
            + ", ".join(shortfalls)
        )


if __name__ == "__main__":
    main()
