# Daniel OS LFM2 pipeline

The website routes each prompt between a verified profile index, a curated
entity index, public retrieval, a privacy boundary, and LFM2-350M synthesis.
Definitions and Daniel-specific claims are kept as separate evidence types.
Wikipedia is a fallback only for neutral definitions not present in the local
entity index; the answer includes its source, and a retrieval failure never
falls through to unsupported model memory. Production loads the Q4 model in a
Web Worker; `_config.dev.yml` disables eager loading for local UI work.

## Train and evaluate

The combined supervised dataset separates five behaviors: profile answers,
definitions grounded in supplied external evidence, public-search tool requests,
missing Daniel-specific facts, and privacy or safety refusals. Contrastive pairs
separate questions such as "What is RT-DETR?" from "What did Daniel contribute
to RT-DETR?" DINOv3 and DETA are held out by entity for the evidence-synthesis
gate. The first evaluated run repeated small behavior groups to at least 64
examples per epoch. A later audit found that 145 of 431 effective slots per
epoch were repeats, including a 6.4x repeat factor for evidence-grounded
definitions. The v3 generator replaces cyclic oversampling with distinct
prompts while keeping curated answers fixed. Loss is computed only on the
assistant completion.

```sh
python3 scripts/validate_daniel_lfm2_data.py
```

```sh
uv run --python 3.12 \
  --with 'torch>=2.6' \
  --with 'transformers>=4.55' \
  --with 'trl>=0.24' \
  --with 'peft>=0.17' \
  --with 'datasets>=3.0' \
  scripts/train_daniel_lfm2.py

uv run --python 3.12 \
  --with 'torch>=2.6' \
  --with 'transformers>=4.55' \
  scripts/evaluate_daniel_lfm2.py artifacts/daniel-lfm2-350m/merged
```

If training finishes but merging is interrupted, resume only that step with:

```sh
python3 scripts/merge_daniel_lfm2.py
```

The training script follows Liquid AI's recommended LFM LoRA rank and scaling
(rank 16, alpha 32). It targets every linear layer because LFM2 is a hybrid
attention/convolution architecture and this narrow identity task must adapt
both paths. It writes both the adapter
and a merged Transformers checkpoint. Generated artifacts are intentionally
excluded from Git.

## Colab GPU data generation and ablation

Open
[`notebooks/daniel_lfm2_gpu_retraining.ipynb`](../notebooks/daniel_lfm2_gpu_retraining.ipynb)
in Colab with a GPU runtime. The notebook performs five stages:

1. reconstruct the old sampling mixture and loss diagnostics;
2. refresh source availability without automatically trusting scraped text;
3. use a 4-bit Qwen3-4B teacher to paraphrase prompts only;
4. build scenario-family-disjoint train and validation sets and compare
   LFM2-350M with LFM2.5-350M at `5e-5`, `1e-4`, and `2e-4`;
5. rank candidates against the currently deployed frozen strict-test baseline.

The teacher never writes a target answer. Every answer, evidence object,
profile key, retrieval token, and behavior comes from curated seed data.
Generated prompts are filtered for language, length, exact duplicates, and
token overlap. Every variation from one seed remains in the same split.

Run the local, model-free parts before allocating a GPU:

```sh
python3 scripts/analyze_daniel_lfm2_data.py
python3 scripts/generate_daniel_lfm2_synthetic.py --seed-only
```

The prepared-data trainer uses a balanced validation pool for checkpoint
selection, also logs full per-behavior loss, evaluates every configured number
of optimizer steps, and supports early stopping:

```sh
python3 scripts/train_daniel_lfm2.py \
  --model LiquidAI/LFM2.5-350M \
  --prepared-train artifacts/daniel-lfm2-v3/train.jsonl \
  --prepared-validation artifacts/daniel-lfm2-v3/validation.jsonl \
  --batch-size 8 --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 --eval-steps 25 --early-stopping-patience 2
```

## Train and publish without using local memory

The GitHub Actions CPU workflow is manual because the first three-epoch run took
about four hours and should not be repeated for every data-pipeline edit. The
Colab notebook is the primary experiment path. Dispatch
`.github/workflows/train-publish-daniel-lfm2.yml` only for the legacy release
path after reviewing a candidate. The workflow:

1. validates every training and held-out record against the profile schema;
2. trains and merges LFM2-350M on a GitHub-hosted runner;
3. requires separate pass thresholds for profile answers, grounded definitions,
   retrieval decisions, missing facts, refusals, Korean responses, and the public strict test;
4. publishes the evaluated FP16 checkpoint and matching SFT dataset to Hugging
   Face, and publishes the source under `daniel-lfm2-source-v2`;
5. exports and smoke-tests symmetric Q4 ONNX before updating the public ONNX
   model, `model-assets` backup branch, and `daniel-lfm2-onnx-v1` release.

This keeps model training and export isolated from the 32 GB development Mac.
After the remote job passes, pin the emitted Hugging Face model revision and
bump the browser cache version with
`python3 scripts/pin_daniel_lfm2_model.py <revision>`.

The merged model gate requires at least 70% overall, 60% profile answers and
grounded definitions, two-thirds of retrieval and missing-fact decisions, and
80% of privacy or safety refusals. Nine routing cases add held-out-entity
evidence, definition-versus-contribution contrasts, and unseen retrieval terms.
The separate 51-case strict gate checks privacy, chronology, Korean output,
hallucination traps, and genuine multi-turn follow-ups before ONNX export.

## Export only

The export script pins Liquid AI's official LiquidONNX revision, creates a
WebGPU-compatible symmetric Q4 graph, runs a CPU smoke test, and prepares the
complete Transformers.js model directory. To re-export the current source
model without retraining, run it on Hugging Face Jobs so the local machine only
submits and monitors the task:

```sh
hf jobs uv run \
  --flavor cpu-upgrade \
  --timeout 2h \
  --secrets HF_TOKEN \
  --detach \
  scripts/export_daniel_lfm2_onnx.py --upload-hf
```

When Hugging Face CLI credentials are unavailable, dispatch
`.github/workflows/export-daniel-lfm2.yml`. It performs the same export on a
GitHub-hosted runner, publishes the browser directory to Hugging Face, and
publishes flattened backup files under the `daniel-lfm2-onnx-v1` release tag.
It also preserves the original structure on the Git LFS-backed `model-assets`
branch as a second backup.

After the job succeeds, pin the public `danelcsb/daniel-lfm2-350m-ONNX` commit
in `MODEL_REVISION` inside `assets/js/lfm-worker.js`. The worker uses Hugging
Face's CORS-enabled resolve endpoint and an immutable commit so the browser
cache and every model file stay tied to the evaluated export. The GitHub release
remains a downloadable backup rather than the browser's primary model origin.

The browser still retrieves JSON evidence before generation. The model learns
how to apply evidence and can emit
`<search_public_knowledge>TERM</search_public_knowledge>` when evidence is absent.
The browser executes that request and replaces the control token with a cited
answer. Source data remains authoritative for dates, metrics, definitions,
private-fact absence, and links.

As an alternative to GitHub Actions, submit the single-file trainer to Hugging
Face Jobs with the dataset, held-out evaluation cases, and profile pinned to
the same Git revision. The trainer evaluates factual answers, missing facts,
and unrelated request refusals before it uploads the merged model:

```sh
hf jobs uv run \
  --flavor a10g-small \
  --timeout 2h \
  --secrets HF_TOKEN \
  --detach \
  scripts/train_daniel_lfm2.py \
  --dataset-url https://raw.githubusercontent.com/SangbumChoi/sangbumchoi.github.io/<revision>/assets/data/daniel-lfm2-sft.jsonl \
  --routing-dataset-url https://raw.githubusercontent.com/SangbumChoi/sangbumchoi.github.io/<revision>/assets/data/daniel-lfm2-routing-sft.jsonl \
  --profile-url https://raw.githubusercontent.com/SangbumChoi/sangbumchoi.github.io/<revision>/assets/data/daniel-profile.json \
  --eval-url https://raw.githubusercontent.com/SangbumChoi/sangbumchoi.github.io/<revision>/assets/data/daniel-lfm2-eval.jsonl \
  --routing-eval-url https://raw.githubusercontent.com/SangbumChoi/sangbumchoi.github.io/<revision>/assets/data/daniel-lfm2-routing-eval.jsonl \
  --training-revision <revision> \
  --push-to-hub
```

## Remote browser verification

Dispatch `.github/workflows/test-daniel-lfm2-webgpu.yml` after deployment. The
workflow waits for the matching Pages asset version, enables SwiftShader WebGPU
in headless Chromium, confirms automatic WebGPU model loading begins, observes
a personalized model request, and verifies ranged access to the Q4 weights.
Full graph inference is covered by the export job's CPU smoke test; the browser
check intentionally avoids compiling the entire 294 MB graph in software
WebGPU, which is not representative of a visitor's hardware GPU.
