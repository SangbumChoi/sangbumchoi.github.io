# Daniel OS LFM2 pipeline

The website uses a verified profile index for factual questions and
automatically loads LFM2-350M for free-form local generation. Production loads
the Q4 model in a Web Worker on entry; `_config.dev.yml` disables that automatic
load so responsive UI work does not consume local memory. The files in this
directory create a genuinely personalized checkpoint before it is exported for
WebGPU.

## Train and evaluate

The supervised dataset separates three behaviors: answers grounded in selected
profile sections, explicit statements that a Daniel-related fact is absent,
and refusals for requests outside the portfolio scope. Validate the source data
before starting a job:

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

## Train and publish without using local memory

Push changes to the dataset, profile, trainer, validator, exporter, or
`.github/workflows/train-publish-daniel-lfm2.yml` on `master`. The workflow:

1. validates every training and held-out record against the profile schema;
2. trains and merges LFM2-350M on a GitHub-hosted runner;
3. requires separate pass thresholds for verified answers, missing facts, and
   out-of-scope refusals;
4. publishes the evaluated FP16 checkpoint under the
   `daniel-lfm2-source-v2` GitHub release;
5. exports and smoke-tests symmetric Q4 ONNX before updating the
   `model-assets` branch and `daniel-lfm2-onnx-v1` release.

This keeps model training and export isolated from the 32 GB development Mac.
The website's immutable model revision is updated only after the remote job
passes.

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
GitHub-hosted runner and publishes the flattened model files under the
`daniel-lfm2-onnx-v1` release tag. It also publishes the original directory
structure to the Git LFS-backed `model-assets` branch so the browser worker can
load the files through GitHub's CORS-enabled media host.

After the job succeeds, pin the `model-assets` branch commit in
`MODEL_REVISION` inside `assets/js/lfm-worker.js`. The worker uses an immutable
`media.githubusercontent.com` URL template so all model files are CORS-enabled
and the browser cache is tied to the exact export revision.

The deterministic profile index must remain in front of generation. A small
model checkpoint is useful for tone and narrow profile synthesis, but it is not
a substitute for grounding when exact dates, metrics, and links matter.

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
  --profile-url https://raw.githubusercontent.com/SangbumChoi/sangbumchoi.github.io/<revision>/assets/data/daniel-profile.json \
  --eval-url https://raw.githubusercontent.com/SangbumChoi/sangbumchoi.github.io/<revision>/assets/data/daniel-lfm2-eval.jsonl \
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
