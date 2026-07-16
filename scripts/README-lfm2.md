# Daniel OS LFM2 pipeline

The website uses a verified profile index for factual questions and
automatically loads LFM2-350M for free-form local generation. Production loads
the Q4 model in a Web Worker on entry; `_config.dev.yml` disables that automatic
load so responsive UI work does not consume local memory. The files in this
directory create a genuinely personalized checkpoint before it is exported for
WebGPU.

## Train and evaluate

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

## Export for the browser without using local memory

The export script pins Liquid AI's official LiquidONNX revision, creates a
WebGPU-compatible symmetric Q4 graph, runs a CPU smoke test, and uploads the
complete Transformers.js model directory. Run it on Hugging Face Jobs so the
local machine only submits and monitors the task:

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

## Remote browser verification

Dispatch `.github/workflows/test-daniel-lfm2-webgpu.yml` after deployment. The
workflow enables SwiftShader WebGPU in headless Chromium, confirms that the
personalized Q4 model reaches `webgpu / private`, and runs one free-form
generation. This keeps the 294 MB model download and runtime memory off the
local development machine.
