---
title: "Building Daniel OS: a browser-native personal AI"
permalink: /posts/daniel-os-lfm2/
date: 2026-07-17
eyebrow: "FIELD NOTE / LOCAL AI"
dek: "How I fine-tuned LFM2-350M, exported a four-bit WebGPU model, grounded it in verified profile data, and kept the runtime private to the browser."
read_time: true
comments: false
share: false
related: false
---

Daniel OS is a personal portfolio assistant that runs its generative model in the visitor's browser. It combines a verified profile index for facts, a personalized language model for conversational synthesis, browser speech recognition, and local speech output. The goal is not to put a chatbot beside a resume. The goal is to make the portfolio itself queryable while keeping private conversations on the visitor's device.

## The model is LoRA-tuned and served at four bits

The precise description matters. Daniel OS is not a one-bit fine-tuned model. I fine-tuned [LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) with LoRA, merged the adapter into the base checkpoint, and then exported the merged weights as a symmetric Q4 ONNX graph. Training and serving precision are separate decisions: LoRA makes adaptation memory-efficient, while Q4 reduces download size and inference memory for deployment.

The training configuration uses rank 16, alpha 32, a batch size of one, and LoRA targets across the linear layers of LFM2's hybrid attention and convolution architecture. The verified dataset contains 75 curated conversations: 48 grounded profile answers, 9 explicit missing-fact answers, and 18 refusals for unrelated requests. Training preserves a larger share of factual answers, maintains minimum coverage for both boundary behaviors, and computes loss only on assistant completions. It does not make the model the database of record.

```text
Verified conversations
        ↓
LFM2-350M + LoRA adapter
        ↓ merge
Personalized Transformers checkpoint
        ↓ symmetric Q4 export
ONNX graph + external weight data
        ↓
Transformers.js Web Worker / WebGPU
```

The merged checkpoint is published as [danelcsb/daniel-lfm2-350m](https://huggingface.co/danelcsb/daniel-lfm2-350m) and in a reproducible [GitHub source release](https://github.com/SangbumChoi/sangbumchoi.github.io/releases/tag/daniel-lfm2-source-v2). On 18 held-out prompts it scored 77.8% overall: 60% on profile synthesis, 100% on missing facts, and 100% on safe refusals. The browser artifact is approximately 294 MB and is pinned to an immutable Git LFS revision. After the first visit, normal browser caching prevents the model from being fetched again unless the revision changes.

## Grounding before generation

A 350M-parameter model is useful for a focused conversational interface, but it should not invent dates, metrics, publication titles, or links. Daniel OS therefore routes recognizable profile questions through a small verified JSON index. Toss Bank work, multimodal training, publications, education, and open-source contributions are answered directly from that source. A production browser test submits six representative questions and verifies their source and required facts. Free-form questions use the local model with only the relevant verified context inserted into the system prompt.

This split also improves perceived latency. Visitors can ask common questions while the Q4 model downloads in the background. Once the WebGPU session is ready, the same interface can synthesize less structured answers without sending the conversation to an application server.

## Browser runtime and fallback

The model runs inside a module Web Worker so downloading, ONNX session creation, and token generation do not block the interface. A recent Chromium browser uses WebGPU; unsupported environments fall back to WASM. Progress events update the same compact runtime indicator used by the interface, and model errors leave deterministic profile answers available.

The website begins loading the personalized model automatically. This is deliberate for the current experience, although it trades initial bandwidth for immediate model availability. The model revision, quantization type, and approximate size are visible before loading completes.

## Speech input and output

English speech input uses the browser's speech-recognition interface and sends the final transcript through the same grounded question path as typed input. Speech output currently uses voices installed in the browser or operating system. It is not presented as a clone of my voice.

A real personal voice requires a clean, consented recording set, text alignment, a speaker-conditioned TTS model, and listening tests for intelligibility and identity similarity. No suitable voice recording is stored in this repository, so substituting a generic voice and labeling it as mine would be inaccurate. The speech boundary is intentionally separate so a validated personal voice asset can replace the browser voice later.

## Keeping development within 32 GB

The local Mac is used for source changes, mock interaction, Jekyll builds, and responsive browser checks. Memory-heavy conversion and runtime tests are delegated to remote jobs. The export pipeline pins Liquid AI's ONNX tooling, generates symmetric Q4, runs a CPU inference smoke test, and publishes immutable artifacts. A GitHub Actions browser check separately validates WebGPU availability, automatic loading, worker asset requests, and ranged access to the weights without compiling the complete graph in software WebGPU.

The complete implementation is reproducible from the repository:

- [Training and merge script](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/train_daniel_lfm2.py)
- [Held-out behavioral evaluation](https://github.com/SangbumChoi/sangbumchoi.github.io/releases/download/daniel-lfm2-source-v2/daniel-lfm2-evaluation.json)
- [Q4 ONNX export script](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/export_daniel_lfm2_onnx.py)
- [Browser worker](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/js/lfm-worker.js)
- [Q4 model release](https://github.com/SangbumChoi/sangbumchoi.github.io/releases/tag/daniel-lfm2-onnx-v1)

The important design choice is modest: use the language model where language modeling helps, and keep verified facts, privacy boundaries, and deployment behavior explicit everywhere else.
