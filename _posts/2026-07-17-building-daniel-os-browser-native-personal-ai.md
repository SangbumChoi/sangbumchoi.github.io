---
title: "Building Daniel OS: data, training, and strict evaluation"
permalink: /posts/daniel-os-lfm2/
date: 2026-07-17
eyebrow: "FIELD NOTE / LOCAL AI"
dek: "How I built a source-grounded personal dataset, fine-tuned LFM2-350M, measured its boundaries, and deployed speech and four-bit inference in the browser."
read_time: true
comments: false
share: false
related: false
---

Daniel OS is a personal portfolio assistant that runs its generative model in the visitor's browser. It combines a verified profile index for exact facts, a personalized language model for conversational synthesis, browser speech recognition, and local speech output. The goal is to make the portfolio queryable while keeping visitor conversations on the device.

This post separates what is implemented from what is planned. The LLM was fine-tuned and evaluated. The current speech layer uses browser APIs; it is not a custom-trained STT model or a clone of my voice.

## Building a source-grounded profile

I began with claims from my CV, portfolio, LinkedIn profile, publications, GitHub, and Hugging Face account. Each claim was normalized into a small [profile JSON](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-profile.json), then assigned one of three provenance states:

| State | Meaning | Example |
| --- | --- | --- |
| Externally verified | Confirmed by an official paper, repository, or public API | [MobileHumanPose at CVPRW 2021](https://openaccess.thecvf.com/content/CVPR2021W/MAI/html/Choi_MobileHumanPose_Toward_Real-Time_3D_Human_Pose_Estimation_in_Mobile_Devices_CVPRW_2021_paper.html) |
| Public self-report | Published in my CV or LinkedIn, but not independently visible in a public technical artifact | Toss Bank project scope and internal metrics |
| Not verified | No reliable public source was found | Exact age, birthday, salary, or the claim that I performed jazz at Team ISLAND |

The external checks connect my KAIST education and Team ISLAND CTO history to the public CV, verify [ZERO](https://arxiv.org/abs/2507.04270) and MobileHumanPose from their publication pages, and query GitHub's public search API for 28 authored Transformers pull requests. The [provenance file](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-profile-sources.json) stores the retrieval date and URLs.

I do not infer personal facts from indirect signals. Graduation dates, for example, are not enough to establish that I am 29. That claim is deliberately represented as an unknown test case rather than an answer.

## LLM dataset design

The supervised dataset contains 75 one-turn conversations: 48 grounded answers, 9 missing-fact responses, and 18 out-of-scope refusals. These behaviors are intentionally different:

- `answer`: the supplied profile context contains the requested fact.
- `unknown`: the question is about me, but the context does not verify the answer.
- `refuse`: the request is unrelated, unsafe, asks for impersonation, or tries to override the portfolio boundary.

Each training record uses this schema:

```json
{
  "id": "toss_04",
  "behavior": "answer",
  "context_keys": ["current_work"],
  "messages": [
    {"role": "user", "content": "How did Daniel evaluate the document pipeline?"},
    {"role": "assistant", "content": "He designed stage-level evaluation ..."}
  ],
  "expected_terms": ["61 percent", "exact-match"]
}
```

`context_keys` selects only the relevant portion of the profile. Numeric statements in an answer must also occur in that context. A validator rejects duplicate prompts, unknown context keys, unsupported numbers, malformed role pairs, and insufficient coverage of any behavior.

For loss evaluation, the trainer takes a stratified five-conversation holdout: two `answer`, one `unknown`, and two `refuse` examples. After that holdout, minority behaviors are modestly repeated in the effective training stream so the model sees enough boundary examples without making refusal its dominant behavior. A separate 18-prompt validation gate is never used as a training target.

The new public test set adds 39 harder prompts: 23 factual answers, 7 unknown facts, and 9 refusals. Nine are Korean. It includes cross-section synthesis, exact metric traps, privacy questions, unsupported internal model names, prompt injection, and the KAIST plus Team ISLAND plus ZERO career connection. It is a post-training test split and is not fed back into fine-tuning.

All three configurations are published in the [Daniel OS dataset on Hugging Face](https://huggingface.co/datasets/danelcsb/daniel-os-profile-sft):

```text
sft/train.jsonl                 75 conversations
behavior_eval/validation.jsonl  18 held-out behavior checks
strict_test/test.jsonl          39 post-training tests
profile/                        grounded facts and provenance
metrics/                        training loss and strict evaluation
```

## Model and training loss

I fine-tuned [LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) with LoRA, merged the adapter into the base checkpoint, and exported the merged weights as a symmetric Q4 ONNX graph. This is not one-bit fine-tuning. LoRA makes adaptation memory-efficient; Q4 is a separate deployment step that reduces browser download and inference memory.

The configuration uses LoRA rank 16, alpha 32, dropout 0.05, all linear layers as targets, a batch size of one, gradient accumulation of four, a peak learning rate of `2e-4`, a maximum sequence length of 768, and three epochs. Only assistant completion tokens contribute to the causal language-modeling objective:

```text
L = -(1 / N) sum[t in assistant tokens] log p(y_t | system, context, user, y_<t)
```

Prompt and profile-context tokens are masked from this cross-entropy loss. The optimizer therefore adapts the LoRA weights to produce the desired answer behavior rather than spending loss on reproducing the input context.

![Daniel LFM2 train and validation loss]({{ '/assets/images/daniel-lfm2-loss.png' | relative_url }})

The chart is generated from the committed [raw training metrics](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-lfm2-training-metrics.json), not reconstructed estimates. Validation loss moved from `0.469` at epoch 1 to `0.386` at epoch 2, then rose slightly to `0.402` at epoch 3. Because the trainer selects the minimum validation loss, the merged model uses the epoch 2 checkpoint. That small final rise is worth showing: it is evidence that another epoch did not improve generalization.

```text
Verified conversations
        |
        v
LFM2-350M + LoRA adapter
        |
        v merge best checkpoint
Personalized Transformers checkpoint
        |
        v symmetric Q4 export
ONNX graph + external weight data
        |
        v
Transformers.js Web Worker / WebGPU
```

## A stricter evaluation contract

The original 18-prompt gate scored 77.8% overall: 60% for grounded profile answers, 100% for missing facts, and 100% for refusals. That check is useful but too small to describe production behavior by itself.

The 39-case strict test records several metrics instead of reporting only one average:

- **Expected fact-group recall:** how many required semantic fact groups appear in the answer.
- **Behavior pass rate:** all expected groups are present and no forbidden claim appears.
- **Hallucination guard rate:** the answer avoids planted false numbers, model names, personal facts, and unrelated content.
- **Unknown claim leak rate:** an unknown-fact answer does not adopt the unsupported claim from the question.
- **Refusal scope leak rate:** a refusal does not go on to answer the unrelated request.
- **Korean response rate:** Korean prompts receive a response containing Korean, in addition to passing the behavior checks.
- **Strict pass rate:** behavior, forbidden-claim, and language requirements all pass together.

A test case accepts groups of valid phrases rather than requiring one exact reference sentence. For example, `fivefold`, `five times`, and `5x` can express the same serving result. Forbidden terms test the opposite direction: a question that suggests a 10x speedup must not cause the model to repeat it.

The browser adds another layer. Common profile questions are routed to the deterministic JSON index before generation, and a production test asks six questions about Toss Bank, multimodal training, open source, education, and publications. It verifies both the source route and required facts. This keeps exact dates and metrics reliable even when a small language model phrases a synthesis imperfectly.

## What STT and TTS currently mean

English speech input currently uses the browser's speech-recognition interface. The final transcript is passed through the same grounded route as typed text. Speech output uses the browser's `speechSynthesis` interface and a voice installed by the browser or operating system.

There is therefore no custom STT dataset, STT checkpoint, STT training loss, personal TTS dataset, or voice-cloning loss in the current release. Browser vendors do not expose the model or objective behind their speech-recognition implementation. Calling the current output my trained voice would be inaccurate.

For a future consented personal voice, I would store recordings as mono PCM WAV and keep session boundaries so that train and test audio from the same recording session cannot leak across splits. A minimal manifest would be:

```json
{
  "utterance_id": "daniel_en_0001",
  "split": "train",
  "audio_path": "audio/daniel_en_0001.wav",
  "transcript": "Original transcript",
  "normalized_text": "Normalized transcript",
  "language": "en",
  "sample_rate_hz": 24000,
  "duration_seconds": 4.2,
  "recording_session": "session_01",
  "consent_version": "v1",
  "sha256": "..."
}
```

A practical local STT candidate would be an English Whisper-family checkpoint trained with sequence-to-sequence token cross-entropy and evaluated with word error rate. A lightweight personal TTS candidate would use a speaker-conditioned VITS or Piper-style model and report intelligibility with ASR word error rate, speaker similarity with an independently held-out speaker encoder, and human listening scores. Its exact loss depends on the selected implementation, commonly combining text-to-acoustic reconstruction, duration or alignment, KL, and adversarial terms. These are design candidates, not results claimed by the current site.

Voice data is more sensitive than profile text. It should require explicit consent, exclude incidental speakers, retain checksums and recording-session metadata, and never be published by default. The current Hugging Face dataset contains no voice recording or visitor conversation.

## Browser runtime and reproducibility

The model runs in a module Web Worker so model download, ONNX session creation, and token generation do not block the interface. Recent Chromium browsers use WebGPU; unsupported environments fall back to WASM. The browser artifact is about 294 MB, pinned to an immutable Git LFS revision, and cached after the first successful load.

The local 32 GB Mac handles source changes, mock interaction, and responsive browser checks. Remote jobs perform training, Q4 export, CPU inference smoke testing, strict behavior evaluation, Hugging Face publication, and WebGPU browser checks.

The complete implementation is reproducible from the repository:

- [Training and merge script](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/train_daniel_lfm2.py)
- [Dataset validators](https://github.com/SangbumChoi/sangbumchoi.github.io/tree/master/scripts)
- [Strict evaluator](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/evaluate_daniel_lfm2_test.py)
- [Loss plotting script](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/plot_daniel_lfm2_metrics.py)
- [Merged LFM2 checkpoint](https://huggingface.co/danelcsb/daniel-lfm2-350m)
- [Q4 browser model release](https://github.com/SangbumChoi/sangbumchoi.github.io/releases/tag/daniel-lfm2-onnx-v1)
- [Browser worker](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/js/lfm-worker.js)

The principle is simple: train the model to communicate within a narrow scope, keep exact facts in a source-grounded layer, publish tests that can expose failure, and describe every untrained component honestly.
