---
title: "Building Daniel OS: data, training, and strict evaluation"
permalink: /posts/daniel-os-lfm2/
date: 2026-07-17
last_modified_at: 2026-07-20
eyebrow: "FIELD NOTE / LOCAL AI"
dek: "How I separated personal facts from public knowledge, trained evidence-routing behavior into LFM2-350M, and evaluated a browser-native AI portfolio without hiding its limits."
read_time: true
comments: false
share: false
related: false
---

Daniel OS is a personal portfolio assistant that runs its generative model in the visitor's browser. It combines a verified profile index, a cited entity index, optional public retrieval, a personalized language model for conversational synthesis, browser speech recognition, and local speech output. The goal is to make the portfolio queryable without allowing personalization to distort ordinary technical knowledge.

This post separates what is implemented from what is planned. The LLM was fine-tuned and evaluated. The current speech layer uses browser APIs; it is not a custom-trained STT model or a clone of my voice. A reproducible STT data, fine-tuning, and evaluation pipeline now exists in the repository, but its checkpoint will not replace the browser API until real multi-speaker data and browser tests pass.

## Building a source-grounded profile

I began with claims from my CV, portfolio, LinkedIn profile, publications, GitHub, and Hugging Face account. Each claim was normalized into a small [profile JSON](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-profile.json), then assigned one of three provenance states:

| State | Meaning | Example |
| --- | --- | --- |
| Externally verified | Confirmed by an official paper, repository, or public API | [MobileHumanPose at CVPRW 2021](https://openaccess.thecvf.com/content/CVPR2021W/MAI/html/Choi_MobileHumanPose_Toward_Real-Time_3D_Human_Pose_Estimation_in_Mobile_Devices_CVPRW_2021_paper.html) |
| Public self-report | Published in my CV or LinkedIn, but not independently visible in a public technical artifact | Toss Bank project scope and internal metrics |
| Not verified | No reliable public source was found | Exact age, birthday, salary, or the claim that I performed jazz at Team ISLAND |

The external checks connect my KAIST education and Team ISLAND CTO history to the public CV, verify [ZZAZZ](https://www.venturesquare.net/821623) as Team ISLAND's mobile video-editing application, verify [ZERO](https://arxiv.org/abs/2507.04270) and MobileHumanPose from their publication pages, and query GitHub's public search API for 28 authored Transformers pull requests. The [provenance file](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-profile-sources.json) stores the retrieval date and URLs.

I do not infer personal facts from indirect signals. Graduation dates, for example, are not enough to establish that I am 29. That claim is deliberately represented as an unknown test case rather than an answer.

## Why personalized SFT was not enough

The first version assumed that every question was about me. That made the assistant very good at recognizing portfolio keywords, but it also created a systematic error: a general noun was pulled into my biography even when the visitor asked for an ordinary definition.

Two failures exposed different causes:

- "What is RT-DETR?" produced an invented description of "Daniel's work" involving few-shot learning and negative sampling. The small model had no retrieved definition, but the SFT distribution strongly rewarded Daniel-shaped answers.
- "Where is UIUC?" returned my KAIST, POSTECH, and UIUC education history. This answer never came from the model. A broad JavaScript keyword rule intercepted `uiuc` before generation.

Adding more memorized RT-DETR or UIUC answers would patch those nouns without fixing the system. The actual distinction is semantic: "What is X?" asks for X, while "What did Daniel do with X?" asks for a portfolio relation.

## Five evidence routes

The browser now classifies a prompt before generation:

```text
visitor question
    |
    +-- profile fact ----------> verified profile JSON
    +-- known entity ---------> cited local entity index
    +-- unknown factual noun -> public retrieval
    +-- private-person data --> local refusal
    +-- synthesis -----------> LFM2 with supplied evidence
```

The [knowledge router](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/js/knowledge-router.mjs) distinguishes definitions, profile relationships, neutral external lookups, and private-person requests. It also remembers the last portfolio entity so "What did he do with it?" can resolve a follow-up without turning every pronoun into a fact.

The [entity knowledge file](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-entity-knowledge.json) stores definitions separately from portfolio relations. RT-DETR cites its [original paper](https://arxiv.org/abs/2304.08069), ViTPose cites its [paper](https://arxiv.org/abs/2204.12484), UIUC cites the [university's location page](https://www.admissions.illinois.edu/about), SAM 2 cites [Meta AI](https://ai.meta.com/research/sam2/), and Molmo 2 cites [Ai2](https://allenai.org/molmo). Each record can therefore answer either side of the contrast without mixing them:

```json
{
  "name": "RT-DETR",
  "definition_en": "A cited definition of the detector itself.",
  "portfolio_relation_en": "Daniel's specific Transformers contribution.",
  "sources": [{"label": "RT-DETR paper", "url": "https://arxiv.org/abs/2304.08069"}]
}
```

Known entities are answered locally and immediately. An unseen neutral lookup uses Wikipedia's public API and displays the retrieved page as a citation. The browser never falls back from failed retrieval to model memory. This preserves a useful guarantee: a fluent sentence is not treated as evidence.

Public retrieval has a privacy cost because the lookup term leaves the device. Portfolio facts, known entities, private-data checks, model inference, and conversation history remain local; only an uncached general lookup is sent to Wikipedia. GitHub Pages cannot safely hide a commercial search API key, so unrestricted multi-source search would require a rate-limited server or edge proxy. The current fallback is intentionally smaller and inspectable.

## SFT dataset redesign

The combined supervised dataset now contains 296 conversations across five behaviors:

| Behavior | Records | Training target |
| --- | ---: | --- |
| `answer` | 177 | Answer only from selected profile evidence |
| `ground_external` | 12 | Define an entity only from supplied external evidence |
| `retrieve` | 15 | Request a public source instead of guessing |
| `unknown` | 58 | State that a Daniel-specific fact is not verified |
| `refuse` | 34 | Protect private data and reject unsafe or non-factual tasks |

The routing subset contains contrastive examples such as "What is RT-DETR?" versus "What did Daniel contribute to RT-DETR?", and "Where is UIUC?" versus "When did Daniel study at UIUC?" It includes English and Korean prompts, pronoun follow-ups, and neutral facts that must request retrieval instead of being refused.

A grounded external record carries evidence inside the training prompt:

```json
{
  "behavior": "ground_external",
  "context_keys": [],
  "evidence": {
    "entity": "ViTPose",
    "definition": "ViTPose uses a plain Vision Transformer backbone ...",
    "sources": ["https://arxiv.org/abs/2204.12484"]
  },
  "messages": [
    {"role": "user", "content": "What is ViTPose?"},
    {"role": "assistant", "content": "An evidence-grounded definition."}
  ]
}
```

When no evidence is supplied, the target is a small tool protocol rather than a fabricated answer:

```text
<search_public_knowledge>contrastive learning</search_public_knowledge>
```

The browser executes that request, retrieves evidence, replaces the control token with a cited answer, and never shows the token as the final response. Direct JavaScript routing handles common forms first; the fine-tuned tool behavior is a fallback for phrasings the deterministic router misses.

The validator checks duplicate prompts, role order, known profile keys, evidence presence, unsupported numeric claims, exact tool-call syntax, bilingual coverage, and minimum behavior counts. Profile and external evidence are separate fields so a number found in one cannot silently justify a claim in the other.

The trainer holds out examples inside each behavior and balances the effective stream differently from the original personalization-only run. Every minority behavior contributes at least 64 examples per epoch, while the 177 profile answers are not multiplied further. This reduces the prior that every question must produce a biography without discarding the broad profile corpus.

Most importantly, evaluation holds out whole entities, not just paraphrases. DINOv3 and DETA definitions do not appear in routing SFT. Their evaluation cases supply previously unseen evidence and test whether the model can synthesize it without inventing a relationship to me. Other cases remove evidence and require a search request for unseen entities such as CLIP, NeRF, and Carnegie Mellon University.

The published dataset layout is:

```text
sft/train.jsonl                  268 profile conversations
sft/routing.jsonl                28 routing conversations
behavior_eval/validation.jsonl   36 profile behavior checks
routing_eval/validation.jsonl     9 routing and entity-holdout checks
strict_test/test.jsonl           51 post-training tests
profile/                         profile, provenance, and entity knowledge
metrics/                         training loss and strict evaluation
```

## Model and training loss

I fine-tune [LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) with LoRA, merge the adapter into the base checkpoint, and export the merged weights as a symmetric Q4 ONNX graph. This is not one-bit fine-tuning. LoRA makes adaptation memory-efficient; Q4 is a separate deployment step that reduces browser download and inference memory.

The configuration uses LoRA rank 16, alpha 32, dropout 0.05, all linear layers as targets, a batch size of one, gradient accumulation of four, a peak learning rate of `2e-4`, a maximum sequence length of 1,152, and three epochs. Only assistant completion tokens contribute to the causal language-modeling objective:

```text
L = -(1 / N) sum[t in assistant tokens] log p(y_t | policy, profile, evidence, user, y_<t)
```

System policy, profile context, retrieved evidence, and visitor tokens are masked from the loss. The model learns the answer or routing behavior, not how to reproduce its input evidence.

![Daniel LFM2 train and validation loss]({{ '/assets/images/daniel-lfm2-loss.png' | relative_url }})

This chart and the committed [raw metrics](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-lfm2-training-metrics.json) describe the previous personalization-only checkpoint. Its validation loss moved from `0.469` at epoch 1 to `0.386` at epoch 2, then rose to `0.402` at epoch 3, so the trainer selected epoch 2. I keep that curve as a baseline rather than presenting it as evidence for the new routing dataset. The routing revision must produce its own finite validation curve and pass every behavioral gate before replacing the hosted Q4 model.

```text
profile SFT + routing SFT
        |
        v
LFM2-350M + rank-16 LoRA
        |
        v select minimum validation loss
merged Transformers checkpoint
        |
        v profile + routing + strict gates
symmetric Q4 ONNX export
        |
        v
Transformers.js Worker / WebGPU
```

## Evaluation contract

The training-time gate now contains 45 cases: 36 profile cases and nine routing cases. It reports profile answers, evidence-grounded definitions, retrieval decisions, unknown facts, refusals, and Korean behavior separately. The public strict set remains untouched by training and contains 51 cases: 31 profile answers, 10 unknown facts, nine refusals, and one general retrieval case.

- **Route accuracy:** definition, profile relation, retrieval, privacy, or refusal behavior is selected correctly.
- **Expected fact-group recall:** required semantic fact groups appear in the response.
- **Forbidden-claim avoidance:** planted false model names, metrics, personal facts, and unrelated claims do not appear.
- **Evidence support:** every factual answer can be traced to the selected profile or external evidence object.
- **Unknown claim leak rate:** a missing profile fact is not adopted from the question.
- **Refusal scope leak rate:** a refusal does not continue into the unsafe request.
- **Korean response rate:** Korean prompts receive Korean user-facing answers.
- **Strict pass rate:** behavior, evidence, forbidden-claim, and language requirements pass together.

A test accepts groups of valid phrases rather than one exact sentence. `fivefold`, `five times`, and `5x`, for example, express the same serving result. Forbidden terms test the opposite direction: a prompt suggesting a 10x speedup must not make the model repeat it.

The earlier checkpoint's strict results remain a baseline, not a claim about this revision. It showed that a small personalized model could learn strong abstention while still having weak compositional recall and poor Korean output. The new evaluation is designed to expose a different failure: whether the model can use unseen evidence without converting every entity into "Daniel's work." All generated answers are published so aggregate scores cannot hide a fluent hallucination.

The browser runtime has its own deterministic tests for the exact regression pairs. They verify RT-DETR definition versus contribution, ViTPose definition, UIUC location versus study history, entity pronoun follow-ups, private bank-account requests, unseen Wikipedia retrieval, and citation rendering. This makes the product gate broader than the model checkpoint gate: both the model behavior and the code that routes around it must be correct.

## What STT and TTS currently mean

English speech input currently uses the browser's speech-recognition interface. The final transcript is passed through the same grounded route as typed text. Speech output uses the browser's `speechSynthesis` interface and a voice installed by the browser or operating system.

There is therefore no custom STT checkpoint or personal TTS checkpoint in the current release. Browser vendors do not expose the model or objective behind their speech-recognition implementation. Calling the current output a WebGPU STT model or my trained voice would be inaccurate.

STT and TTS also require opposite data strategies. A personal TTS model should learn one consented target speaker: me. STT must recognize a visitor it has never heard before, so tuning it mostly on my voice would optimize the wrong problem. The [new STT pipeline](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/README-stt.md) requires many pseudonymous speakers and keeps every utterance from one speaker in exactly one of train, validation, or test. It also rejects cross-split recording-session and audio-hash leakage.

The STT manifest records the factors needed to diagnose generalization without storing a real name:

```json
{
  "utterance_id": "speaker_hash_session_utterance",
  "audio_path": "audio/example.wav",
  "transcript": "What did he build at Toss Bank?",
  "speaker_id": "pseudonymous_speaker_hash",
  "session_id": "pseudonymous_session_hash",
  "language": "en",
  "source": "consented",
  "domain": "portfolio",
  "environment": "quiet_mobile",
  "accent_group": "self_reported_coarse_group",
  "consent": "explicit-v1",
  "split": "train"
}
```

Audio is mono 16 kHz PCM WAV, 0.4-30 seconds. The intended mixture is primarily licensed multi-speaker English speech, plus consented phone, laptop, and headset recordings. The committed [capture prompts](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-stt-capture-prompts.jsonl) include natural variations and difficult names such as Hugging Face, Molmo2, ZZAZZ, ZERO, WebGPU, and Toss Bank. Corrected failures may enter an error-replay slice only after explicit opt-in; the fixed test recording itself never becomes a training sample.

The browser-sized target is `openai/whisper-tiny.en`, adapted with rank-16 LoRA on attention query and value projections. The 16 kHz waveform becomes a log-Mel spectrogram, and padded transcript tokens are ignored while the decoder minimizes sequence-to-sequence cross-entropy:

```text
L_ASR = -(1 / N) sum[t in transcript tokens] log p(y_t | log-Mel(audio), y_<t)
```

Training applies light gain, speed, and SNR perturbation together with Whisper SpecAugment. Those augmentations support, rather than replace, real diversity across speakers, accents, rooms, and microphones. A larger Distil-Whisper model can propose pseudo-labels for untranscribed consented audio, but uncertain labels and portfolio names still require human review. I chose the smaller deployment target because Transformers.js already supports Whisper ASR on WebGPU and the first-download and memory budget matter in a portfolio page.

The release gate is deliberately broader than one average WER. It reports micro WER, macro and worst-speaker WER, substitution/deletion/insertion counts, WER by domain, environment, and coarse self-reported accent group, recall of portfolio keywords, model-side latency, and real-time factor. A passed PyTorch checkpoint is only a candidate: an ONNX build must repeat the frozen suite in WebGPU and WASM and also pass download-size, peak-memory, microphone-lifecycle, first-load-latency, and browser real-time-factor checks.

There is no STT loss chart or WER result yet because no real audio corpus has been admitted to the pipeline. The trainer saves its raw log history, and the [plotter](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/plot_daniel_stt_metrics.py) refuses to draw unless both real train and validation loss points exist. This avoids turning a planned experiment into an apparent result.

Personal TTS remains a separate later stage. Its recordings should be only my explicitly consented voice, stored as session-separated mono PCM WAV at the sample rate required by the chosen implementation. A speaker-conditioned VITS or Piper-style candidate would report intelligibility through an independent ASR, speaker similarity through a held-out speaker encoder, and human listening scores. Its objective commonly combines text-to-acoustic reconstruction, duration or alignment, KL, and adversarial terms, but the exact loss belongs to the selected implementation rather than this untrained design.

Voice data is more sensitive than profile text. The [dataset publisher](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/publish_daniel_stt_dataset.py) creates a private repository by default and requires a separate `allow_publication` flag on every item before a public upload. Visitor audio remains ephemeral by default, incidental speakers are excluded, and the current public Hugging Face profile dataset contains no voice recording or visitor conversation.

## Browser runtime and reproducibility

The model runs in a module Web Worker so model download, ONNX session creation, and token generation do not block the interface. Recent Chromium browsers use WebGPU; unsupported environments fall back to WASM. The pinned Q4 external weight file is exactly 289,140,736 bytes, plus the graph, tokenizer, and configuration files, and it is cached after the first successful load.

WebGPU does not give each model independent hardware. A page can create multiple logical `GPUDevice` objects, but GPU memory and compute are machine-global resources shared with other workers, tabs, pages, and applications. More resident models therefore add weight and intermediate-buffer memory and compete for command execution. Under enough pressure an allocation can fail or the browser can lose a device. This follows the [WebGPU specification](https://gpuweb.github.io/gpuweb/) and its [design explainer](https://gpuweb.github.io/gpuweb/explainer/), rather than an assumption based on one fast development computer.

The deployed runtime now probes a WebGPU adapter without requesting an extra device, then assigns a conservative compatibility, low, balanced, or high tier. A software adapter, at most 4 GB of reported device memory, or at most four logical CPU cores disables eager loading. The Q4 LLM then loads only for a free-form request and is released after 90 seconds idle. No WebGPU adapter means an on-demand WASM fallback. These are hints, not a VRAM measurement: `navigator.deviceMemory` is coarse and optional, while adapter limits report legal buffer sizes rather than currently free memory.

Only one heavyweight model may be resident inside Daniel OS. When local STT and personal TTS are eventually promoted, the execution order will be `STT -> LLM -> TTS`, releasing one session before acquiring the next instead of keeping all three on the GPU. The current speech APIs use no WebGPU model, so today the Q4 LLM is the only GPU session. Separate tabs can still instantiate separate copies; the reproducible benchmark includes an explicit two-tab contention mode so that cost can be measured rather than hidden.

Q4 is the default for LFM2 because the first download and resident weights dominate on a portfolio page. It is not declared universally fastest: dequantization can make Q8 or FP16 faster on some GPUs. The [Transformers.js dtype guide](https://huggingface.co/docs/transformers.js/guides/dtypes) also warns that encoder-decoder models such as Whisper can be especially sensitive to quantization. The future STT gate therefore compares Q8 and a Q8-encoder/Q4-decoder build, with FP16 encoder experiments on `shader-f16` hardware; an all-Q4 build ships only if WER, worst-group WER, keyword recall, memory, and browser real-time factor all pass. Personal TTS will compare Q8 and FP16 against intelligibility, speaker-similarity, and listening tests. One-bit inference is not part of the current ONNX/Transformers.js path.

I also do not begin by writing custom WGSL kernels. ONNX Runtime already supplies WebGPU operators and recommends profiling, minimizing CPU/GPU transfers, and using I/O binding where recurrent tensors stay on the GPU. A supported export, ORT-format or reduced-operator build, and graph-level fusion come first. A custom kernel becomes reasonable only if the [ONNX Runtime Web profiler](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html) identifies one stable dominant unsupported or slow operator and the replacement passes correctness tests across GPU vendors.

The local 32 GB Mac handles source changes, mock interaction, and responsive browser checks. Existing LLM remote jobs perform training, Q4 export, CPU inference smoke testing, strict behavior evaluation, Hugging Face publication, and WebGPU browser checks. The STT workflow is wired to a remote A10G job, but it has not been dispatched because the required consented multi-speaker corpus does not yet exist.

The [runtime policy and benchmark protocol](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/README-webgpu.md) records the device matrix and commands. It deliberately labels a 4 GB/four-core browser override as policy emulation, not a low-end speed result. Actual first-load, warm-generation, and optional two-tab timings must be collected on the development Mac and real 4 GB and 8 GB integrated-GPU devices before a speech model is promoted.

The first controlled development-Mac audit used visible Chromium on Apple Metal 3 with the same prompt and cleared context. Q4 initialization took 30.0 seconds and one warm generation took 3.22 seconds inside the worker. With two independently initialized tabs generating simultaneously, the same completion took 6.58 and 6.84 seconds, or 2.04x and 2.12x the single-session time. That is a single-machine audit rather than a universal benchmark, but it confirms that logical sessions contend and supports sequential residency. Headless Chromium exposed SwiftShader and was conservatively placed in the low, on-demand tier.

The complete implementation is reproducible from the repository:

- [Training and merge script](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/train_daniel_lfm2.py)
- [Knowledge router and public-retrieval fallback](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/js/knowledge-router.mjs)
- [Cited portfolio entity index](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-entity-knowledge.json)
- [Routing SFT and entity-held-out evaluation](https://github.com/SangbumChoi/sangbumchoi.github.io/tree/master/assets/data)
- [Dataset validators](https://github.com/SangbumChoi/sangbumchoi.github.io/tree/master/scripts)
- [Strict evaluator](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/evaluate_daniel_lfm2_test.py)
- [Loss plotting script](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/plot_daniel_lfm2_metrics.py)
- [Merged LFM2 checkpoint](https://huggingface.co/danelcsb/daniel-lfm2-350m)
- [Q4 browser model on Hugging Face](https://huggingface.co/danelcsb/daniel-lfm2-350m-ONNX)
- [Q4 browser model release](https://github.com/SangbumChoi/sangbumchoi.github.io/releases/tag/daniel-lfm2-onnx-v1)
- [Browser worker](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/js/lfm-worker.js)
- [Adaptive WebGPU runtime policy](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/js/runtime-policy.mjs)
- [WebGPU benchmark protocol](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/README-webgpu.md)
- [Speaker-disjoint STT preparer](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/prepare_daniel_stt_dataset.py)
- [Whisper LoRA trainer and release gate](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/train_daniel_stt.py)
- [Grouped STT evaluator](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/score_daniel_stt_predictions.py)

The principle is simple: fine-tune behavior, retrieve knowledge, keep personal and general evidence separate, publish tests that expose fluent mistakes, and describe every untrained component honestly.
