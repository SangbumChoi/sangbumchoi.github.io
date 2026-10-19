---
title: "Building Daniel OS: data, training, and strict evaluation"
title_ko: "Daniel OS 만들기: 데이터, 학습, 그리고 엄격한 evaluation"
permalink: /posts/daniel-os-lfm2/
date: 2026-07-17
last_modified_at: 2026-07-21
eyebrow: "FIELD NOTE / LOCAL AI"
dek: "How I separated personal facts from public knowledge, trained evidence-routing behavior into LFM2-350M, and evaluated a browser-native AI portfolio without hiding its limits."
dek_ko: "개인 정보와 공개 지식을 어떻게 분리했는지, LFM2-350M에 근거 routing 행동을 어떻게 학습시켰는지, 그리고 한계를 숨기지 않고 브라우저 네이티브 AI 포트폴리오를 어떻게 평가했는지 정리합니다."
read_time: true
comments: false
share: false
related: false
---

<div class="lang-block" lang="en" markdown="1">

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

The first trainer held out examples inside each behavior and balanced the effective stream differently from the original personalization-only run. Every minority behavior contributed at least 64 examples per epoch, while the 177 profile answers were not multiplied further. This reduced the prior that every question must produce a biography without discarding the broad profile corpus, but the later loss audit below found that this implementation repeated too many identical minority examples.

Most importantly, evaluation holds out evidence conditions, not just paraphrases. DINOv3 and DETA appear in routing SFT only as no-evidence requests that must trigger public retrieval; their definitions are withheld until evaluation. The evaluation cases then supply that unseen evidence and test whether the model switches from search to grounded synthesis without inventing a relationship to me. CLIP, NeRF, and Carnegie Mellon University provide a separate lexical holdout: none appears in SFT, and each must trigger a search request when no evidence is supplied.

The published dataset layout is:

```text
sft/train.jsonl                  268 profile conversations
sft/routing.jsonl                28 routing conversations
behavior_eval/validation.jsonl   36 profile behavior checks
routing_eval/validation.jsonl     9 routing and evidence-holdout checks
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

This chart and the committed [raw metrics](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-lfm2-training-metrics.json) come from routing revision `e54fa04`. Validation loss moved from `0.754` at epoch 1 to `0.536` at epoch 2, then rose to `0.620` at epoch 3, so the trainer restored epoch 2 rather than publishing the more overfit final epoch. Reported average training loss was `0.490`. The CPU GitHub runner took 15,045 seconds, about 4 hours 11 minutes, for the balanced three-epoch stream. The merged checkpoint then passed both behavior gates and the untouched strict set before its symmetric-Q4 ONNX export passed a CPU inference smoke test.

### Why the validation curve is not smooth

The curve has only three validation points, and each point was computed from ten examples: at most two from each behavior. The [new diagnostic script](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/analyze_daniel_lfm2_data.py) also reconstructs the effective sampler. Of 431 training slots per epoch, 145, or `33.6%`, were repeated slots. Evidence-grounded definitions were especially concentrated: ten unique training records were cycled to 64 slots, a `6.4x` repeat factor. Training loss continued to fall while the final validation loss rose `15.7%` above the epoch-2 minimum.

That is evidence of post-epoch-2 overfitting, but it is not enough to claim a precisely measured generalization curve. A ten-example validation average has high variance, the behavior groups have very different response lengths, and the training mixture differs sharply from the validation mixture. More generated data alone would not repair those measurement problems.

The retry is based on a narrower reading of synthetic-data research. [Self-Instruct](https://aclanthology.org/2023.acl-long.754/) generates and filters diverse instructions rather than duplicating seeds. [LIMA](https://papers.neurips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html) shows that a small carefully curated alignment set can outperform a much larger noisy one. [AlpaGasus](https://arxiv.org/abs/2307.08701) likewise reports better results after filtering 52,000 examples to 9,000 higher-quality examples, and [DEITA](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6091f2bb355e960600f62566ac0e2862-Abstract-Conference.html) frames selection around quality, complexity, and diversity. The practical conclusion here is to expand coverage, not duplicate frequency.

The [Colab GPU notebook](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/notebooks/daniel_lfm2_gpu_retraining.ipynb) therefore uses a 4-bit Qwen3-4B teacher to generate **questions only**. Target answers remain curated, so a teacher hallucination cannot silently become a profile fact. New entity-definition seeds come from the cited entity index, while unrelated public topics target the search-tool protocol rather than memorized world facts. Exact duplicates, near duplicates, wrong-language prompts, and meta-instructions are rejected. Every variation from one seed or scenario family remains entirely in train or validation.

The target experiment uses roughly 1,820 training records and 300 validation records across the five behaviors, with equal-size behavior slices for checkpoint selection and full per-behavior loss diagnostics. It evaluates every 25 optimizer steps, applies two-evaluation early stopping, and sweeps `5e-5`, `1e-4`, and `2e-4` on both LFM2-350M and [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M). This is an experiment plan, not a reported result: the deployed checkpoint and its metrics remain unchanged until a Colab run meets or exceeds every frozen strict baseline gate.

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

The 45-case behavior gate reached `84.4%` overall. Evidence-grounded definitions and refusals scored `100%`, profile answers `81.8%`, unknown facts and retrieval decisions `75%` each. The separate 51-case [strict result](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-lfm2-strict-evaluation.json) reached `80.4%`: retrieval and refusals scored `100%`, unknown facts `90%`, and profile answers `71.0%`. Korean prompts received Korean responses in every case. Unknown-claim leak, refusal-scope leak, and answer-hallucination rates were all `0%`. The lower strict profile-answer score is the remaining weakness; routing is substantially more reliable than long-tail compositional recall in this 350M checkpoint.

The committed [behavior evaluation](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-lfm2-behavior-evaluation.json) and strict JSON include every generated answer, so aggregate scores cannot hide a fluent hallucination. DINOv3 and DETA specifically verify the switch from no-evidence retrieval to evidence-grounded synthesis without converting either entity into "Daniel's work."

The browser runtime has its own tests for the exact regression pairs. They verify RT-DETR definition versus contribution, ViTPose definition, UIUC location versus study history, entity pronoun follow-ups, visitor identity, private bank-account requests, and cited Wikipedia retrieval for unrelated questions such as Python's creator and distributed hash tables. This makes the product gate broader than the model checkpoint gate: both the model behavior and the code that routes around it must be correct.

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
- [Routing SFT and evidence-condition holdouts](https://github.com/SangbumChoi/sangbumchoi.github.io/tree/master/assets/data)
- [Dataset validators](https://github.com/SangbumChoi/sangbumchoi.github.io/tree/master/scripts)
- [Strict evaluator](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/evaluate_daniel_lfm2_test.py)
- [Loss plotting script](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/plot_daniel_lfm2_metrics.py)
- [GPU data-generation and ablation notebook](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/notebooks/daniel_lfm2_gpu_retraining.ipynb)
- [Synthetic prompt generator](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/generate_daniel_lfm2_synthetic.py)
- [Loss and leakage diagnostic](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/analyze_daniel_lfm2_data.py)
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

</div>

<div class="lang-block" lang="ko" markdown="1">

Daniel OS는 방문자의 브라우저 안에서 생성 모델을 실행하는 개인 포트폴리오 어시스턴트입니다. 검증된 프로필 인덱스, 출처가 달린 entity 인덱스, 선택적인 공개 retrieval, 대화형 합성을 위한 개인화 언어 모델, 브라우저 음성 인식, 로컬 음성 출력을 하나로 묶었습니다. 목표는 개인화가 일반적인 기술 지식을 왜곡하지 않도록 하면서 포트폴리오에 질문할 수 있게 만드는 것입니다.

이 글에서는 이미 구현된 것과 계획 단계인 것을 구분해서 설명합니다. LLM은 fine-tuning과 evaluation을 마쳤습니다. 현재 음성 계층은 브라우저 API를 사용하며, 직접 학습한 STT 모델도 아니고 제 목소리를 복제한 것도 아닙니다. 재현 가능한 STT 데이터, fine-tuning, evaluation pipeline이 이제 저장소에 있지만, 실제 다화자(multi-speaker) 데이터와 브라우저 테스트를 통과하기 전까지는 그 checkpoint가 브라우저 API를 대체하지 않습니다.

## 출처에 근거한 프로필 구축

먼저 제 CV, 포트폴리오, LinkedIn 프로필, 논문, GitHub, Hugging Face 계정에 있는 주장들을 모았습니다. 각 주장을 작은 [profile JSON](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-profile.json)으로 정규화한 뒤, 세 가지 출처(provenance) 상태 중 하나를 부여했습니다.

| 상태 | 의미 | 예시 |
| --- | --- | --- |
| 외부 검증됨 | 공식 논문, 저장소, 공개 API로 확인됨 | [MobileHumanPose at CVPRW 2021](https://openaccess.thecvf.com/content/CVPR2021W/MAI/html/Choi_MobileHumanPose_Toward_Real-Time_3D_Human_Pose_Estimation_in_Mobile_Devices_CVPRW_2021_paper.html) |
| 공개 자기 보고 | CV나 LinkedIn에 공개했지만, 공개된 기술 산출물에서 독립적으로 확인할 수는 없음 | 토스뱅크 프로젝트 범위와 내부 지표 |
| 검증되지 않음 | 신뢰할 만한 공개 출처를 찾지 못함 | 정확한 나이, 생일, 연봉, 또는 Team ISLAND에서 재즈 공연을 했다는 주장 |

외부 검증 과정에서는 KAIST 학력과 Team ISLAND CTO 이력을 공개 CV와 연결하고, [ZZAZZ](https://www.venturesquare.net/821623)가 Team ISLAND의 모바일 영상 편집 애플리케이션임을 확인했으며, [ZERO](https://arxiv.org/abs/2507.04270)와 MobileHumanPose는 논문 페이지로 확인했습니다. 또한 GitHub 공개 search API로 제가 작성한 Transformers pull request 28개를 조회했습니다. [provenance 파일](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-profile-sources.json)에는 조회 날짜와 URL이 저장되어 있습니다.

간접적인 신호로부터 개인 정보를 추론하지는 않습니다. 예를 들어 졸업 연도만으로는 제가 29살이라는 사실을 확정할 수 없습니다. 이 주장은 답변이 아니라 의도적으로 "알 수 없음" 테스트 케이스로 표현했습니다.

## 개인화 SFT만으로는 부족했던 이유

첫 번째 버전은 모든 질문이 저에 관한 것이라고 가정했습니다. 그 결과 어시스턴트는 포트폴리오 키워드를 매우 잘 알아보게 되었지만, 체계적인 오류도 생겼습니다. 방문자가 평범한 정의를 물었는데도 일반 명사가 제 이력 쪽으로 끌려 들어간 것입니다.

두 가지 실패가 서로 다른 원인을 드러냈습니다.

- "What is RT-DETR?"라는 질문에 few-shot learning과 negative sampling이 등장하는 "Daniel의 작업"에 대한 지어낸 설명이 나왔습니다. 작은 모델은 retrieval로 가져온 정의가 없었는데, SFT 분포는 Daniel 중심의 답변에 강한 보상을 주고 있었습니다.
- "Where is UIUC?"라는 질문에는 제 KAIST, POSTECH, UIUC 학력이 반환되었습니다. 이 답변은 모델에서 나온 것이 아니었습니다. 광범위한 JavaScript 키워드 규칙이 생성 전에 `uiuc`를 가로챘던 것입니다.

RT-DETR이나 UIUC에 대한 답을 더 외우게 하면 그 명사들은 땜질할 수 있겠지만 시스템은 고쳐지지 않습니다. 실제 구분은 의미적인 것입니다. "What is X?"는 X에 대해 묻는 것이고, "What did Daniel do with X?"는 포트폴리오와의 관계를 묻는 것입니다.

## 다섯 가지 근거 경로

이제 브라우저는 생성 전에 프롬프트를 분류합니다.

```text
visitor question
    |
    +-- profile fact ----------> verified profile JSON
    +-- known entity ---------> cited local entity index
    +-- unknown factual noun -> public retrieval
    +-- private-person data --> local refusal
    +-- synthesis -----------> LFM2 with supplied evidence
```

[knowledge router](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/js/knowledge-router.mjs)는 정의, 프로필 관계, 중립적인 외부 조회, 개인 정보 요청을 구분합니다. 또한 마지막으로 언급된 포트폴리오 entity를 기억하기 때문에, 모든 대명사를 사실로 바꾸지 않으면서도 "What did he do with it?" 같은 후속 질문을 해석할 수 있습니다.

[entity knowledge 파일](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-entity-knowledge.json)은 정의와 포트폴리오 관계를 분리해서 저장합니다. RT-DETR은 [원 논문](https://arxiv.org/abs/2304.08069)을, ViTPose는 [논문](https://arxiv.org/abs/2204.12484)을, UIUC는 [대학의 위치 안내 페이지](https://www.admissions.illinois.edu/about)를, SAM 2는 [Meta AI](https://ai.meta.com/research/sam2/)를, Molmo 2는 [Ai2](https://allenai.org/molmo)를 출처로 인용합니다. 따라서 각 레코드는 두 쪽을 섞지 않고 대비되는 어느 질문에든 답할 수 있습니다.

```json
{
  "name": "RT-DETR",
  "definition_en": "A cited definition of the detector itself.",
  "portfolio_relation_en": "Daniel's specific Transformers contribution.",
  "sources": [{"label": "RT-DETR paper", "url": "https://arxiv.org/abs/2304.08069"}]
}
```

알려진 entity는 로컬에서 즉시 답변합니다. 처음 보는 중립적 조회는 Wikipedia 공개 API를 사용하며, 가져온 페이지를 인용으로 표시합니다. retrieval이 실패했을 때 브라우저가 모델의 기억으로 대체하는 일은 절대 없습니다. 이렇게 해서 "유창한 문장은 근거로 취급하지 않는다"는 유용한 보장을 유지합니다.

공개 retrieval에는 조회어가 기기 밖으로 나간다는 프라이버시 비용이 있습니다. 포트폴리오 사실, 알려진 entity, 개인 정보 검사, 모델 추론, 대화 기록은 로컬에 남고, 캐시되지 않은 일반 조회만 Wikipedia로 전송됩니다. GitHub Pages는 상용 검색 API 키를 안전하게 숨길 수 없으므로, 제한 없는 다중 출처 검색을 하려면 rate limit이 걸린 서버나 edge proxy가 필요합니다. 현재의 fallback은 의도적으로 더 작고 검사 가능하게 유지했습니다.

## SFT 데이터셋 재설계

통합된 supervised 데이터셋은 이제 다섯 가지 행동에 걸쳐 296개의 대화를 담고 있습니다.

| 행동 | 레코드 수 | 학습 목표 |
| --- | ---: | --- |
| `answer` | 177 | 선택된 프로필 근거만으로 답변 |
| `ground_external` | 12 | 제공된 외부 근거만으로 entity를 정의 |
| `retrieve` | 15 | 추측하는 대신 공개 출처를 요청 |
| `unknown` | 58 | Daniel에 관한 특정 사실이 검증되지 않았음을 명시 |
| `refuse` | 34 | 개인 정보를 보호하고 안전하지 않거나 사실 기반이 아닌 작업을 거절 |

routing 부분집합에는 "What is RT-DETR?"와 "What did Daniel contribute to RT-DETR?", "Where is UIUC?"와 "When did Daniel study at UIUC?" 같은 대조(contrastive) 예시가 들어 있습니다. 영어와 한국어 프롬프트, 대명사를 쓰는 후속 질문, 그리고 거절하는 대신 retrieval을 요청해야 하는 중립적 사실도 포함합니다.

외부 근거에 기반한 레코드는 학습 프롬프트 안에 근거를 담고 있습니다.

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

근거가 제공되지 않은 경우, 목표 출력은 지어낸 답변이 아니라 작은 도구 호출 프로토콜입니다.

```text
<search_public_knowledge>contrastive learning</search_public_knowledge>
```

브라우저는 이 요청을 실행해 근거를 가져오고, 제어 토큰을 출처가 달린 답변으로 교체하며, 토큰 자체를 최종 응답으로 보여 주지 않습니다. 흔한 형태의 질문은 JavaScript routing이 먼저 직접 처리하고, fine-tuning된 도구 호출 행동은 결정론적 router가 놓친 표현을 위한 fallback입니다.

validator는 중복 프롬프트, 역할 순서, 알려진 프로필 키, 근거 존재 여부, 뒷받침되지 않는 수치 주장, 정확한 도구 호출 문법, 이중 언어 커버리지, 행동별 최소 개수를 검사합니다. 프로필 근거와 외부 근거는 별도의 필드이므로, 한쪽에서 찾은 숫자가 다른 쪽의 주장을 슬그머니 정당화할 수 없습니다.

첫 번째 trainer는 각 행동 안에서 예시를 held-out으로 분리했고, 개인화만 했던 원래 실행과는 다르게 실제 학습 스트림의 균형을 맞췄습니다. 소수 행동은 epoch마다 최소 64개의 예시를 기여하도록 했고, 177개의 프로필 답변은 더 이상 복제하지 않았습니다. 이렇게 하면 폭넓은 프로필 코퍼스를 버리지 않고도 "모든 질문에는 이력을 답해야 한다"는 사전 경향을 줄일 수 있었지만, 아래에서 설명할 이후의 loss 점검 결과 이 구현이 동일한 소수 예시를 지나치게 많이 반복했다는 사실이 드러났습니다.

가장 중요한 점은 evaluation이 단순히 paraphrase가 아니라 근거 조건(evidence condition)을 held-out으로 둔다는 것입니다. DINOv3와 DETA는 routing SFT에서 공개 retrieval을 촉발해야 하는 근거 없는 요청으로만 등장하며, 그 정의는 evaluation 때까지 공개되지 않습니다. evaluation 케이스는 이 보지 못한 근거를 제공하고, 모델이 저와의 관계를 지어내지 않으면서 검색에서 근거 기반 합성으로 전환하는지 테스트합니다. CLIP, NeRF, Carnegie Mellon University는 별도의 어휘 holdout 역할을 합니다. 이들은 SFT에 전혀 등장하지 않으며, 근거가 제공되지 않으면 각각 검색 요청을 촉발해야 합니다.

공개된 데이터셋 구성은 다음과 같습니다.

```text
sft/train.jsonl                  268 profile conversations
sft/routing.jsonl                28 routing conversations
behavior_eval/validation.jsonl   36 profile behavior checks
routing_eval/validation.jsonl     9 routing and evidence-holdout checks
strict_test/test.jsonl           51 post-training tests
profile/                         profile, provenance, and entity knowledge
metrics/                         training loss and strict evaluation
```

## 모델과 학습 loss

[LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M)을 LoRA로 fine-tuning하고, adapter를 base checkpoint에 병합한 뒤, 병합된 가중치를 대칭(symmetric) Q4 ONNX 그래프로 export합니다. 이것은 1비트 fine-tuning이 아닙니다. LoRA는 적응 과정을 메모리 효율적으로 만들고, Q4는 브라우저 다운로드 크기와 추론 메모리를 줄이기 위한 별도의 배포 단계입니다.

설정은 LoRA rank 16, alpha 32, dropout 0.05, 모든 linear layer를 대상으로, batch size 1, gradient accumulation 4, 최대 learning rate `2e-4`, 최대 시퀀스 길이 1,152, 3 epoch입니다. causal language modeling 목적 함수에는 어시스턴트 응답 토큰만 기여합니다.

```text
L = -(1 / N) sum[t in assistant tokens] log p(y_t | policy, profile, evidence, user, y_<t)
```

시스템 정책, 프로필 컨텍스트, 가져온 근거, 방문자 토큰은 loss에서 마스킹됩니다. 모델은 입력 근거를 그대로 재현하는 법이 아니라 답변 또는 routing 행동을 학습합니다.

![Daniel LFM2 학습 및 검증 loss]({{ '/assets/images/daniel-lfm2-loss.png' | relative_url }})

이 차트와 커밋된 [원시 지표](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-lfm2-training-metrics.json)는 routing 리비전 `e54fa04`에서 나온 것입니다. validation loss는 epoch 1의 `0.754`에서 epoch 2의 `0.536`으로 내려갔다가 epoch 3에서 `0.620`으로 올랐습니다. 그래서 trainer는 더 overfit된 마지막 epoch를 공개하는 대신 epoch 2를 복원했습니다. 보고된 평균 training loss는 `0.490`이었습니다. CPU GitHub runner는 균형을 맞춘 3 epoch 스트림에 15,045초, 약 4시간 11분이 걸렸습니다. 이후 병합된 checkpoint는 두 가지 행동 gate와 학습에 전혀 쓰이지 않은 strict 세트를 모두 통과했고, 그 다음 symmetric-Q4 ONNX export가 CPU 추론 smoke test를 통과했습니다.

### validation 곡선이 매끄럽지 않은 이유

곡선에는 validation 지점이 세 개뿐이며, 각 지점은 10개의 예시, 즉 행동당 최대 두 개로 계산되었습니다. [새 진단 스크립트](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/analyze_daniel_lfm2_data.py)는 실제 sampler도 재구성합니다. epoch당 431개의 학습 슬롯 중 145개, 즉 `33.6%`가 반복 슬롯이었습니다. 근거 기반 정의는 특히 집중되어 있었습니다. 10개의 고유한 학습 레코드가 64개 슬롯으로 순환되어 반복 배율이 `6.4x`였습니다. training loss는 계속 내려갔지만 마지막 validation loss는 epoch 2 최솟값보다 `15.7%` 높아졌습니다.

이는 epoch 2 이후 overfitting이 있었다는 증거이지만, 일반화 곡선을 정밀하게 측정했다고 주장하기에는 부족합니다. 10개 예시의 validation 평균은 분산이 크고, 행동 그룹마다 응답 길이가 크게 다르며, 학습 데이터 혼합 비율도 validation 혼합 비율과 크게 다릅니다. 생성 데이터를 더 늘리는 것만으로는 이런 측정 문제를 해결할 수 없습니다.

재시도는 synthetic data 연구를 좀 더 좁게 해석한 것에 기반합니다. [Self-Instruct](https://aclanthology.org/2023.acl-long.754/)는 seed를 복제하는 대신 다양한 instruction을 생성하고 필터링합니다. [LIMA](https://papers.neurips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html)는 신중하게 선별된 작은 alignment 세트가 훨씬 크고 노이즈가 많은 세트보다 나은 성능을 낼 수 있음을 보여 줍니다. [AlpaGasus](https://arxiv.org/abs/2307.08701) 역시 52,000개의 예시를 품질이 더 높은 9,000개로 필터링한 뒤 더 나은 결과를 보고하며, [DEITA](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6091f2bb355e960600f62566ac0e2862-Abstract-Conference.html)는 품질, 복잡도, 다양성을 중심으로 데이터 선택을 설계합니다. 여기서 얻은 실용적인 결론은 빈도를 복제하지 말고 커버리지를 넓히라는 것입니다.

그래서 [Colab GPU notebook](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/notebooks/daniel_lfm2_gpu_retraining.ipynb)은 4-bit Qwen3-4B teacher를 사용해 **질문만** 생성합니다. 목표 답변은 계속 사람이 선별하므로, teacher의 hallucination이 슬그머니 프로필 사실이 될 수 없습니다. 새로운 entity 정의 seed는 출처가 달린 entity 인덱스에서 가져오고, 관련 없는 공개 주제는 외운 세계 지식이 아니라 검색 도구 프로토콜을 목표로 합니다. 정확한 중복, 유사 중복, 언어가 잘못된 프롬프트, 메타 instruction은 제거합니다. 하나의 seed나 시나리오 계열에서 나온 변형은 모두 train 또는 validation 중 한쪽에만 들어갑니다.

목표 실험은 다섯 가지 행동에 걸쳐 약 1,820개의 학습 레코드와 300개의 validation 레코드를 사용하며, checkpoint 선택을 위해 행동별로 같은 크기의 슬라이스를 두고 행동별 loss 진단을 전부 수행합니다. optimizer step 25회마다 evaluation을 수행하고, 2회 evaluation 기준 early stopping을 적용하며, LFM2-350M과 [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) 모두에서 `5e-5`, `1e-4`, `2e-4`를 sweep합니다. 이것은 실험 계획이지 보고된 결과가 아닙니다. Colab 실행이 고정된 strict 기준선 gate를 모두 충족하거나 넘어서기 전까지 배포된 checkpoint와 그 지표는 바뀌지 않습니다.

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

## Evaluation 계약

학습 시점 gate는 이제 45개 케이스로 구성됩니다. 프로필 케이스 36개와 routing 케이스 9개입니다. 프로필 답변, 근거 기반 정의, retrieval 결정, 알 수 없는 사실, 거절, 한국어 행동을 각각 따로 보고합니다. 공개 strict 세트는 여전히 학습에 전혀 쓰이지 않았으며 51개 케이스로 구성됩니다. 프로필 답변 31개, 알 수 없는 사실 10개, 거절 9개, 일반 retrieval 케이스 1개입니다.

- **Route 정확도:** 정의, 프로필 관계, retrieval, 프라이버시, 거절 행동을 올바르게 선택하는지.
- **기대 사실 그룹 recall:** 필요한 의미적 사실 그룹이 응답에 나타나는지.
- **금지된 주장 회피:** 심어 둔 가짜 모델 이름, 지표, 개인 정보, 관련 없는 주장이 나타나지 않는지.
- **근거 뒷받침:** 모든 사실적 답변을 선택된 프로필 또는 외부 근거 객체로 추적할 수 있는지.
- **알 수 없는 주장 유출률:** 없는 프로필 사실을 질문에서 받아들여 버리지 않는지.
- **거절 범위 유출률:** 거절한 뒤 안전하지 않은 요청을 계속 수행하지 않는지.
- **한국어 응답률:** 한국어 프롬프트에 사용자에게 보이는 답변이 한국어로 나오는지.
- **Strict 통과율:** 행동, 근거, 금지된 주장, 언어 요건을 동시에 모두 통과하는지.

테스트는 정확히 일치하는 한 문장이 아니라 유효한 표현들의 그룹을 허용합니다. 예를 들어 `fivefold`, `five times`, `5x`는 같은 서빙 결과를 표현합니다. 금지어는 반대 방향을 테스트합니다. 10x 속도 향상을 암시하는 프롬프트가 있더라도 모델이 그것을 따라 말해서는 안 됩니다.

45개 케이스 행동 gate는 전체 `84.4%`에 도달했습니다. 근거 기반 정의와 거절은 `100%`, 프로필 답변은 `81.8%`, 알 수 없는 사실과 retrieval 결정은 각각 `75%`였습니다. 별도의 51개 케이스 [strict 결과](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-lfm2-strict-evaluation.json)는 `80.4%`에 도달했습니다. retrieval과 거절은 `100%`, 알 수 없는 사실은 `90%`, 프로필 답변은 `71.0%`였습니다. 한국어 프롬프트에는 모든 케이스에서 한국어 응답이 나왔습니다. 알 수 없는 주장 유출, 거절 범위 유출, 답변 hallucination 비율은 모두 `0%`였습니다. strict 프로필 답변 점수가 낮은 것이 남아 있는 약점입니다. 이 350M checkpoint에서는 routing이 long-tail 조합형 recall보다 훨씬 안정적입니다.

커밋된 [behavior evaluation](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-lfm2-behavior-evaluation.json)과 strict JSON에는 생성된 모든 답변이 포함되어 있으므로, 집계 점수가 유창한 hallucination을 가릴 수 없습니다. DINOv3와 DETA는 특히 어느 entity도 "Daniel의 작업"으로 바꾸지 않으면서 근거 없는 retrieval에서 근거 기반 합성으로 전환하는지를 검증합니다.

브라우저 런타임에는 정확히 그 회귀 쌍들을 위한 자체 테스트가 있습니다. RT-DETR 정의와 기여, ViTPose 정의, UIUC 위치와 학업 이력, entity 대명사 후속 질문, 방문자 신원, 개인 은행 계좌 요청, 그리고 Python을 만든 사람이나 distributed hash table 같은 관련 없는 질문에 대한 출처가 달린 Wikipedia retrieval을 검증합니다. 이렇게 하면 제품 gate가 모델 checkpoint gate보다 넓어집니다. 모델의 행동과 그 주변을 routing하는 코드가 모두 올바라야 합니다.

## 현재 STT와 TTS가 의미하는 것

영어 음성 입력은 현재 브라우저의 음성 인식 인터페이스를 사용합니다. 최종 transcript는 타이핑한 텍스트와 동일한 근거 기반 경로를 거칩니다. 음성 출력은 브라우저의 `speechSynthesis` 인터페이스와 브라우저 또는 운영체제에 설치된 음성을 사용합니다.

따라서 현재 릴리스에는 직접 만든 STT checkpoint도, 개인 TTS checkpoint도 없습니다. 브라우저 벤더는 음성 인식 구현 뒤에 있는 모델이나 목적 함수를 공개하지 않습니다. 현재 출력을 WebGPU STT 모델이나 제가 학습시킨 목소리라고 부르는 것은 부정확합니다.

STT와 TTS는 데이터 전략도 정반대입니다. 개인 TTS 모델은 동의한 한 명의 목표 화자, 즉 저를 학습해야 합니다. STT는 한 번도 들어 본 적 없는 방문자를 인식해야 하므로, 주로 제 목소리로 튜닝하면 잘못된 문제를 최적화하게 됩니다. [새 STT pipeline](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/README-stt.md)은 가명 처리된 여러 화자를 요구하며, 한 화자의 모든 발화를 train, validation, test 중 정확히 한 곳에만 둡니다. 또한 split 간 녹음 세션 및 오디오 해시 유출도 거부합니다.

STT manifest는 실명을 저장하지 않으면서 일반화를 진단하는 데 필요한 요인들을 기록합니다.

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

오디오는 모노 16 kHz PCM WAV, 0.4-30초입니다. 의도한 데이터 혼합은 주로 라이선스를 받은 다화자 영어 음성이며, 여기에 동의를 받은 휴대폰, 노트북, 헤드셋 녹음을 더합니다. 커밋된 [녹음 프롬프트](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-stt-capture-prompts.jsonl)에는 자연스러운 변형과 함께 Hugging Face, Molmo2, ZZAZZ, ZERO, WebGPU, Toss Bank 같은 어려운 이름이 포함되어 있습니다. 교정된 실패 사례는 명시적인 opt-in 이후에만 error-replay 슬라이스에 들어갈 수 있으며, 고정된 테스트 녹음 자체는 절대 학습 샘플이 되지 않습니다.

브라우저에 맞는 크기의 목표 모델은 `openai/whisper-tiny.en`이며, attention의 query와 value projection에 rank-16 LoRA를 적용해 적응시킵니다. 16 kHz 파형은 log-Mel spectrogram으로 변환되고, decoder가 sequence-to-sequence cross-entropy를 최소화하는 동안 padding된 transcript 토큰은 무시됩니다.

```text
L_ASR = -(1 / N) sum[t in transcript tokens] log p(y_t | log-Mel(audio), y_<t)
```

학습에는 Whisper SpecAugment와 함께 가벼운 gain, 속도, SNR perturbation을 적용합니다. 이런 augmentation은 화자, 억양, 공간, 마이크 전반에 걸친 실제 다양성을 대체하는 것이 아니라 보조하는 역할입니다. 더 큰 Distil-Whisper 모델이 transcript가 없는 동의 오디오에 pseudo-label을 제안할 수는 있지만, 불확실한 label과 포트폴리오 이름은 여전히 사람의 검토가 필요합니다. 더 작은 배포 목표를 선택한 이유는 Transformers.js가 이미 WebGPU에서 Whisper ASR을 지원하고, 포트폴리오 페이지에서는 첫 다운로드와 메모리 예산이 중요하기 때문입니다.

릴리스 gate는 의도적으로 평균 WER 하나보다 넓게 잡았습니다. micro WER, macro WER과 최악 화자 WER, substitution/deletion/insertion 개수, domain·environment·대략적 자기 보고 억양 그룹별 WER, 포트폴리오 키워드 recall, 모델 측 latency, real-time factor를 보고합니다. 통과한 PyTorch checkpoint는 후보일 뿐입니다. ONNX 빌드가 WebGPU와 WASM에서 고정된 테스트 모음을 다시 통과해야 하고, 다운로드 크기, 최대 메모리, 마이크 lifecycle, 첫 로드 latency, 브라우저 real-time factor 검사도 통과해야 합니다.

아직 STT loss 차트나 WER 결과는 없습니다. 실제 오디오 코퍼스가 pipeline에 들어온 적이 없기 때문입니다. trainer는 원시 로그 기록을 저장하며, [plotter](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/plot_daniel_stt_metrics.py)는 실제 train loss와 validation loss 지점이 모두 존재하지 않으면 그리기를 거부합니다. 이렇게 해서 계획된 실험이 결과처럼 보이는 것을 막습니다.

개인 TTS는 이후의 별도 단계로 남아 있습니다. 녹음은 명시적으로 동의한 제 목소리만 사용해야 하며, 선택한 구현이 요구하는 sample rate의 세션별로 분리된 모노 PCM WAV로 저장해야 합니다. 화자 조건부 VITS 또는 Piper 계열 후보라면 독립적인 ASR로 명료도(intelligibility)를, held-out speaker encoder로 화자 유사도를, 그리고 사람의 청취 점수를 보고하게 될 것입니다. 목적 함수는 보통 text-to-acoustic 재구성, duration 또는 alignment, KL, adversarial 항을 결합하지만, 정확한 loss는 아직 학습하지 않은 이 설계가 아니라 선택된 구현에 따라 정해집니다.

음성 데이터는 프로필 텍스트보다 민감합니다. [dataset publisher](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/publish_daniel_stt_dataset.py)는 기본적으로 비공개 저장소를 만들며, 공개 업로드 전에 모든 항목에 별도의 `allow_publication` 플래그를 요구합니다. 방문자 오디오는 기본적으로 저장되지 않고, 우연히 녹음된 화자는 제외되며, 현재 공개된 Hugging Face 프로필 데이터셋에는 음성 녹음이나 방문자 대화가 전혀 포함되어 있지 않습니다.

## 브라우저 런타임과 재현성

모델은 module Web Worker에서 실행되므로 모델 다운로드, ONNX 세션 생성, 토큰 생성이 인터페이스를 막지 않습니다. 최신 Chromium 브라우저는 WebGPU를 사용하고, 지원하지 않는 환경은 WASM으로 fallback합니다. 고정된 Q4 외부 가중치 파일은 정확히 289,140,736바이트이며, 여기에 그래프, tokenizer, 설정 파일이 더해집니다. 첫 로드에 성공하면 캐시됩니다.

WebGPU가 모델마다 독립적인 하드웨어를 주는 것은 아닙니다. 한 페이지가 여러 개의 논리적 `GPUDevice` 객체를 만들 수는 있지만, GPU 메모리와 연산은 다른 worker, 탭, 페이지, 애플리케이션과 공유되는 기기 전역 자원입니다. 따라서 상주하는 모델이 많아질수록 가중치와 중간 버퍼 메모리가 늘어나고 명령 실행을 두고 경쟁하게 됩니다. 부하가 충분히 커지면 할당이 실패하거나 브라우저가 device를 잃을 수 있습니다. 이는 빠른 개발용 컴퓨터 한 대에 근거한 가정이 아니라 [WebGPU 명세](https://gpuweb.github.io/gpuweb/)와 그 [설계 설명서](https://gpuweb.github.io/gpuweb/explainer/)를 따른 것입니다.

배포된 런타임은 이제 추가 device를 요청하지 않고 WebGPU adapter를 확인한 뒤, 보수적으로 compatibility, low, balanced, high 등급 중 하나를 부여합니다. software adapter이거나, 보고된 기기 메모리가 4 GB 이하이거나, 논리 CPU 코어가 4개 이하이면 eager loading을 끕니다. 이 경우 Q4 LLM은 자유 형식 요청이 있을 때만 로드되고 90초 동안 유휴 상태이면 해제됩니다. WebGPU adapter가 없으면 필요할 때 WASM으로 fallback합니다. 이것들은 VRAM 측정값이 아니라 힌트입니다. `navigator.deviceMemory`는 대략적이고 선택적인 값이며, adapter limit은 현재 남은 메모리가 아니라 허용되는 버퍼 크기를 보고합니다.

Daniel OS 안에서는 한 번에 하나의 무거운 모델만 상주할 수 있습니다. 나중에 로컬 STT와 개인 TTS가 승격되면 실행 순서는 `STT -> LLM -> TTS`가 되며, 세 모델을 모두 GPU에 올려 두는 대신 다음 세션을 확보하기 전에 이전 세션을 해제합니다. 현재 음성 API는 WebGPU 모델을 사용하지 않으므로, 지금은 Q4 LLM이 유일한 GPU 세션입니다. 별도의 탭은 여전히 별도의 사본을 만들 수 있습니다. 재현 가능한 benchmark에는 명시적인 2탭 경합 모드가 있어서 그 비용을 숨기지 않고 측정할 수 있습니다.

LFM2의 기본값이 Q4인 이유는 포트폴리오 페이지에서 첫 다운로드와 상주 가중치가 지배적이기 때문입니다. 그렇다고 Q4가 어디서나 가장 빠르다고 선언하는 것은 아닙니다. 일부 GPU에서는 dequantization 때문에 Q8이나 FP16이 더 빠를 수 있습니다. [Transformers.js dtype 가이드](https://huggingface.co/docs/transformers.js/guides/dtypes)도 Whisper 같은 encoder-decoder 모델이 quantization에 특히 민감할 수 있다고 경고합니다. 따라서 향후 STT gate는 Q8과 Q8-encoder/Q4-decoder 빌드를 비교하고, `shader-f16` 하드웨어에서는 FP16 encoder 실험도 진행합니다. 전부 Q4인 빌드는 WER, 최악 그룹 WER, 키워드 recall, 메모리, 브라우저 real-time factor를 모두 통과할 때만 배포합니다. 개인 TTS는 Q8과 FP16을 명료도, 화자 유사도, 청취 테스트 기준으로 비교할 것입니다. 1비트 추론은 현재 ONNX/Transformers.js 경로에 포함되어 있지 않습니다.

또한 처음부터 custom WGSL kernel을 작성하지도 않습니다. ONNX Runtime은 이미 WebGPU 연산자를 제공하며, profiling, CPU/GPU 전송 최소화, 그리고 recurrent tensor가 GPU에 머무르는 경우 I/O binding 사용을 권장합니다. 지원되는 export, ORT 포맷 또는 연산자를 줄인 빌드, 그래프 수준 fusion이 먼저입니다. custom kernel은 [ONNX Runtime Web profiler](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html)가 지원되지 않거나 느린 연산자 하나를 안정적인 지배 요인으로 지목하고, 그 대체 구현이 여러 GPU 벤더에서 정확성 테스트를 통과할 때에만 합리적인 선택이 됩니다.

로컬의 32 GB Mac은 소스 변경, mock 상호작용, 반응형 브라우저 확인을 담당합니다. 기존 LLM 원격 작업은 학습, Q4 export, CPU 추론 smoke test, strict 행동 evaluation, Hugging Face 게시, WebGPU 브라우저 확인을 수행합니다. STT workflow는 원격 A10G 작업에 연결되어 있지만, 필요한 동의 기반 다화자 코퍼스가 아직 없기 때문에 실행하지 않았습니다.

[런타임 정책과 benchmark 프로토콜](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/README-webgpu.md)에는 기기 매트릭스와 명령어가 기록되어 있습니다. 4 GB/4코어 브라우저 override는 저사양 속도 결과가 아니라 정책 에뮬레이션이라고 의도적으로 표시합니다. 음성 모델을 승격하기 전에 실제 첫 로드, warm 생성, 선택적 2탭 시간을 개발용 Mac과 실제 4 GB 및 8 GB 내장 GPU 기기에서 수집해야 합니다.

첫 번째로 통제된 개발용 Mac 점검은 Apple Metal 3 위에서 화면이 보이는 Chromium으로, 같은 프롬프트와 초기화된 컨텍스트를 사용해 진행했습니다. worker 안에서 Q4 초기화는 30.0초, warm 생성 1회는 3.22초가 걸렸습니다. 독립적으로 초기화된 두 탭이 동시에 생성할 때는 같은 응답에 6.58초와 6.84초가 걸려, 단일 세션 시간의 2.04배와 2.12배였습니다. 이는 보편적인 benchmark가 아니라 한 기기에서의 점검이지만, 논리적 세션들이 경합한다는 점을 확인해 주고 순차적 상주 방식을 뒷받침합니다. Headless Chromium은 SwiftShader를 노출했고, 보수적으로 low, on-demand 등급에 배정되었습니다.

전체 구현은 저장소에서 재현할 수 있습니다.

- [학습 및 병합 스크립트](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/train_daniel_lfm2.py)
- [Knowledge router와 공개 retrieval fallback](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/js/knowledge-router.mjs)
- [출처가 달린 포트폴리오 entity 인덱스](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/data/daniel-entity-knowledge.json)
- [Routing SFT와 근거 조건 holdout](https://github.com/SangbumChoi/sangbumchoi.github.io/tree/master/assets/data)
- [데이터셋 validator](https://github.com/SangbumChoi/sangbumchoi.github.io/tree/master/scripts)
- [Strict evaluator](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/evaluate_daniel_lfm2_test.py)
- [Loss 플로팅 스크립트](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/plot_daniel_lfm2_metrics.py)
- [GPU 데이터 생성 및 ablation notebook](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/notebooks/daniel_lfm2_gpu_retraining.ipynb)
- [Synthetic 프롬프트 생성기](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/generate_daniel_lfm2_synthetic.py)
- [Loss 및 유출 진단](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/analyze_daniel_lfm2_data.py)
- [병합된 LFM2 checkpoint](https://huggingface.co/danelcsb/daniel-lfm2-350m)
- [Hugging Face의 Q4 브라우저 모델](https://huggingface.co/danelcsb/daniel-lfm2-350m-ONNX)
- [Q4 브라우저 모델 릴리스](https://github.com/SangbumChoi/sangbumchoi.github.io/releases/tag/daniel-lfm2-onnx-v1)
- [브라우저 worker](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/js/lfm-worker.js)
- [적응형 WebGPU 런타임 정책](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/assets/js/runtime-policy.mjs)
- [WebGPU benchmark 프로토콜](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/README-webgpu.md)
- [화자 분리 STT 데이터 준비 스크립트](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/prepare_daniel_stt_dataset.py)
- [Whisper LoRA trainer와 릴리스 gate](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/train_daniel_stt.py)
- [그룹별 STT evaluator](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/score_daniel_stt_predictions.py)

원칙은 단순합니다. 행동은 fine-tuning하고, 지식은 retrieval로 가져오며, 개인 근거와 일반 근거를 분리하고, 유창한 실수를 드러내는 테스트를 공개하고, 학습하지 않은 모든 구성 요소를 정직하게 설명하는 것입니다.

</div>
