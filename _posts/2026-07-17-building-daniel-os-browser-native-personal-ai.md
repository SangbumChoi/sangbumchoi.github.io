---
title: "Building Daniel OS: data, training, and strict evaluation"
permalink: /posts/daniel-os-lfm2/
date: 2026-07-17
eyebrow: "FIELD NOTE / LOCAL AI"
dek: "How I built a source-grounded personal dataset, fine-tuned LFM2-350M, measured its boundaries, and designed a generalized browser STT pipeline."
read_time: true
comments: false
share: false
related: false
---

Daniel OS is a personal portfolio assistant that runs its generative model in the visitor's browser. It combines a verified profile index for exact facts, a personalized language model for conversational synthesis, browser speech recognition, and local speech output. The goal is to make the portfolio queryable while keeping visitor conversations on the device.

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

## LLM dataset design

The supervised dataset contains 79 one-turn conversations: 52 grounded answers, 9 missing-fact responses, and 18 out-of-scope refusals. These behaviors are intentionally different:

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

For loss evaluation, the trainer takes a stratified five-conversation holdout: two `answer`, one `unknown`, and two `refuse` examples. After that holdout, minority behaviors are modestly repeated in the effective training stream so the model sees enough boundary examples without making refusal its dominant behavior. A separate 20-prompt validation gate is never used as a training target.

The public test set contains 42 harder prompts: 26 factual answers, 7 unknown facts, and 9 refusals. Ten are Korean. It includes cross-section synthesis, exact metric traps, privacy questions, unsupported internal model names, prompt injection, the KAIST plus Team ISLAND plus ZERO career connection, and a three-case ZZAZZ product-depth extension. It is a post-training test split and is not fed back into fine-tuning.

All three configurations are published in the [Daniel OS dataset on Hugging Face](https://huggingface.co/datasets/danelcsb/daniel-os-profile-sft):

```text
sft/train.jsonl                 79 conversations
behavior_eval/validation.jsonl  20 held-out behavior checks
strict_test/test.jsonl          42 post-training tests
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

The original 18-prompt gate used for the current checkpoint scored 77.8% overall: 60% for grounded profile answers, 100% for missing facts, and 100% for refusals. The current data revision expands that gate to 20 prompts, but those new results require the next training run.

The 42-case strict test records several metrics instead of reporting only one average. It preserves the original 39 cases and adds three untouched ZZAZZ product questions:

- **Expected fact-group recall:** how many required semantic fact groups appear in the answer.
- **Behavior pass rate:** all expected groups are present and no forbidden claim appears.
- **Forbidden-claim avoidance:** a controlled hallucination proxy that checks whether the answer repeats planted false numbers, model names, personal facts, or unrelated content.
- **Unknown claim leak rate:** an unknown-fact answer does not adopt the unsupported claim from the question.
- **Refusal scope leak rate:** a refusal does not go on to answer the unrelated request.
- **Korean response rate:** Korean prompts receive a response containing Korean, in addition to passing the behavior checks.
- **Strict pass rate:** behavior, forbidden-claim, and language requirements all pass together.

On the 42-case revision, the pre-ZZAZZ checkpoint scored 42.9% on behavior pass, 31.0% on strict pass, 47.6% on expected fact-group recall, and 97.6% on controlled forbidden-claim avoidance. By behavior, it reached 34.6% for factual answers, 28.6% for unknown facts, and 77.8% for refusals. Korean response rate remained 0%. The complete [strict evaluation JSON](https://huggingface.co/datasets/danelcsb/daniel-os-profile-sft/resolve/main/metrics/strict-evaluation.json) includes all 42 prompts and generated answers. These are deliberately reported as pre-retraining measurements; the new product examples had not yet entered that checkpoint.

This is a baseline, not a success claim. The model has learned a useful refusal boundary and usually avoids planted false claims, but its compositional fact recall is weak and the English-only SFT data did not produce Korean answers. A manual audit also found an unsupported `Max Bin` name in one English response to a Korean identity prompt. That error was not one of the planted forbidden terms, so the 97.4% proxy does not measure every possible hallucination. Publishing every generated answer makes that limitation auditable. I keep this test version fixed and public rather than tuning directly on its failures. A later bilingual training revision should use newly written Korean examples and a separate untouched test set.

A test case accepts groups of valid phrases rather than requiring one exact reference sentence. For example, `fivefold`, `five times`, and `5x` can express the same serving result. Forbidden terms test the opposite direction: a question that suggests a 10x speedup must not cause the model to repeat it.

The browser adds another layer. Common profile questions are routed to the deterministic JSON index before generation. ZZAZZ uses a small local profile tool: it resolves the product from a direct Team ISLAND question, remembers that topic for follow-ups such as "What was that?" or "How did it work?", and returns curated VentureSquare and theBell links without sending the conversation to a remote service. A production test verifies both the product answer and the pronoun-style technical follow-up alongside questions about Toss Bank, multimodal training, open source, education, and publications. This keeps exact dates, metrics, and product definitions reliable even when a small language model phrases a synthesis imperfectly.

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

The principle is simple: train the model to communicate within a narrow scope, keep exact facts in a source-grounded layer, publish tests that can expose failure, and describe every untrained component honestly.
