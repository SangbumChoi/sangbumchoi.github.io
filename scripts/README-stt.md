# Daniel OS generalized browser STT

The deployed site currently uses the browser `SpeechRecognition` interface.
This directory adds a reproducible path to replace it with a small English
Whisper checkpoint after the custom model passes accuracy, privacy, memory, and
browser-latency gates. It does not claim that the custom checkpoint is deployed.

## Why STT and TTS use different data

Personal TTS is a one-speaker problem: only Daniel's explicitly consented voice
belongs in the target-voice dataset. STT is a many-speaker generalization
problem. It needs variation in speaker, accent, microphone, room, speaking rate,
and noise, while keeping each speaker entirely inside one split.

The STT mixture has three sources:

1. Licensed multi-speaker English speech for broad coverage.
2. Consented phone, laptop, and headset recordings of the committed capture prompts.
3. Opt-in error replays with a human-corrected transcript for recurrent failures.

Portfolio terms and error replays should remain a small slice of training. The
fixed test set is never converted into training data. A larger Whisper or
Distil-Whisper checkpoint may propose pseudo-labels, but uncertain labels and
portfolio names require human review before admission.

## Manifest and split preparation

Audio must be mono 16 kHz PCM WAV, 0.4-30 seconds. Start from
`assets/data/daniel-stt-manifest.example.jsonl`, replace the pseudonymous sample
records with consented recordings, and omit `split` when the script should assign
speakers deterministically:

```sh
python3 scripts/prepare_daniel_stt_dataset.py \
  --manifest private-audio/raw-manifest.jsonl \
  --output-dir artifacts/daniel-stt-data \
  --minimum-speakers 50
```

The preparer verifies consent, transcript normalization, duration, sample rate,
SHA-256, unique IDs, and public-data licenses. It copies audio into the ignored
`artifacts/` directory and rejects speaker, session, or audio-hash leakage across
train, validation, and test. The example schema can be checked without audio:

```sh
python3 scripts/prepare_daniel_stt_dataset.py \
  --manifest assets/data/daniel-stt-manifest.example.jsonl \
  --output-dir /tmp/daniel-stt-schema \
  --schema-only
```

## Private dataset publication

The publisher creates a private Hugging Face dataset by default:

```sh
HF_TOKEN=... uv run scripts/publish_daniel_stt_dataset.py \
  --data-dir artifacts/daniel-stt-data \
  --repo-id danelcsb/daniel-os-stt
```

`--public` is intentionally harder: every manifest row must contain
`"allow_publication": true`. Recording consent alone is not treated as consent
to publish a biometric voice sample.

## Training

The browser target is `openai/whisper-tiny.en`. LoRA rank 16 adapters are placed
on attention query and value projections. Padded target tokens are masked, and
the remaining decoder tokens use sequence-to-sequence cross-entropy:

```text
L_ASR = -(1 / N) sum[t in transcript tokens] log p(y_t | log-Mel(audio), y_<t)
```

The train collator applies light gain, speed, and SNR perturbation; Whisper
SpecAugment masks time and feature regions. Recorded environment diversity is
still required because synthetic noise is not a replacement for real devices
and rooms.

For a direct remote run:

```sh
hf jobs uv run \
  --name daniel-stt-tiny-en \
  --flavor a10g-small \
  --timeout 4h \
  --secrets HF_TOKEN \
  scripts/train_daniel_stt.py \
  --dataset-repo danelcsb/daniel-os-stt \
  --hub-repo danelcsb/daniel-stt-tiny-en \
  --push-to-hub
```

`.github/workflows/train-publish-daniel-stt.yml` submits the same job manually.
The model repository is private unless `--public-model` is explicit.

## Evaluation contract

The validation split selects the checkpoint by WER. The untouched test split
then measures:

- micro WER and substitution/deletion/insertion counts;
- macro and worst-speaker WER;
- WER by domain, environment, and self-reported coarse accent group;
- recall for names such as Toss Bank, Hugging Face, Molmo2, ZZAZZ, and ZERO;
- model-side latency and real-time factor.

The default release gate is test WER <= 0.35, worst non-speaker group WER <=
0.65, and keyword recall >= 0.75. These initial gates should tighten after the
first audited baseline; they are not benchmark results.

Browser transcripts can be scored independently:

```sh
python3 scripts/score_daniel_stt_predictions.py \
  --predictions artifacts/browser-stt-predictions.jsonl \
  --max-wer 0.35 \
  --max-worst-group-wer 0.65 \
  --min-keyword-recall 0.75
```

After training, `plot_daniel_stt_metrics.py` renders actual train and validation
loss from `training_metrics.json`. No placeholder curve is committed.

## Browser promotion

A passed PyTorch checkpoint is only a candidate. Convert the merged model to
ONNX, quantize it, and load it through a Web Worker with Transformers.js. The
site should switch away from browser speech recognition only after a frozen
browser suite passes on WebGPU and WASM for WER, microphone lifecycle, download
size, peak memory, first-load latency, and real-time factor. The current browser
API remains the fallback.

Primary references: [Whisper](https://github.com/openai/whisper),
[Hugging Face ASR training](https://huggingface.co/docs/transformers/tasks/asr),
[audio datasets](https://huggingface.co/docs/datasets/audio_dataset),
[PEFT LoRA](https://github.com/huggingface/peft), and
[Transformers.js WebGPU](https://huggingface.co/docs/transformers.js/en/guides/webgpu).
