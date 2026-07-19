---
language:
- en
task_categories:
- automatic-speech-recognition
pretty_name: Daniel OS Generalized Browser STT
---

# Daniel OS generalized browser STT

This card defines the private training format for Daniel OS speech recognition.
The public repository contains only the schema, capture prompts, and pipeline;
it does not contain voice recordings or visitor conversations.

## Objective

TTS personalization learns one consented target voice. STT has the opposite
requirement: it must recognize previously unseen speakers across accents,
microphones, speaking rates, and background conditions. All examples from one
speaker are therefore assigned to exactly one of train, validation, or test.
Recording sessions and audio hashes are checked for the same leakage.

## Data mixture

The intended training mixture is mostly licensed, multi-speaker general English
speech, supplemented by consented mobile and laptop recordings. Portfolio names
and error-replay examples are capped so adaptation does not turn into
single-speaker or single-domain memorization. A practical first collection target
is at least 50 speakers, with at least 10 unseen speakers in the fixed test set.

Every record contains:

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
  "accent_group": "self_reported_group",
  "consent": "explicit-v1",
  "split": "train"
}
```

Audio is mono, 16 kHz PCM WAV, between 0.4 and 30 seconds. Demographic and
accent labels are optional, self-reported, coarse, and used only to diagnose
coverage gaps. Raw identity is never used as `speaker_id`.

## Privacy

- Visitor audio is processed ephemerally and is not a training source by default.
- An error replay requires an explicit opt-in, corrected transcript, and consent version.
- Incidental speakers are excluded.
- The dataset repository is private unless every item separately permits publication.
- The untouched test set is not mined for training examples.

## Evaluation

The release gate reports micro WER, macro speaker WER, worst-speaker WER,
domain/environment/accent-group WER, substitution/deletion/insertion counts,
portfolio keyword recall, model-side latency, and real-time factor. Browser
deployment adds download size, peak memory, WebGPU/WASM latency, and regression
tests on the same frozen audio suite.
