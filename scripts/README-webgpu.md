# Daniel OS WebGPU runtime policy

The deployed page currently has one WebGPU model: the Q4 personalized
LFM2-350M worker. Its external weight file is exactly 289,140,736 bytes. Speech
input still uses browser `SpeechRecognition`, speech output uses browser
`speechSynthesis`, portrait landmark detection runs once on CPU, and the live
portrait warp is Canvas 2D. Those components do not currently create additional
WebGPU model sessions.

## Resource policy

WebGPU devices are logical objects, while GPU memory and compute remain
machine-global resources. Creating separate model workers or tabs therefore
does not create independent throughput. Weight memory, intermediate buffers,
command submission, and browser GPU work still contend, and memory pressure can
end in allocation failure or device loss.

`assets/js/runtime-policy.mjs` applies these rules:

- probe an adapter without creating another `GPUDevice`;
- keep at most one heavyweight model resident inside the application;
- run future local speech and language stages sequentially as STT, LLM, TTS;
- use Q4 for the deployed LLM;
- avoid eager loading on a software adapter, a device reporting at most 4 GB,
  or a device reporting at most four logical CPU cores;
- release the low-tier model after 90 seconds idle;
- use WASM on demand when no WebGPU adapter is available.

`navigator.deviceMemory` is optional, coarse, and browser-capped. Adapter
limits describe legal allocation sizes, not free VRAM. The policy therefore
uses those values as conservative hints rather than claiming to measure memory.
Separate browser tabs can still create separate sessions, so the contention
benchmark below exists to expose that cost rather than assuming it away.

## Quantization matrix

There is no universal best bit width. Promotion requires both a quality gate
and measurements of first load, warm latency, peak memory, and artifact size.

| Component | Default candidate | Alternatives to measure | Quality gate |
| --- | --- | --- | --- |
| LFM2-350M | Q4 | Q8, FP16 on capable hardware | strict factual, privacy, and refusal suite |
| Whisper STT | Q8 or Q8 encoder/Q4 decoder | FP16 encoder/Q4 decoder when `shader-f16` is available | overall and worst-group WER, keyword recall, browser RTF |
| Personal TTS | Q8 | FP16 | independent-ASR intelligibility, speaker similarity, listening score |

Q4 is the LLM default because download and resident weight size dominate on a
portfolio page. It is not asserted to be fastest on every GPU: dequantization
overhead can make a wider dtype faster on some hardware. Whisper is an
encoder-decoder model and its encoder can be quantization-sensitive, so an
all-Q4 STT build must not ship merely because it is smaller. A 1-bit build is
outside the supported ONNX/Transformers.js path and would require a specialized
runtime and kernels; it is not a current candidate.

## Kernel decision

Do not hand-write WGSL kernels first. Transformers.js and ONNX Runtime already
provide WebGPU operator kernels and transformer optimizations. Start with a
supported export, keep recurrent tensors on the GPU where the runtime supports
I/O binding, and profile operator timing. ORT format and a reduced-operator
custom build may reduce startup and runtime download size. A custom kernel is
justified only when profiling identifies one stable dominant operation that is
unsupported or consistently slow, after correctness tests across Apple,
Intel/AMD, and software fallback adapters.

## Reproducible checks

Policy tests do not download the model:

```sh
node --test scripts/test_runtime_policy.mjs
```

Inspect the local browser and pinned Q4 size:

```sh
NODE_PATH=/path/to/node_modules node scripts/benchmark_webgpu_runtime.mjs
```

Emulate only the low-spec policy decision:

```sh
EMULATE_LOW_SPEC=1 NODE_PATH=/path/to/node_modules \
  node scripts/benchmark_webgpu_runtime.mjs
```

Run actual model initialization and one warm generation on the current
machine in a visible hardware-backed browser, then optionally compare two
independent tabs:

```sh
HEADLESS=0 FULL_MODEL=1 NODE_PATH=/path/to/node_modules \
  node scripts/benchmark_webgpu_runtime.mjs

HEADLESS=0 FULL_MODEL=1 CONTENTION=1 NODE_PATH=/path/to/node_modules \
  node scripts/benchmark_webgpu_runtime.mjs
```

Never describe the low-spec emulation as a hardware benchmark. Before promoting
a browser STT or TTS model, repeat the full matrix on at least a 4 GB integrated
GPU/low-memory device, an 8 GB integrated device, and the development Mac.

## Development Mac audit

On 2026-07-20, visible Chromium reported the 32 GB development Mac as an Apple
Metal 3 adapter with 10 logical CPU cores, `shader-f16`, and a non-fallback
adapter. The controlled run used the same deterministic prompt after clearing
conversation state in both tabs:

| Measurement | Result |
| --- | ---: |
| First Q4 load and session initialization | 30,017 ms |
| Single-tab warm generation, worker time | 3,223 ms |
| Second independent tab initialization | 29,197 ms |
| Two simultaneous generations, worker time | 6,579 ms / 6,837 ms |
| Two-tab wall time | 6,995 ms |

The two concurrent sessions took 2.04x and 2.12x the single-session worker
time. This one-device audit is not a universal throughput benchmark, but it
directly rejects the assumption that separate WebGPU sessions preserve speed.
It supports the one-resident-model, sequential-stage policy. Headless Chromium
on the same Mac exposed SwiftShader instead of Metal and was correctly assigned
the low tier; a 4 GB/four-core override also stayed on-demand and did not fetch
the model automatically.

Primary references: [WebGPU specification](https://gpuweb.github.io/gpuweb/),
[WebGPU explainer](https://gpuweb.github.io/gpuweb/explainer/),
[Transformers.js dtypes](https://huggingface.co/docs/transformers.js/guides/dtypes),
[Transformers.js WebGPU](https://huggingface.co/docs/transformers.js/en/guides/webgpu),
[ONNX Runtime Web performance diagnosis](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html),
[ONNX Runtime WebGPU execution provider](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html),
and [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/).
