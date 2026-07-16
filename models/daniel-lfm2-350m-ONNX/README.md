---
library_name: transformers.js
pipeline_tag: text-generation
base_model:
- LiquidAI/LFM2-350M
- danelcsb/daniel-lfm2-350m
tags:
- transformers.js
- onnx
- webgpu
- lfm2
- portfolio-assistant
license: lfm1.0
---

# Daniel OS LFM2-350M ONNX

Browser-ready Q4 ONNX export of [danelcsb/daniel-lfm2-350m](https://huggingface.co/danelcsb/daniel-lfm2-350m),
the personalized language model used by Sangbum Daniel Choi's portfolio.

The model was exported with Liquid AI's official
[LiquidONNX](https://github.com/Liquid4All/onnx-export) tooling at revision
`9a23ddd23035165f7414a5de3220a51e85780f64`. Q4 uses symmetric quantization for WebGPU compatibility.

The portfolio keeps a deterministic verified-profile index in front of model
generation. This checkpoint supplies conversational synthesis and tone; it is
not used as the source of truth for dates, metrics, or links.

```javascript
import { pipeline } from "@huggingface/transformers";

const generator = await pipeline(
  "text-generation",
  "danelcsb/daniel-lfm2-350m-ONNX",
  { device: "webgpu", dtype: "q4" },
);
```
