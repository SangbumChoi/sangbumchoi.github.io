#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub>=1.0"]
# ///
"""Export the personalized Daniel OS checkpoint to browser-ready Q4 ONNX."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


EXPORTER_REPOSITORY = "https://github.com/Liquid4All/onnx-export.git"
EXPORTER_REVISION = "9a23ddd23035165f7414a5de3220a51e85780f64"
DEFAULT_SOURCE = "danelcsb/daniel-lfm2-350m"
DEFAULT_DESTINATION = "danelcsb/daniel-lfm2-350m-ONNX"


def run(*command: str, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def write_model_card(export_dir: Path, source: str) -> None:
    source_reference = (
        source
        if not Path(source).exists()
        else "SangbumChoi/sangbumchoi.github.io release daniel-lfm2-source-v2"
    )
    source_link = (
        f"[{source}](https://huggingface.co/{source})"
        if not Path(source).exists()
        else "the Daniel OS merged source checkpoint published on GitHub"
    )
    (export_dir / "README.md").write_text(
        f"""---
library_name: transformers.js
pipeline_tag: text-generation
base_model:
- LiquidAI/LFM2-350M
- {source_reference}
tags:
- transformers.js
- onnx
- webgpu
- lfm2
- portfolio-assistant
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2-350M/blob/main/LICENSE
---

# Daniel OS LFM2-350M ONNX

Browser-ready Q4 ONNX export of {source_link},
the personalized language model used by Sangbum Daniel Choi's portfolio.

The model was exported with Liquid AI's official
[LiquidONNX](https://github.com/Liquid4All/onnx-export) tooling at revision
`{EXPORTER_REVISION}`. Q4 uses symmetric quantization for WebGPU compatibility.

The portfolio keeps a deterministic verified-profile index in front of model
generation. This checkpoint supplies conversational synthesis and tone; it is
not used as the source of truth for dates, metrics, or links.

```javascript
import {{ pipeline }} from "@huggingface/transformers";

const generator = await pipeline(
  "text-generation",
  "{DEFAULT_DESTINATION}",
  {{ device: "webgpu", dtype: "q4" }},
);
```
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--destination", default=DEFAULT_DESTINATION)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/onnx-export"))
    parser.add_argument("--upload-hf", action="store_true")
    parser.add_argument("--skip-smoke-test", action="store_true")
    args = parser.parse_args()

    if args.upload_hf and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required to upload the ONNX model.")

    with tempfile.TemporaryDirectory(prefix="daniel-lfm2-onnx-") as temp:
        root = Path(temp)
        exporter = root / "liquidonnx"
        output_root = args.output_dir.resolve()

        run("git", "clone", "--filter=blob:none", EXPORTER_REPOSITORY, str(exporter))
        run("git", "checkout", EXPORTER_REVISION, cwd=exporter)
        run(
            "uv",
            "run",
            "--project",
            str(exporter),
            "lfm2-export",
            args.source,
            "--output-dir",
            str(output_root),
            "--output-name",
            "daniel-lfm2-350m-ONNX",
            "--precision",
            "q4",
        )

        export_dir = output_root / "exports" / "daniel-lfm2-350m-ONNX"
        q4_model = export_dir / "onnx" / "model_q4.onnx"
        if not q4_model.exists():
            raise FileNotFoundError(f"Expected Q4 graph was not created: {q4_model}")

        if not args.skip_smoke_test:
            run(
                "uv",
                "run",
                "--project",
                str(exporter),
                "lfm2-infer",
                "--model",
                str(q4_model),
                "--prompt",
                "Who are you?",
                "--cpu",
            )

        write_model_card(export_dir, args.source)
        print(f"Browser export written to {export_dir}", flush=True)

        if args.upload_hf:
            api = HfApi()
            api.create_repo(args.destination, repo_type="model", private=False, exist_ok=True)
            commit = api.upload_folder(
                repo_id=args.destination,
                repo_type="model",
                folder_path=export_dir,
                commit_message="Upload personalized Q4 WebGPU export",
            )
            print(f"Uploaded {args.destination}: {commit.oid}", flush=True)


if __name__ == "__main__":
    main()
