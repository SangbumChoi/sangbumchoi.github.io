#!/usr/bin/env python3
"""Pin a published Hugging Face model revision and invalidate browser caches."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


WORKER = Path("assets/js/lfm-worker.js")
JARVIS = Path("assets/js/jarvis.js")
CONFIG = Path("_config.yml")


def replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not update {label}")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("revision", help="40-character Hugging Face model commit")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("revision must be a lowercase 40-character Git SHA")

    worker = WORKER.read_text(encoding="utf-8")
    current = re.search(r'const MODEL_REVISION = "([0-9a-f]{40})";', worker)
    if not current:
        raise RuntimeError("Could not find MODEL_REVISION")
    if current.group(1) == args.revision:
        print(f"Model revision is already pinned to {args.revision}")
        return

    jarvis = JARVIS.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")
    js_version = re.search(r'const ASSET_VERSION = "(\d+)";', jarvis)
    config_version = re.search(r"^jarvis_asset_version\s*:\s*(\d+)\s*$", config, re.MULTILINE)
    if not js_version or not config_version:
        raise RuntimeError("Could not read matching Jarvis asset versions")
    if js_version.group(1) != config_version.group(1):
        raise RuntimeError("Jarvis JavaScript and Jekyll asset versions differ")
    next_version = str(int(js_version.group(1)) + 1)

    worker = replace_once(
        worker,
        r'const MODEL_REVISION = "[0-9a-f]{40}";',
        f'const MODEL_REVISION = "{args.revision}";',
        "MODEL_REVISION",
    )
    jarvis = replace_once(
        jarvis,
        r'const ASSET_VERSION = "\d+";',
        f'const ASSET_VERSION = "{next_version}";',
        "ASSET_VERSION",
    )
    jarvis, import_count = re.subn(
        r'((?:portrait-landmarks\.js|portrait-mesh\.js|runtime-policy\.mjs)\?v=)\d+',
        rf"\g<1>{next_version}",
        jarvis,
    )
    if import_count != 3:
        raise RuntimeError("Could not update all versioned Jarvis module imports")
    config = replace_once(
        config,
        r"^(jarvis_asset_version\s*:\s*)\d+\s*$",
        rf"\g<1>{next_version}",
        "jarvis_asset_version",
    )

    WORKER.write_text(worker, encoding="utf-8")
    JARVIS.write_text(jarvis, encoding="utf-8")
    CONFIG.write_text(config, encoding="utf-8")
    print(f"Pinned model {args.revision}; Jarvis asset version is now {next_version}")


if __name__ == "__main__":
    main()
