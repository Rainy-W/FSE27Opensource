#!/usr/bin/env python3
"""Download the pretrained checkpoints used by the released HUMER models."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_IDS = {
    "codebert": "microsoft/codebert-base",
    "unixcoder": "microsoft/unixcoder-base",
    "codet5": "Salesforce/codet5-base",
    "linevul": "microsoft/codebert-base",
    "vulgpt": "microsoft/codebert-base-mlm",
    "epvd": "microsoft/codebert-base",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="models", help="Directory for local checkpoints.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_IDS),
        default=sorted(MODEL_IDS),
        help="Model adapters whose base checkpoints should be downloaded.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    for model in args.models:
        model_id = MODEL_IDS[model]
        if model_id in downloaded:
            print(f"{model}: reuse {downloaded[model_id]}")
            continue
        target = output_dir / model_id.replace("/", "--")
        snapshot_download(repo_id=model_id, local_dir=target)
        downloaded[model_id] = target
        print(f"{model}: {target}")


if __name__ == "__main__":
    main()
