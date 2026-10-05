#!/usr/bin/env python3
"""Check the HUMER installation without downloading a pretrained model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers

from humer.config import HumerConfig, SUPPORTED_MODELS


def validate_split(path: Path) -> tuple[int, dict[int, int]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list.")
    counts = {0: 0, 1: 0}
    ids: set[int] = set()
    for index, row in enumerate(rows):
        if not {"code", "label"}.issubset(row):
            raise ValueError(f"{path} row {index} must contain code and label.")
        label = int(row["label"])
        if label not in counts:
            raise ValueError(f"{path} row {index} has non-binary label {label}.")
        sample_id = int(row.get("id", index))
        if sample_id in ids:
            raise ValueError(f"{path} contains duplicate sample ID {sample_id}.")
        ids.add(sample_id)
        counts[label] += 1
    return len(rows), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    config = HumerConfig.from_yaml(
        args.config,
        overrides={"data_dir": args.data_dir, "output_dir": "outputs/install_check"},
    )
    print(f"torch={torch.__version__}")
    print(f"transformers={transformers.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"supported_models={','.join(SUPPORTED_MODELS)}")
    for split in ("train", "val", "test"):
        path = Path(config.data_dir) / f"{split}.json"
        total, counts = validate_split(path)
        print(f"{split}: total={total} label0={counts[0]} label1={counts[1]}")
    print("installation_check=passed")


if __name__ == "__main__":
    main()
