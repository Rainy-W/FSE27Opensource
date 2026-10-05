#!/usr/bin/env python3
"""Convert explicit dataset splits to HUMER's JSON-list format without resplitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list or use the .jsonl extension.")
    return payload


def first_value(row: dict[str, Any], names: tuple[str, ...], index: int) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"Row {index} does not contain any of the expected fields: {names}")


def integer_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        digest = hashlib.sha256(str(value).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def convert(source: Path, target: Path) -> None:
    converted = []
    seen_ids: set[int] = set()
    for index, row in enumerate(read_rows(source)):
        code = str(first_value(row, ("code", "func", "function"), index))
        label = int(first_value(row, ("label", "target"), index))
        if label not in {0, 1}:
            raise ValueError(f"Row {index} in {source} has non-binary label {label}.")
        raw_id = next((row[name] for name in ("id", "idx", "sample_id") if name in row), index)
        sample_id = integer_id(raw_id)
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample ID {sample_id} in {source}.")
        seen_ids.add(sample_id)
        converted.append({"id": sample_id, "code": code, "label": label})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{source} -> {target}: {len(converted)} samples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    convert(args.train, args.output_dir / "train.json")
    convert(args.validation, args.output_dir / "val.json")
    convert(args.test, args.output_dir / "test.json")


if __name__ == "__main__":
    main()
