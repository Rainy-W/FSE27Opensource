#!/usr/bin/env python3
"""Install the minimal runtime files from a user-downloaded EPVD artifact."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Path to the extracted EPVD directory.")
    args = parser.parse_args()
    source = Path(args.source_dir).resolve()
    required = {
        source / "code" / "c_cfg.py": Path("third_party/epvd_official/EPVD/code/c_cfg.py"),
        source / "parserTool" / "my-languages.so": Path(
            "third_party/epvd_official/EPVD/parserTool/my-languages.so"
        ),
    }
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"The selected EPVD artifact is missing: {missing}")
    for source_path, target_path in required.items():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        print(f"installed {target_path}")


if __name__ == "__main__":
    main()
