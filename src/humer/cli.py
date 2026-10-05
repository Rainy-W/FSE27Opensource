"""Command-line entry point for the released HUMER pipeline."""

from __future__ import annotations

import argparse
import os

from .config import HumerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HUMER from a versioned YAML configuration.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    parser.add_argument("--output-dir", required=True, help="New directory for this run.")
    parser.add_argument("--data-dir", default=None, help="Override the dataset directory from YAML.")
    parser.add_argument("--model-name", default=None, help="Override the Hugging Face model ID or local path.")
    parser.add_argument("--seed", type=int, default=None, help="Override the configured random seed.")
    parser.add_argument("--device", default=None, help="PyTorch device, for example cuda:0 or cpu.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=None,
        help="Load pretrained weights only from the local Hugging Face cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "output_dir": args.output_dir,
        "data_dir": args.data_dir,
        "model_name": args.model_name,
        "seed": args.seed,
        "device": args.device,
        "local_files_only": args.local_files_only,
    }
    config = HumerConfig.from_yaml(args.config, overrides=overrides)
    if config.strict_determinism:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from .runner import HumerRunner

    HumerRunner(config).run()


if __name__ == "__main__":
    main()
