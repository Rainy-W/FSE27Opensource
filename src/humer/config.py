"""Configuration loading and validation for the released HUMER pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_MODELS = ("codebert", "unixcoder", "codet5", "linevul", "vulgpt", "epvd")
DEFAULT_MODEL_NAMES = {
    "codebert": "microsoft/codebert-base",
    "unixcoder": "microsoft/unixcoder-base",
    "codet5": "Salesforce/codet5-base",
    "linevul": "microsoft/codebert-base",
    "vulgpt": "microsoft/codebert-base-mlm",
    "epvd": "microsoft/codebert-base",
}


@dataclass
class HumerConfig:
    """Resolved runtime configuration.

    The curriculum and review formulas are intentionally not configurable in
    the release pipeline. This dataclass only exposes dataset, model, runtime,
    and conventional optimization parameters.
    """

    data_dir: str
    output_dir: str
    model: str
    model_name: str | None = None
    seed: int = 42
    batch_size: int = 32
    eval_batch_size: int = 32
    max_length: int = 512
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_buckets: int = 5
    consolidation_max_epochs: int = 20
    consolidation_patience: int = 5
    hf_cache_dir: str = ".cache/huggingface"
    difficulty_cache_dir: str = "outputs/difficulty_cache"
    local_files_only: bool = False
    device: str | None = None
    strict_determinism: bool = True
    decision_threshold: float = 0.5
    epvd_block_size: int = 400
    epvd_cnn_size: int = 128
    epvd_filter_size: int = 3
    epvd_d_size: int = 128
    epvd_micro_batch_size: int = 4
    epvd_eval_batch_size: int = 8
    epvd_path_cache: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "HumerConfig":
        config_path = Path(path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Configuration must be a mapping: {config_path}")
        payload.update({key: value for key, value in (overrides or {}).items() if value is not None})
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        self.model = self.model.lower()
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model {self.model!r}; choose one of {SUPPORTED_MODELS}.")
        self.model_name = self.model_name or DEFAULT_MODEL_NAMES[self.model]
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2.")
        if self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be positive.")
        if self.max_length < 8:
            raise ValueError("max_length must be at least 8.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.num_buckets < 2:
            raise ValueError("num_buckets must be at least 2.")
        if self.consolidation_max_epochs < 1:
            raise ValueError("Final knowledge consolidation cannot be disabled.")
        if self.consolidation_patience < 1:
            raise ValueError("consolidation_patience must be positive.")
        if not 0.0 < self.decision_threshold < 1.0:
            raise ValueError("decision_threshold must be between 0 and 1.")
        if self.model == "epvd":
            if self.epvd_filter_size != 3:
                raise ValueError("The released EPVD adapter requires exactly three execution paths.")
            if self.batch_size % self.epvd_micro_batch_size:
                raise ValueError("EPVD batch_size must be divisible by epvd_micro_batch_size.")
            self.eval_batch_size = self.epvd_eval_batch_size
            if self.epvd_path_cache is None:
                dataset = Path(self.data_dir).name.lower()
                self.epvd_path_cache = f"outputs/epvd_path_cache/{dataset}_three_paths.sqlite3"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
