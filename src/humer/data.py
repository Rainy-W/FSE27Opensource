"""Dataset and batching utilities for HUMER."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence


try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - exercised only in minimal envs
    torch = None
    Dataset = object


TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|0x[0-9A-Fa-f]+|\d+\.\d+|\d+|==|!=|<=|>=|&&|\|\||"
    r"->|\+\+|--|<<|>>|[-+*/%=&|^~!<>?:;,.()\[\]{}]"
)


@dataclass(frozen=True)
class CodeSample:
    """A single vulnerability-detection sample."""

    code: str
    label: int
    sample_id: int
    difficulty: Optional[float] = None


def simple_code_tokenize(code: str) -> List[str]:
    """Tokenize code with a deterministic regex tokenizer."""

    return TOKEN_RE.findall(code)


class JsonCodeDataset(Dataset):
    """Load Devign/ReVeal/BigVul style JSON data.

    The expected file format is a JSON list whose items contain at least
    ``code`` and ``label`` fields.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: Optional[Callable[[str], Sequence[str]]] = None,
        max_length: int = 512,
        vocab: Optional[dict[str, int]] = None,
        build_vocab: bool = False,
        min_freq: int = 1,
    ) -> None:
        if torch is None:
            raise ImportError("JsonCodeDataset requires PyTorch. Install torch first.")

        self.path = Path(path)
        self.tokenizer = tokenizer or simple_code_tokenize
        self.max_length = max_length
        self.samples = self._load_samples(self.path)

        if build_vocab:
            self.vocab = self.build_vocab(self.samples, self.tokenizer, min_freq=min_freq)
        elif vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = {"<pad>": 0, "<unk>": 1}

    @staticmethod
    def _load_samples(path: Path) -> List[CodeSample]:
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)

        samples = []
        for idx, row in enumerate(rows):
            samples.append(
                CodeSample(
                    code=str(row["code"]),
                    label=int(row["label"]),
                    sample_id=int(row.get("id", idx)),
                )
            )
        return samples

    @staticmethod
    def build_vocab(
        samples: Sequence[CodeSample],
        tokenizer: Callable[[str], Sequence[str]],
        min_freq: int = 1,
    ) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for sample in samples:
            counter.update(tokenizer(sample.code))

        vocab = {"<pad>": 0, "<unk>": 1}
        for token, freq in counter.most_common():
            if freq >= min_freq and token not in vocab:
                vocab[token] = len(vocab)
        return vocab

    def subset(self, indices: Sequence[int]) -> "InMemoryCodeDataset":
        return InMemoryCodeDataset(
            [self.samples[i] for i in indices],
            vocab=self.vocab,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def with_samples(self, samples: Sequence[CodeSample]) -> "InMemoryCodeDataset":
        return InMemoryCodeDataset(
            samples,
            vocab=self.vocab,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return encode_sample(self.samples[index], self.vocab, self.tokenizer, self.max_length)


class InMemoryCodeDataset(Dataset):
    """Dataset backed by an existing list of ``CodeSample`` objects."""

    def __init__(
        self,
        samples: Sequence[CodeSample],
        vocab: dict[str, int],
        tokenizer: Callable[[str], Sequence[str]] = simple_code_tokenize,
        max_length: int = 512,
    ) -> None:
        if torch is None:
            raise ImportError("InMemoryCodeDataset requires PyTorch. Install torch first.")

        self.samples = list(samples)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length

    def subset(self, indices: Sequence[int]) -> "InMemoryCodeDataset":
        return InMemoryCodeDataset(
            [self.samples[i] for i in indices],
            vocab=self.vocab,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def with_samples(self, samples: Sequence[CodeSample]) -> "InMemoryCodeDataset":
        return InMemoryCodeDataset(
            samples,
            vocab=self.vocab,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return encode_sample(self.samples[index], self.vocab, self.tokenizer, self.max_length)


class HuggingFaceCodeDataset(Dataset):
    """Dataset that encodes code with a HuggingFace tokenizer.

    It keeps the same ``samples`` surface as ``JsonCodeDataset`` so difficulty
    calculators, curriculum slicing, and augmentation can continue to work on
    raw source code while the model receives pretrained-tokenizer IDs.
    """

    def __init__(
        self,
        path_or_samples: str | Path | Sequence[CodeSample],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        if torch is None:
            raise ImportError("HuggingFaceCodeDataset requires PyTorch. Install torch first.")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = {"<hf-pad>": int(getattr(tokenizer, "pad_token_id", 0) or 0)}
        self.pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

        if isinstance(path_or_samples, (str, Path)):
            self.path = Path(path_or_samples)
            self.samples = JsonCodeDataset._load_samples(self.path)
        else:
            self.path = None
            self.samples = list(path_or_samples)

    def subset(self, indices: Sequence[int]) -> "HuggingFaceCodeDataset":
        return HuggingFaceCodeDataset(
            [self.samples[i] for i in indices],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def with_samples(self, samples: Sequence[CodeSample]) -> "HuggingFaceCodeDataset":
        return HuggingFaceCodeDataset(
            samples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        encoded = self.tokenizer(
            sample.code,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.float32),
            "label": torch.tensor(sample.label, dtype=torch.long),
            "sample_id": sample.sample_id,
            "code": sample.code,
            "difficulty": sample.difficulty,
            "pad_token_id": self.pad_token_id,
        }


def encode_sample(
    sample: CodeSample,
    vocab: dict[str, int],
    tokenizer: Callable[[str], Sequence[str]],
    max_length: int,
) -> dict:
    tokens = list(tokenizer(sample.code))[:max_length]
    ids = [vocab.get(token, vocab.get("<unk>", 1)) for token in tokens]
    if not ids:
        ids = [vocab.get("<unk>", 1)]

    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "label": torch.tensor(sample.label, dtype=torch.long),
        "sample_id": sample.sample_id,
        "code": sample.code,
        "difficulty": sample.difficulty,
    }


def collate_code_batch(batch: Sequence[dict]) -> dict:
    """Pad variable-length token IDs into a batch."""

    if torch is None:
        raise ImportError("collate_code_batch requires PyTorch. Install torch first.")

    lengths = torch.tensor([len(item["input_ids"]) for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    pad_token_id = int(batch[0].get("pad_token_id", 0))
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.float32)

    for i, item in enumerate(batch):
        length = len(item["input_ids"])
        input_ids[i, :length] = item["input_ids"]
        if "attention_mask" in item:
            attention_mask[i, :length] = item["attention_mask"].to(dtype=torch.float32)
        else:
            attention_mask[i, :length] = 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.stack([item["label"] for item in batch]),
        "sample_ids": [item["sample_id"] for item in batch],
        "codes": [item["code"] for item in batch],
    }


def iter_raw_samples(dataset: object) -> Iterator[CodeSample]:
    """Return raw samples from supported dataset wrappers."""

    if hasattr(dataset, "samples"):
        yield from getattr(dataset, "samples")
        return

    for idx in range(len(dataset)):  # type: ignore[arg-type]
        item = dataset[idx]  # type: ignore[index]
        yield CodeSample(
            code=str(item["code"]),
            label=int(item["label"]),
            sample_id=int(item.get("sample_id", idx)),
            difficulty=item.get("difficulty"),
        )
