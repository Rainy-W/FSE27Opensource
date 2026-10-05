"""Code-based difficulty scoring and label-stratified curriculum construction."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .data import CodeSample


CODE_OPERATORS = (
    "!=", "!", "%=", "%", "&&", "&=", "&", "||", "|=", "|", "(", ")",
    "*=", "*", "++", "+=", "+", "--", "-=", "->", "-", "...", ".", "/=",
    "/", "::", ":", "<<=", "<<", "<=", "<", "==", "=", ">>=", ">=", ">>",
    ">", "?", "[", "]", "^=", "^", "{", "}", "~", ",", ";",
)
STRING_LITERAL_RE = re.compile(r'"(?:[^"]|\\")*[^\\]"')


@dataclass(frozen=True)
class DifficultyRecord:
    sample: CodeSample
    score: float


class CodeDifficultyCalculator:
    """Compute the negative Maintainability Index used by HUMER."""

    def __init__(self, eps: float = 1e-5) -> None:
        self.eps = eps

    def score(self, code: str) -> float:
        code = self._purify(code)
        loc = max(1, sum(1 for line in code.splitlines() if len(line.strip()) > 1))
        complexity = self._cyclomatic_complexity(code)
        volume = self._halstead_volume(code)
        maintainability = (
            171.0
            - 5.2 * math.log(volume + self.eps)
            - 0.23 * complexity
            - 16.2 * math.log(loc + self.eps)
        )
        return -maintainability

    def calculate(self, samples: Sequence[CodeSample]) -> list[DifficultyRecord]:
        return [DifficultyRecord(sample=sample, score=self.score(sample.code)) for sample in samples]

    @staticmethod
    def _purify(code: str) -> str:
        code = re.sub(r"/\*[\w\W]*?\*/", "", code)
        code = re.sub(r"//.*?\n", "\n", code)
        code = re.sub(r"[^\x00-\x7F]+", "", code)
        code = re.sub(r"^#.*", "", code, flags=re.MULTILINE)
        return "\n".join(line.strip() for line in code.splitlines() if line.strip())

    @staticmethod
    def _cyclomatic_complexity(code: str) -> int:
        paths = 1
        for line in code.splitlines():
            paths += int(re.search(r"if", line) is not None)
            paths += int(re.search(r"for", line) is not None)
            paths += int(re.search(r"while", line) is not None)
            paths += int(re.search(r"case", line) is not None)
            paths += int(re.search(r"catch", line) is not None)
            paths += int("&&" in line) + int("||" in line) + int("else" in line) + int("?" in line)
            paths -= int("else if" in line)
        return paths

    @staticmethod
    def _halstead_volume(code: str) -> float:
        operators: dict[str, int] = {}
        operands: dict[str, int] = {}
        for original_line in code.splitlines():
            line = original_line
            for literal in STRING_LITERAL_RE.findall(line):
                if literal != " ":
                    operands[literal] = operands.get(literal, 0) + 1
            line = re.sub(STRING_LITERAL_RE, " ", line)
            for operator in CODE_OPERATORS:
                operators[operator] = operators.get(operator, 0) + line.count(operator)
                line = line.replace(operator, " ")
            for token in line.split():
                operands[token] = operands.get(token, 0) + 1
        distinct_operators = sum(1 for op, count in operators.items() if count > 0 and op not in ")}]")
        total_operators = sum(count for op, count in operators.items() if count > 0 and op not in ")}]")
        distinct_operands = sum(1 for count in operands.values() if count > 0)
        total_operands = sum(count for count in operands.values() if count > 0)
        total = total_operators + total_operands
        vocabulary = distinct_operators + distinct_operands
        return 1.0 if total == 0 or vocabulary <= 1 else float(total) * math.log2(float(vocabulary))


def curriculum_buckets(records: Sequence[DifficultyRecord], num_buckets: int) -> list[list[DifficultyRecord]]:
    """Split each class by difficulty and merge equal-level class buckets."""

    by_label: dict[int, list[DifficultyRecord]] = {}
    for record in records:
        by_label.setdefault(int(record.sample.label), []).append(record)
    buckets: list[list[DifficultyRecord]] = [[] for _ in range(num_buckets)]
    for label in sorted(by_label):
        ordered = sorted(by_label[label], key=lambda item: (item.score, item.sample.sample_id))
        for bucket_id, indices in enumerate(split_indices(len(ordered), num_buckets)):
            buckets[bucket_id].extend(ordered[index] for index in indices)
    for bucket in buckets:
        bucket.sort(key=lambda item: (item.score, item.sample.label, item.sample.sample_id))
    return buckets


def split_indices(length: int, parts: int) -> list[list[int]]:
    base, remainder = divmod(length, parts)
    result: list[list[int]] = []
    start = 0
    for part in range(parts):
        width = base + (1 if part < remainder else 0)
        result.append(list(range(start, start + width)))
        start += width
    return result


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_compute_difficulty(
    samples: Sequence[CodeSample], train_path: str | Path, cache_dir: str | Path
) -> tuple[list[DifficultyRecord], Path, str]:
    """Load an exact dataset cache or compute and persist every score."""

    train_path = Path(train_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = file_sha256(train_path)
    cache_path = cache_dir / f"{train_path.parent.name}_code_{fingerprint[:16]}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        scores = {int(row["sample_id"]): float(row["score"]) for row in payload["scores"]}
        sample_ids = {sample.sample_id for sample in samples}
        if set(scores) != sample_ids:
            raise ValueError(f"Difficulty cache sample IDs do not match {train_path}.")
        records = [
            DifficultyRecord(sample=sample, score=scores[sample.sample_id])
            for sample in samples
        ]
        return records, cache_path, "loaded"
    records = CodeDifficultyCalculator().calculate(samples)
    payload = {
        "version": 1,
        "difficulty": "code",
        "train_file": str(train_path.resolve()),
        "train_sha256": fingerprint,
        "scores": [
            {"sample_id": item.sample.sample_id, "label": item.sample.label, "score": item.score}
            for item in records
        ],
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return records, cache_path, "computed"
