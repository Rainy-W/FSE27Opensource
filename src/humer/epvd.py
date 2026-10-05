"""EPVD detector and syntax-path tokenization adapter.

The architecture and three-path extraction follow the official EPVD artifact
vendored under ``third_party/epvd_official``.  The adapter flattens the three
fixed-length path sequences so it can use Humer's existing dataset/collator.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sqlite3
import threading
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


_ROOT = Path(__file__).resolve().parents[2]
_EPVD_ROOT = _ROOT / "third_party" / "epvd_official" / "EPVD"
_CFG_PATH = _EPVD_ROOT / "code" / "c_cfg.py"
_LANGUAGE_LIBRARY = _EPVD_ROOT / "parserTool" / "my-languages.so"


def _load_official_cfg_class():
    if not _CFG_PATH.exists():
        raise FileNotFoundError(
            "EPVD support files are missing. Follow the EPVD setup section in README.md "
            "and run scripts/prepare_epvd.py first."
        )
    spec = importlib.util.spec_from_file_location("humer_epvd_official_c_cfg", _CFG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official EPVD CFG module: {_CFG_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.C_CFG


def remove_c_comments(code: str) -> tuple[str, dict[int, str]]:
    """Match the official EPVD C comment removal and line renumbering."""

    def replacer(match: re.Match[str]) -> str:
        value = match.group(0)
        return " " if value.startswith("/") else value

    pattern = re.compile(
        r"//.*?$|/\*.*?\*/|'(?:\\.|[^\\'])*'|\"(?:\\.|[^\\\"])*\"",
        re.DOTALL | re.MULTILINE,
    )
    lines: list[str] = []
    line_map: dict[int, str] = {}
    for source_line in re.sub(pattern, replacer, code).split("\n"):
        if source_line.strip():
            lines.append(source_line)
            line_map[len(lines)] = source_line
    return "\n".join(lines), line_map


class EpvdPathExtractor:
    """Extract EPVD's two shortest/coverage paths plus one coverage path."""

    def __init__(self) -> None:
        try:
            from tree_sitter import Language, Parser
        except ImportError as exc:  # pragma: no cover
            raise ImportError("EPVD requires tree-sitter==0.20.0.") from exc
        if not _LANGUAGE_LIBRARY.exists():
            raise FileNotFoundError(f"Missing EPVD language library: {_LANGUAGE_LIBRARY}")
        self.cfg_class = _load_official_cfg_class()
        self.parser = Parser()
        self.parser.set_language(Language(str(_LANGUAGE_LIBRARY), "c"))

    def extract(self, code: str) -> tuple[list[str], str, str]:
        clean_code, source_lines = remove_c_comments(code)
        if not clean_code:
            return [code, code, code], "fallback", "empty_after_comment_removal"
        try:
            syntax_tree = self.parser.parse(clean_code.encode("utf-8"))
            cfg = self.cfg_class()
            cfg.parse_ast_file(syntax_tree.root_node)
            _, path_sequences, _, _ = cfg.get_allpath()
            paths = [
                "".join(source_lines[line] for line in path if line != "exit" and line in source_lines)
                for path in path_sequences
            ]
            paths = sorted(paths, key=len)
            if not paths:
                return [clean_code, clean_code, clean_code], "fallback", "no_cfg_paths"
            while len(paths) < 3:
                paths.append(clean_code)
            return paths[:3], "ok", ""
        except Exception as exc:  # preserve split membership on unsupported snippets
            reason = f"{type(exc).__name__}: {exc}"[:1000]
            return [clean_code, clean_code, clean_code], "fallback", reason


class EpvdPathCache:
    """Persistent deterministic cache keyed by the complete source text hash."""

    def __init__(self, path: str | Path, extractor: EpvdPathExtractor) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.extractor = extractor
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS paths (
                code_sha256 TEXT PRIMARY KEY,
                paths_json TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    @staticmethod
    def key(code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def get(self, code: str) -> tuple[list[str], str, str, bool]:
        key = self.key(code)
        with self._lock:
            row = self._connection.execute(
                "SELECT paths_json, status, error FROM paths WHERE code_sha256 = ?", (key,)
            ).fetchone()
            if row is not None:
                return list(json.loads(row[0])), str(row[1]), str(row[2]), True
            paths, status, error = self.extractor.extract(code)
            self._connection.execute(
                "INSERT INTO paths(code_sha256, paths_json, status, error) VALUES (?, ?, ?, ?)",
                (key, json.dumps(paths, ensure_ascii=False), status, error),
            )
            self._connection.commit()
            return paths, status, error, False

    def failures(self) -> list[dict[str, str]]:
        rows = self._connection.execute(
            "SELECT code_sha256, status, error FROM paths WHERE status != 'ok' ORDER BY code_sha256"
        ).fetchall()
        return [{"code_sha256": row[0], "status": row[1], "error": row[2]} for row in rows]


class EpvdPathTokenizer:
    """Tokenize EPVD's three execution paths into one flattened tensor."""

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        block_size: int = 400,
        cache_path: str | Path = "outputs/epvd_path_cache/paths.sqlite3",
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        base_tokenizer=None,
        extractor: EpvdPathExtractor | None = None,
    ) -> None:
        if base_tokenizer is None:
            try:
                from transformers import RobertaTokenizer
            except ImportError as exc:  # pragma: no cover
                raise ImportError("EPVD requires transformers.") from exc
            base_tokenizer = RobertaTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        self.tokenizer = base_tokenizer
        self.block_size = int(block_size)
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.path_cache = EpvdPathCache(cache_path, extractor or EpvdPathExtractor())

    def prepare(self, samples: Iterable[object]) -> dict[str, int]:
        stats: Counter[str] = Counter()
        for sample in samples:
            code = str(getattr(sample, "code", sample))
            _, status, _, cached = self.path_cache.get(code)
            stats[status] += 1
            stats["cache_hit" if cached else "cache_miss"] += 1
            stats["total"] += 1
        return dict(stats)

    def write_failures(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.path_cache.failures(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 512,
        padding: bool = False,
        return_attention_mask: bool = True,
    ) -> dict[str, list[int]]:
        del truncation, max_length, padding
        paths, _, _, _ = self.path_cache.get(text)
        all_ids: list[int] = []
        all_masks: list[int] = []
        for path in paths[:3]:
            tokens = self.tokenizer.tokenize(path)[: self.block_size - 2]
            tokens = [self.tokenizer.cls_token, *tokens, self.tokenizer.sep_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            mask = [1] * len(token_ids)
            padding_length = self.block_size - len(token_ids)
            token_ids.extend([self.pad_token_id] * padding_length)
            mask.extend([0] * padding_length)
            all_ids.extend(token_ids)
            all_masks.extend(mask)
        result = {"input_ids": all_ids}
        if return_attention_mask:
            result["attention_mask"] = all_masks
        return result


class EpvdConv1d(nn.Module if nn is not None else object):
    def __init__(self, in_channels: int, out_channels: int, filter_sizes: list[int]) -> None:
        if nn is None:
            raise ImportError("EPVD requires PyTorch.")
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels, out_channels, kernel_size=size) for size in filter_sizes]
        )
        for layer in self.convs:
            nn.init.xavier_uniform_(layer.weight.data)
            nn.init.constant_(layer.bias.data, 0.1)

    def forward(self, inputs):
        return [F.relu(layer(inputs)) for layer in self.convs]


class EpvdTextCnn(nn.Module if nn is not None else object):
    def __init__(self, hidden_size: int, cnn_size: int, filter_size: int, d_size: int) -> None:
        if nn is None:
            raise ImportError("EPVD requires PyTorch.")
        super().__init__()
        self.convs = EpvdConv1d(hidden_size, cnn_size, list(range(1, filter_size + 1)))
        self.fc = nn.Linear(filter_size * cnn_size, d_size)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs):
        convolved = self.convs(inputs.permute(0, 2, 1))
        pooled = [F.max_pool1d(item, item.shape[2]).squeeze(2) for item in convolved]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class EpvdClassificationHead(nn.Module if nn is not None else object):
    """Official EPVD CNN path-fusion head."""

    def __init__(self, config, cnn_size: int = 128, filter_size: int = 3, d_size: int = 128) -> None:
        if nn is None:
            raise ImportError("EPVD requires PyTorch.")
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.filter_size = int(filter_size)
        self.dense = nn.Linear(2 * d_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, 1)
        self.W_w = nn.Parameter(torch.empty(2 * hidden_size, 2 * hidden_size))
        self.u_w = nn.Parameter(torch.empty(2 * hidden_size, 1))
        self.linear = nn.Linear(filter_size * hidden_size, d_size)
        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            3,
            bidirectional=True,
            batch_first=True,
            dropout=config.hidden_dropout_prob,
        )
        self.cnn = EpvdTextCnn(hidden_size, cnn_size, filter_size, d_size)
        self.linear_mlp = nn.Linear(6 * hidden_size, d_size)
        self.linear_multi = nn.Linear(hidden_size, hidden_size)

    def forward(self, flattened_path_features):
        path_features = flattened_path_features.reshape(
            flattened_path_features.shape[0], self.filter_size, -1
        )
        cnn_features = self.cnn(path_features)
        linear_features = self.linear(flattened_path_features)
        hidden = self.dropout(torch.cat((cnn_features, linear_features), dim=-1))
        hidden = torch.tanh(self.dense(hidden))
        return self.out_proj(self.dropout(hidden))


class EpvdClassifier(nn.Module if nn is not None else object):
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        block_size: int = 400,
        cnn_size: int = 128,
        filter_size: int = 3,
        d_size: int = 128,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        if nn is None:
            raise ImportError("EPVD requires PyTorch.")
        try:
            from transformers import RobertaConfig, RobertaModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError("EPVD requires transformers.") from exc
        super().__init__()
        self.config = RobertaConfig.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.encoder = RobertaModel.from_pretrained(
            model_name,
            config=self.config,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.block_size = int(block_size)
        self.filter_size = int(filter_size)
        self.pad_token_id = int(self.config.pad_token_id)
        self.linear = nn.Linear(filter_size, 1)  # Present in the official model.
        self.cnnclassifier = EpvdClassificationHead(
            self.config,
            cnn_size=cnn_size,
            filter_size=filter_size,
            d_size=d_size,
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        expected = self.filter_size * self.block_size
        if input_ids.shape[1] != expected:
            raise ValueError(f"EPVD expects {expected} flattened path tokens, got {input_ids.shape[1]}.")
        path_ids = input_ids.reshape(-1, self.filter_size, self.block_size)
        flat_ids = path_ids.reshape(-1, self.block_size)
        flat_mask = flat_ids.ne(self.pad_token_id)
        encoded = self.encoder(input_ids=flat_ids, attention_mask=flat_mask, return_dict=True)
        cls_features = encoded.last_hidden_state[:, 0, :]
        flattened_features = cls_features.reshape(input_ids.shape[0], -1)
        vulnerable_logit = self.cnnclassifier(flattened_features)
        logits = torch.cat((torch.zeros_like(vulnerable_logit), vulnerable_logit), dim=-1)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits, labels)
        return result
