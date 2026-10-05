"""Model implementations and factories supported by HUMER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("This module requires PyTorch. Install torch first.")


class CodeBERTClassifier(nn.Module if nn is not None else object):
    """CodeBERT sequence classifier wrapper.

    This class is optional. It requires ``transformers`` and local or network
    access to the requested pretrained checkpoint.
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_labels: int = 2,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        require_torch()
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError as exc:  # pragma: no cover
            raise ImportError("CodeBERTClassifier requires transformers.") from exc

        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


class SimpleFCHead(nn.Module if nn is not None else object):
    """SPLVD-style classification head over pooled encoder embeddings."""

    def __init__(self, input_dim: int, num_labels: int, dropout: float = 0.2) -> None:
        require_torch()
        super().__init__()
        self.dense = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class EncoderClassifier(nn.Module if nn is not None else object):
    """Classifier for encoder-only pretrained code models.

    Supports CodeT5 via ``T5EncoderModel`` and UniXcoder via the encoder-only
    forward pass used by SPLVD. The public forward returns the same dict shape
    as the existing bag model so Humer's trainer, baseline, and difficulty
    calculators can use it without a separate training loop.
    """

    def __init__(
        self,
        model_type: str,
        model_name: str,
        num_labels: int = 2,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        require_torch()
        super().__init__()
        self.model_type = model_type.lower()
        if self.model_type == "codet5":
            try:
                from transformers import T5EncoderModel
            except ImportError as exc:  # pragma: no cover
                raise ImportError("CodeT5 support requires transformers.") from exc
            self.encoder = T5EncoderModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            hidden_size = int(self.encoder.config.d_model)
        elif self.model_type == "unixcoder":
            self.encoder = UniXcoderEncoder(
                model_name=model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            hidden_size = int(self.encoder.config.hidden_size)
        else:
            raise ValueError(f"Unsupported encoder model_type: {model_type}")
        self.head = SimpleFCHead(hidden_size, num_labels)

    def extract_embeddings(self, input_ids, attention_mask=None):
        if self.model_type == "codet5":
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeddings = outputs.last_hidden_state
            if attention_mask is None:
                attention_mask = input_ids.ne(getattr(self.encoder.config, "pad_token_id", 0)).float()
            mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
            return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.encoder(input_ids)

    def forward(self, input_ids, attention_mask=None, labels=None):
        pooled = self.extract_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.head(pooled)
        if labels is None:
            return {"logits": logits}
        loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


class UniXcoderEncoder(nn.Module if nn is not None else object):
    """Minimal UniXcoder encoder used for classification embeddings."""

    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        require_torch()
        try:
            from transformers import RobertaConfig, RobertaModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError("UniXcoder support requires transformers.") from exc
        super().__init__()
        self.config = RobertaConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.config.is_decoder = True
        self.model = RobertaModel.from_pretrained(
            model_name,
            config=self.config,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    def forward(self, input_ids):
        mask = input_ids.ne(self.config.pad_token_id)
        token_embeddings = self.model(
            input_ids,
            attention_mask=mask.unsqueeze(1) * mask.unsqueeze(2),
            return_dict=True,
        ).last_hidden_state
        return (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1).clamp_min(1.0)


class UniXcoderTokenizerWrapper:
    """Callable tokenizer that mirrors UniXcoder's ``<encoder-only>`` mode."""

    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        try:
            from transformers import RobertaConfig, RobertaTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("UniXcoder tokenization requires transformers.") from exc
        self.tokenizer = RobertaTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.config = RobertaConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.pad_token_id = int(self.config.pad_token_id)
        self.tokenizer.add_tokens(["<mask0>"], special_tokens=True)

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 512,
        padding: bool = False,
        return_attention_mask: bool = True,
    ) -> dict[str, list[int]]:
        if max_length >= 1024:
            raise ValueError("UniXcoder max_length must be less than 1024.")
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[: max_length - 4] if truncation else tokens
        tokens = [self.tokenizer.cls_token, "<encoder-only>", self.tokenizer.sep_token] + tokens
        tokens = tokens[: max_length - 1] + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if padding:
            input_ids = input_ids + [self.pad_token_id] * max(0, max_length - len(input_ids))
        result = {"input_ids": input_ids}
        if return_attention_mask:
            result["attention_mask"] = [1] * len(input_ids)
        return result


def build_pretrained_tokenizer(
    model_type: str,
    model_name: str,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
):
    model_type = model_type.lower()
    if model_type == "unixcoder":
        return UniXcoderTokenizerWrapper(
            model_name=model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(f"{model_type} training requires transformers.") from exc
    return AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        use_fast=True,
    )


@dataclass
class ModelFactory:
    """Callable model factory used by difficulty calculators and trainer."""

    builder: Callable[[], object]

    def create(self):
        return self.builder()


def build_codebert_factory(
    model_name: str = "microsoft/codebert-base",
    num_labels: int = 2,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> ModelFactory:
    return ModelFactory(
        lambda: CodeBERTClassifier(
            model_name=model_name,
            num_labels=num_labels,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )


def build_linevul_factory(
    model_name: str = "microsoft/codebert-base",
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> ModelFactory:
    from .related_detectors import LineVulClassifier

    return ModelFactory(
        lambda: LineVulClassifier(
            model_name=model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )


def build_epvd_factory(
    model_name: str = "microsoft/codebert-base",
    block_size: int = 400,
    cnn_size: int = 128,
    filter_size: int = 3,
    d_size: int = 128,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> ModelFactory:
    from .epvd import EpvdClassifier

    return ModelFactory(
        lambda: EpvdClassifier(
            model_name=model_name,
            block_size=block_size,
            cnn_size=cnn_size,
            filter_size=filter_size,
            d_size=d_size,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )


def build_encoder_model_factory(
    model_type: str,
    model_name: str,
    num_labels: int = 2,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> ModelFactory:
    return ModelFactory(
        lambda: EncoderClassifier(
            model_type=model_type,
            model_name=model_name,
            num_labels=num_labels,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )
