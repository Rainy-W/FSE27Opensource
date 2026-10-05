"""Architecture adapters for vulnerability detectors used in RQ4."""

from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


class LineVulClassificationHead(nn.Module if nn is not None else object):
    """The sentence-level classification head from the LineVul implementation."""

    def __init__(self, config) -> None:
        if nn is None:
            raise ImportError("LineVul requires PyTorch.")
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features):
        hidden = features[:, 0, :]
        hidden = self.dropout(hidden)
        hidden = torch.tanh(self.dense(hidden))
        hidden = self.dropout(hidden)
        return self.out_proj(hidden)


class LineVulClassifier(nn.Module if nn is not None else object):
    """LineVul's CodeBERT encoder and sentence-level vulnerability head.

    Line-level attribution in LineVul is an inference-time explanation step;
    its detector is trained with function labels through this classifier.
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        if nn is None:
            raise ImportError("LineVul requires PyTorch.")
        try:
            from transformers import RobertaConfig, RobertaForSequenceClassification
        except ImportError as exc:  # pragma: no cover
            raise ImportError("LineVul requires transformers.") from exc

        super().__init__()
        config = RobertaConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.encoder = RobertaForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.classifier = LineVulClassificationHead(config)
        self.pad_token_id = int(config.pad_token_id)

    def forward(self, input_ids, attention_mask=None, labels=None):
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        features = self.encoder.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        logits = self.classifier(features)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result
