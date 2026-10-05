"""Binary classification metrics used for validation and final testing."""

from __future__ import annotations

import math


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def binary_metrics(labels: list[int], predictions: list[int], probabilities: list[float]) -> dict[str, float | int]:
    tp = sum(pred == 1 and label == 1 for pred, label in zip(predictions, labels))
    tn = sum(pred == 0 and label == 0 for pred, label in zip(predictions, labels))
    fp = sum(pred == 1 and label == 0 for pred, label in zip(predictions, labels))
    fn = sum(pred == 0 and label == 1 for pred, label in zip(predictions, labels))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    negative_precision = safe_div(tn, tn + fn)
    negative_recall = safe_div(tn, tn + fp)
    negative_f1 = safe_div(2.0 * negative_precision * negative_recall, negative_precision + negative_recall)
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    result: dict[str, float | int] = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": safe_div(tp + tn, len(labels)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": (f1 + negative_f1) / 2.0,
        "balanced_accuracy": (recall + negative_recall) / 2.0,
        "mcc": safe_div(tp * tn - fp * fn, denominator),
        "predicted_positive_rate": safe_div(sum(predictions), len(predictions)),
    }
    if labels:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            result["roc_auc"] = float(roc_auc_score(labels, probabilities))
            result["auprc"] = float(average_precision_score(labels, probabilities))
        except ValueError:
            result["roc_auc"] = 0.0
            result["auprc"] = 0.0
    return result
