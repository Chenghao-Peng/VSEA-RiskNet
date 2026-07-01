"""General utilities, metrics, and reproducibility helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """Move a tensor batch to the target device."""
    return {key: value.to(device) for key, value in batch.items()}


def safe_div(numerator: float, denominator: float) -> float:
    """Divide safely with zero-denominator protection."""
    return float(numerator) / float(denominator) if denominator else 0.0


def macro_f1_score(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> float:
    """Compute macro F1 without external dependencies."""
    scores: List[float] = []
    for cls in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        scores.append(safe_div(2 * precision * recall, precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def binary_f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute binary F1 for risk-vs-normal detection."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return safe_div(2 * precision * recall, precision + recall)


def binary_auc_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Compute binary AUC using the rank-sum formulation."""
    positives = [(s, t) for s, t in zip(y_score, y_true) if t == 1]
    negatives = [(s, t) for s, t in zip(y_score, y_true) if t == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / total


def evidence_f1_score(
    predicted: Iterable[int], targets: Iterable[int]
) -> float:
    """Compute F1 over flattened evidence predictions."""
    pred_list = list(predicted)
    target_list = list(targets)
    tp = sum(1 for p, t in zip(pred_list, target_list) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred_list, target_list) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred_list, target_list) if p == 0 and t == 1)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return safe_div(2 * precision * recall, precision + recall)


def compute_metrics(
    labels: Sequence[int],
    probabilities: np.ndarray,
    node_pred: Sequence[int],
    node_true: Sequence[int],
    edge_pred: Sequence[int],
    edge_true: Sequence[int],
    num_classes: int,
) -> Dict[str, float]:
    """Compute document and evidence metrics."""
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    accuracy = float((predictions == labels_array).mean()) if len(labels_array) else 0.0
    macro_f1 = macro_f1_score(labels_array.tolist(), predictions.tolist(), num_classes)
    risk_true = (labels_array != 0).astype(np.int64).tolist()
    risk_pred = (predictions != 0).astype(np.int64).tolist()
    risk_f1 = binary_f1_score(risk_true, risk_pred)
    risk_score = 1.0 - probabilities[:, 0] if probabilities.size else np.asarray([])
    auc = binary_auc_score(risk_true, risk_score.tolist()) if len(risk_true) else 0.5
    evidence_pred_all = list(node_pred) + list(edge_pred)
    evidence_true_all = list(node_true) + list(edge_true)
    evidence_f1 = evidence_f1_score(evidence_pred_all, evidence_true_all)
    return {
        "Accuracy": accuracy * 100.0,
        "Macro-F1": macro_f1 * 100.0,
        "Risk-F1": risk_f1 * 100.0,
        "AUC": auc * 100.0,
        "Evidence-F1": evidence_f1 * 100.0,
    }


class AverageMeter:
    """Track running averages for scalar losses."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        return safe_div(self.total, self.count)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save a JSON file with pretty formatting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
