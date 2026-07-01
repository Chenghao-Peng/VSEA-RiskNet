"""Cross-validation and revision-cycle runner for VSEA-RiskNet."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import CrossValConfig, ModelConfig, TrainConfig
from .data import generate_synthetic_samples, load_jsonl, write_jsonl
from .train import fit_model
from .utils import save_json, set_seed


def stratified_kfold_indices(labels: Sequence[int], folds: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    """Create deterministic stratified k-fold indices."""
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(index)
    fold_buckets: List[List[int]] = [[] for _ in range(folds)]
    for indices in by_class.values():
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            fold_buckets[offset % folds].append(index)
    splits: List[Tuple[List[int], List[int]]] = []
    all_indices = set(range(len(labels)))
    for fold_id in range(folds):
        valid_indices = sorted(fold_buckets[fold_id])
        train_indices = sorted(all_indices.difference(valid_indices))
        splits.append((train_indices, valid_indices))
    return splits


def summarize_fold_metrics(fold_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Compute mean and standard deviation across folds."""
    if not fold_metrics:
        return {}
    keys = [key for key in fold_metrics[0].keys() if isinstance(fold_metrics[0][key], (float, int))]
    summary: Dict[str, float] = {}
    for key in keys:
        values = np.asarray([metrics[key] for metrics in fold_metrics], dtype=np.float32)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    return summary


def run_cross_validation(
    samples: Sequence[dict],
    model_config: ModelConfig,
    train_config: TrainConfig,
    cv_config: CrossValConfig,
) -> Dict[str, object]:
    """Run stratified k-fold cross-validation."""
    set_seed(train_config.seed)
    labels = [int(sample.get("label", sample.get("risk_label", 0))) for sample in samples]
    splits = stratified_kfold_indices(labels, cv_config.folds, train_config.seed)
    fold_metrics: List[Dict[str, float]] = []
    for fold_id, (train_indices, valid_indices) in enumerate(splits, start=1):
        fold_train_config = copy.deepcopy(train_config)
        fold_train_config.seed = train_config.seed + fold_id
        train_samples = [samples[index] for index in train_indices]
        valid_samples = [samples[index] for index in valid_indices]
        _, metrics = fit_model(train_samples, valid_samples, model_config, fold_train_config)
        metrics = {**metrics, "fold": float(fold_id)}
        fold_metrics.append(metrics)
    summary = summarize_fold_metrics(fold_metrics)
    return {"folds": fold_metrics, "summary": summary}


def revision_presets(base_model: ModelConfig, base_train: TrainConfig) -> List[Tuple[str, ModelConfig, TrainConfig]]:
    """Return three conservative revision presets inspired by validation feedback."""
    presets: List[Tuple[str, ModelConfig, TrainConfig]] = []

    first_model = copy.deepcopy(base_model)
    first_train = copy.deepcopy(base_train)
    first_model.hidden_dim = min(base_model.hidden_dim, 48)
    first_model.dropout = 0.10
    first_model.edge_threshold = 0.55
    first_model.relation_risk_alpha = 1.0
    first_train.evidence_loss_weight = 0.30
    first_train.relation_loss_weight = 0.20
    presets.append(("cycle_1_light_baseline", first_model, first_train))

    second_model = copy.deepcopy(base_model)
    second_train = copy.deepcopy(base_train)
    second_model.hidden_dim = max(base_model.hidden_dim, 64)
    second_model.dropout = 0.15
    second_model.top_k = 6
    second_model.edge_threshold = 0.55
    second_model.relation_risk_alpha = 1.0
    second_train.evidence_loss_weight = 0.50
    second_train.relation_loss_weight = 0.30
    presets.append(("cycle_2_evidence_balanced", second_model, second_train))

    third_model = copy.deepcopy(base_model)
    third_train = copy.deepcopy(base_train)
    third_model.hidden_dim = max(base_model.hidden_dim, 64)
    third_model.dropout = 0.20
    third_model.edge_threshold = 0.50
    third_model.relation_risk_alpha = 1.25
    third_train.evidence_loss_weight = 0.60
    third_train.relation_loss_weight = 0.35
    presets.append(("cycle_3_relation_refined", third_model, third_train))
    return presets


def run_three_revision_cycles(
    samples: Sequence[dict],
    base_model: ModelConfig,
    base_train: TrainConfig,
    cv_config: CrossValConfig,
) -> Dict[str, object]:
    """Run three cross-validation cycles and select the best preset."""
    cycle_results: List[Dict[str, object]] = []
    best_name = ""
    best_score = -1.0
    for name, model_config, train_config in revision_presets(base_model, base_train):
        result = run_cross_validation(samples, model_config, train_config, cv_config)
        summary = result["summary"]
        score = float(summary.get("Macro-F1_mean", 0.0)) + 0.25 * float(summary.get("Evidence-F1_mean", 0.0))
        cycle_results.append(
            {
                "name": name,
                "selection_score": score,
                "model_config": model_config.__dict__,
                "train_config": train_config.__dict__,
                "result": result,
            }
        )
        if score > best_score:
            best_score = score
            best_name = name
    return {"best_cycle": best_name, "best_selection_score": best_score, "cycles": cycle_results}


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description="Run VSEA-RiskNet cross-validation.")
    parser.add_argument("--data", type=str, default=None, help="Path to a JSONL bill-risk dataset.")
    parser.add_argument("--output", type=str, default="outputs/crossval", help="Output directory.")
    parser.add_argument("--folds", type=int, default=3, help="Number of CV folds.")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per fold.")
    parser.add_argument("--samples", type=int, default=72, help="Synthetic samples when --data is omitted.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--three-cycles", action="store_true", help="Run three revision presets.")
    return parser


def main() -> None:
    """Command-line entry point."""
    args = build_arg_parser().parse_args()
    model_config = ModelConfig()
    train_config = TrainConfig(seed=args.seed, epochs=args.epochs)
    cv_config = CrossValConfig(folds=args.folds, synthetic_samples=args.samples)
    samples = load_jsonl(args.data) if args.data else generate_synthetic_samples(args.samples, args.seed, model_config.num_classes)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.data is None:
        write_jsonl(samples, output / "synthetic_validation_samples.jsonl")
    if args.three_cycles:
        result = run_three_revision_cycles(samples, model_config, train_config, cv_config)
    else:
        result = run_cross_validation(samples, model_config, train_config, cv_config)
    save_json(result, output / "crossval_results.json")
    print(result)


if __name__ == "__main__":
    main()
