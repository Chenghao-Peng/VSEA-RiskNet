"""Training and evaluation routines for VSEA-RiskNet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .data import BillRiskDataset, collate_bill_batch, generate_synthetic_samples, load_jsonl
from .losses import compute_vsea_loss
from .model import VSEARiskNet
from .utils import AverageMeter, compute_metrics, move_to_device, save_json, set_seed


def make_loader(
    samples: Sequence[dict],
    model_config: ModelConfig,
    train_config: TrainConfig,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader for encoded bill samples."""
    dataset = BillRiskDataset(samples, model_config)
    return DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_bill_batch,
    )


def train_one_epoch(
    model: VSEARiskNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> Dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    meters = {name: AverageMeter() for name in ["loss", "doc_loss", "node_loss", "edge_loss", "relation_loss"]}
    for batch in loader:
        batch = move_to_device(batch, train_config.device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        losses = compute_vsea_loss(outputs, batch, model_config, train_config)
        losses["loss"].backward()
        if train_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()
        batch_size = int(batch["labels"].numel())
        for name, meter in meters.items():
            meter.update(float(losses[name].detach().cpu()), batch_size)
    return {name: meter.average for name, meter in meters.items()}


@torch.no_grad()
def evaluate(
    model: VSEARiskNet,
    loader: DataLoader,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> Dict[str, float]:
    """Evaluate document risk classification and evidence localization."""
    model.eval()
    labels: List[int] = []
    probabilities: List[np.ndarray] = []
    node_pred: List[int] = []
    node_true: List[int] = []
    edge_pred: List[int] = []
    edge_true: List[int] = []

    for batch in loader:
        batch = move_to_device(batch, train_config.device)
        outputs = model(batch)
        probs = torch.softmax(outputs["doc_logits"], dim=-1).cpu().numpy()
        probabilities.append(probs)
        labels.extend(batch["labels"].cpu().numpy().astype(int).tolist())

        node_mask = batch["node_mask"].cpu().numpy().astype(bool)
        node_scores = torch.sigmoid(outputs["node_logits"]).cpu().numpy()
        node_targets = batch["node_labels"].cpu().numpy()
        node_pred.extend((node_scores[node_mask] >= train_config.evidence_threshold).astype(int).tolist())
        node_true.extend((node_targets[node_mask] >= 0.5).astype(int).tolist())

        pair_mask = outputs["pair_mask"].cpu().numpy().astype(bool)
        edge_scores = torch.sigmoid(outputs["edge_logits"]).cpu().numpy()
        edge_targets = batch["edge_labels"].cpu().numpy()
        edge_pred.extend((edge_scores[pair_mask] >= train_config.evidence_threshold).astype(int).tolist())
        edge_true.extend((edge_targets[pair_mask] >= 0.5).astype(int).tolist())

    prob_array = np.concatenate(probabilities, axis=0) if probabilities else np.zeros((0, model_config.num_classes))
    return compute_metrics(
        labels,
        prob_array,
        node_pred,
        node_true,
        edge_pred,
        edge_true,
        model_config.num_classes,
    )


def fit_model(
    train_samples: Sequence[dict],
    valid_samples: Sequence[dict],
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> Tuple[VSEARiskNet, Dict[str, float]]:
    """Train a model and return the best validation metrics."""
    set_seed(train_config.seed)
    torch.set_num_threads(max(1, int(train_config.num_threads)))
    model = VSEARiskNet(model_config).to(train_config.device)
    train_loader = make_loader(train_samples, model_config, train_config, shuffle=True)
    valid_loader = make_loader(valid_samples, model_config, train_config, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    best_metrics: Dict[str, float] = {}
    best_score = -1.0
    for epoch in range(1, train_config.epochs + 1):
        train_losses = train_one_epoch(model, train_loader, optimizer, model_config, train_config)
        valid_metrics = evaluate(model, valid_loader, model_config, train_config)
        score = valid_metrics["Macro-F1"] + 0.25 * valid_metrics["Evidence-F1"]
        if score > best_score:
            best_score = score
            best_metrics = {**valid_metrics, "epoch": float(epoch), "train_loss": train_losses["loss"]}
    return model, best_metrics


def train_from_path(
    data_path: Optional[str],
    output_dir: str,
    epochs: int = 8,
    seed: int = 42,
) -> Dict[str, float]:
    """Train on a JSONL file or a generated synthetic dataset."""
    model_config = ModelConfig()
    train_config = TrainConfig(seed=seed, epochs=epochs)
    samples = load_jsonl(data_path) if data_path else generate_synthetic_samples(96, seed, model_config.num_classes)
    split = int(len(samples) * 0.8)
    train_samples, valid_samples = samples[:split], samples[split:]
    model, metrics = fit_model(train_samples, valid_samples, model_config, train_config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metrics": metrics, "model_config": model_config.__dict__, "train_config": train_config.__dict__}, output / "vsea_risknet.pt")
    save_json(metrics, output / "metrics.json")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description="Train VSEA-RiskNet on bill risk data.")
    parser.add_argument("--data", type=str, default=None, help="Path to a JSONL bill-risk dataset.")
    parser.add_argument("--output", type=str, default="outputs/train", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main() -> None:
    """Command-line entry point."""
    args = build_arg_parser().parse_args()
    metrics = train_from_path(args.data, args.output, epochs=args.epochs, seed=args.seed)
    print(metrics)


if __name__ == "__main__":
    main()
