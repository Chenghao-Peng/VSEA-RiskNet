"""Training losses for VSEA-RiskNet."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Compute masked binary cross-entropy."""
    weight = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        pos_weight=weight,
        reduction="none",
    )
    mask_float = mask.float()
    denom = mask_float.sum().clamp_min(1.0)
    return (loss * mask_float).sum() / denom


def compute_vsea_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> Dict[str, torch.Tensor]:
    """Compute document, evidence, and relation-consistency losses."""
    doc_loss = F.cross_entropy(outputs["doc_logits"], batch["labels"])

    node_loss = masked_bce_with_logits(
        outputs["node_logits"],
        batch["node_labels"],
        batch["node_mask"],
        pos_weight=model_config.risk_relation_positive_weight,
    )
    edge_loss = masked_bce_with_logits(
        outputs["edge_logits"],
        batch["edge_labels"],
        outputs["pair_mask"],
        pos_weight=model_config.risk_relation_positive_weight,
    )

    # Relation consistency emphasizes labeled risky relation edges while still
    # penalizing irrelevant high-response relations.
    relation_weight = 1.0 + batch["edge_labels"].float() * (
        model_config.risk_relation_positive_weight - 1.0
    )
    relation_raw = F.binary_cross_entropy_with_logits(
        outputs["evidence_path_logits"],
        batch["edge_labels"].float(),
        reduction="none",
    )
    relation_mask = outputs["pair_mask"].float()
    relation_loss = (relation_raw * relation_weight * relation_mask).sum() / relation_mask.sum().clamp_min(1.0)

    evidence_loss = node_loss + edge_loss
    total_loss = (
        train_config.doc_loss_weight * doc_loss
        + train_config.evidence_loss_weight * evidence_loss
        + train_config.relation_loss_weight * relation_loss
    )
    return {
        "loss": total_loss,
        "doc_loss": doc_loss.detach(),
        "node_loss": node_loss.detach(),
        "edge_loss": edge_loss.detach(),
        "relation_loss": relation_loss.detach(),
    }
