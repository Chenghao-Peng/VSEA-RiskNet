"""Configuration objects for VSEA-RiskNet."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModelConfig:
    """Core model hyperparameters."""

    num_classes: int = 4
    text_dim: int = 128
    visual_dim: int = 16
    layout_dim: int = 10
    hidden_dim: int = 64
    role_dim: int = 32
    num_relation_types: int = 3
    num_gnn_layers: int = 2
    dropout: float = 0.20
    max_nodes: int = 48
    top_k: int = 6
    edge_threshold: float = 0.50
    relation_risk_alpha: float = 1.25
    risk_relation_positive_weight: float = 3.0
    role_vocab: List[str] = field(
        default_factory=lambda: [
            "amount",
            "total",
            "tax",
            "item",
            "payer",
            "payee",
            "date",
            "seal",
            "qr",
            "bill_no",
            "account",
            "other",
        ]
    )
    relation_vocab: List[str] = field(
        default_factory=lambda: ["spatial", "semantic", "business"]
    )

    @property
    def role_to_id(self) -> Dict[str, int]:
        """Return a role-to-index mapping with an explicit fallback role."""
        return {name: idx for idx, name in enumerate(self.role_vocab)}

    @property
    def relation_to_id(self) -> Dict[str, int]:
        """Return a relation-type-to-index mapping."""
        return {name: idx for idx, name in enumerate(self.relation_vocab)}


@dataclass
class TrainConfig:
    """Training and validation settings."""

    seed: int = 42
    batch_size: int = 8
    epochs: int = 8
    lr: float = 2e-4
    weight_decay: float = 5e-2
    doc_loss_weight: float = 1.0
    evidence_loss_weight: float = 0.6
    relation_loss_weight: float = 0.35
    grad_clip: float = 5.0
    evidence_threshold: float = 0.5
    device: str = "cpu"
    num_threads: int = 1


@dataclass
class CrossValConfig:
    """Cross-validation settings."""

    folds: int = 3
    repeats: int = 1
    synthetic_samples: int = 72
    save_predictions: bool = True
