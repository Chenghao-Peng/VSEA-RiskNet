"""VSEA-RiskNet model implementation.

The implementation follows the paper-level design: visual-semantic element
encoding, risk-oriented element graph construction, consistency relation
reasoning, and risk evidence aggregation.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig


class MLP(nn.Module):
    """A compact feed-forward block with normalization and dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisualSemanticEncoder(nn.Module):
    """Fuse OCR text, ROI visual proxy, layout, and element-role features."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.text_proj = MLP(config.text_dim, config.hidden_dim, config.hidden_dim, config.dropout)
        self.visual_proj = MLP(config.visual_dim, config.hidden_dim, config.hidden_dim, config.dropout)
        self.layout_proj = MLP(config.layout_dim, config.hidden_dim, config.hidden_dim, config.dropout)
        self.role_embedding = nn.Embedding(len(config.role_vocab), config.role_dim)
        self.fusion = MLP(
            config.hidden_dim * 3 + config.role_dim,
            config.hidden_dim * 2,
            config.hidden_dim,
            config.dropout,
        )

    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
        layout_features: torch.Tensor,
        role_ids: torch.Tensor,
    ) -> torch.Tensor:
        text_hidden = self.text_proj(text_features)
        visual_hidden = self.visual_proj(visual_features)
        layout_hidden = self.layout_proj(layout_features)
        role_hidden = self.role_embedding(role_ids)
        fused = torch.cat([text_hidden, visual_hidden, layout_hidden, role_hidden], dim=-1)
        return self.fusion(fused)


class RiskGraphBuilder(nn.Module):
    """Build relation logits and sparse risk-oriented adjacency."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        pair_dim = config.hidden_dim * 4 + 4
        self.pair_encoder = MLP(pair_dim, config.hidden_dim * 2, config.hidden_dim, config.dropout)
        self.relation_type_head = nn.Linear(config.hidden_dim, config.num_relation_types)
        self.edge_risk_head = nn.Linear(config.hidden_dim, 1)

    @staticmethod
    def _pair_layout_features(layout_features: torch.Tensor) -> torch.Tensor:
        """Compute pairwise layout features from normalized boxes."""
        centers = layout_features[..., 6:8]
        sizes = layout_features[..., 4:6].clamp_min(1e-6)
        center_i = centers.unsqueeze(2)
        center_j = centers.unsqueeze(1)
        size_i = sizes.unsqueeze(2)
        size_j = sizes.unsqueeze(1)
        distance = torch.norm(center_i - center_j, dim=-1, keepdim=True)
        size_ratio = torch.log((size_i[..., :1] * size_i[..., 1:2]) / (size_j[..., :1] * size_j[..., 1:2]) + 1e-6)
        horizontal_gap = torch.abs(center_i[..., :1] - center_j[..., :1])
        vertical_gap = torch.abs(center_i[..., 1:2] - center_j[..., 1:2])
        return torch.cat([distance, size_ratio, horizontal_gap, vertical_gap], dim=-1)

    @staticmethod
    def _topk_mask(scores: torch.Tensor, mask: torch.Tensor, top_k: int) -> torch.Tensor:
        """Return a boolean mask that keeps the top-k neighbors for each node."""
        if top_k <= 0:
            return torch.zeros_like(mask, dtype=torch.bool)
        masked_scores = scores.masked_fill(~mask, -1e9)
        k = min(top_k, scores.size(-1))
        _, indices = torch.topk(masked_scores, k=k, dim=-1)
        selected = torch.zeros_like(mask, dtype=torch.bool)
        selected.scatter_(-1, indices, True)
        return selected & mask

    def forward(
        self,
        node_hidden: torch.Tensor,
        layout_features: torch.Tensor,
        node_mask: torch.Tensor,
        relation_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bsz, num_nodes, hidden_dim = node_hidden.shape
        h_i = node_hidden.unsqueeze(2).expand(bsz, num_nodes, num_nodes, hidden_dim)
        h_j = node_hidden.unsqueeze(1).expand(bsz, num_nodes, num_nodes, hidden_dim)
        pair_layout = self._pair_layout_features(layout_features)
        pair_input = torch.cat([h_i, h_j, torch.abs(h_i - h_j), h_i * h_j, pair_layout], dim=-1)
        pair_hidden = self.pair_encoder(pair_input)
        relation_type_logits = self.relation_type_head(pair_hidden)
        relation_probs = F.softmax(relation_type_logits, dim=-1)
        edge_logits = self.edge_risk_head(pair_hidden).squeeze(-1)

        pair_mask = relation_mask & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        eye = torch.eye(num_nodes, dtype=torch.bool, device=node_hidden.device).unsqueeze(0)
        pair_mask = pair_mask & ~eye

        relation_confidence = relation_probs.max(dim=-1).values
        edge_probability = torch.sigmoid(edge_logits)
        importance = edge_probability * relation_confidence
        threshold_mask = importance >= self.config.edge_threshold
        topk_mask = self._topk_mask(importance, pair_mask, self.config.top_k)
        active_mask = pair_mask & (threshold_mask | topk_mask)

        return {
            "pair_hidden": pair_hidden,
            "relation_type_logits": relation_type_logits,
            "relation_probs": relation_probs,
            "edge_logits": edge_logits,
            "pair_mask": pair_mask,
            "active_mask": active_mask,
            "importance": importance,
        }


class RelationReasoningLayer(nn.Module):
    """One risk-aware relation reasoning layer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.relation_transforms = nn.ModuleList(
            [nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(config.num_relation_types)]
        )
        self.update = MLP(config.hidden_dim * 2, config.hidden_dim * 2, config.hidden_dim, config.dropout)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        node_hidden: torch.Tensor,
        relation_probs: torch.Tensor,
        edge_logits: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        source_by_type = torch.stack([layer(node_hidden) for layer in self.relation_transforms], dim=2)
        messages = torch.einsum("bijr,bjrh->bijh", relation_probs, source_by_type)

        q = self.query(node_hidden)
        k = self.key(node_hidden)
        attention_logits = torch.einsum("bih,bjh->bij", q, k) / math.sqrt(node_hidden.size(-1))
        attention_logits = attention_logits + self.config.relation_risk_alpha * torch.sigmoid(edge_logits)
        attention_logits = attention_logits.masked_fill(~active_mask, -1e9)
        attention = F.softmax(attention_logits, dim=-1)
        attention = torch.where(active_mask.any(dim=-1, keepdim=True), attention, torch.zeros_like(attention))
        aggregated = torch.einsum("bij,bijh->bih", attention, messages)
        updated = self.update(torch.cat([node_hidden, aggregated], dim=-1))
        return self.norm(node_hidden + self.dropout(updated))


class VisualSemanticConsistencyReasoner(nn.Module):
    """Stack multiple relation reasoning layers."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([RelationReasoningLayer(config) for _ in range(config.num_gnn_layers)])

    def forward(
        self,
        node_hidden: torch.Tensor,
        relation_probs: torch.Tensor,
        edge_logits: torch.Tensor,
        active_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = node_hidden
        for layer in self.layers:
            hidden = layer(hidden, relation_probs, edge_logits, active_mask)
            hidden = hidden * node_mask.unsqueeze(-1).float()
        return hidden


class RiskEvidenceAggregator(nn.Module):
    """Aggregate node and edge evidence into document-level risk predictions."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.node_head = nn.Linear(config.hidden_dim, 1)
        self.edge_project = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.doc_head = MLP(config.hidden_dim * 2, config.hidden_dim * 2, config.num_classes, config.dropout)
        self.evidence_path_head = nn.Linear(config.hidden_dim, 1)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        masked_logits = logits.masked_fill(~mask, -1e9)
        probs = F.softmax(masked_logits, dim=dim)
        probs = torch.where(mask.any(dim=dim, keepdim=True), probs, torch.zeros_like(probs))
        return probs

    def forward(
        self,
        node_hidden: torch.Tensor,
        pair_hidden: torch.Tensor,
        edge_logits: torch.Tensor,
        node_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        node_logits = self.node_head(node_hidden).squeeze(-1)
        node_attention = self._masked_softmax(node_logits, node_mask, dim=-1)
        node_doc = torch.einsum("bn,bnh->bh", node_attention, node_hidden)

        edge_scores = edge_logits.masked_fill(~pair_mask, -1e9)
        flat_edge_scores = edge_scores.flatten(start_dim=1)
        flat_pair_mask = pair_mask.flatten(start_dim=1)
        flat_edge_attention = self._masked_softmax(flat_edge_scores, flat_pair_mask, dim=-1)
        edge_features = self.edge_project(pair_hidden).flatten(start_dim=1, end_dim=2)
        edge_doc = torch.einsum("be,beh->bh", flat_edge_attention, edge_features)

        doc_repr = torch.cat([node_doc, edge_doc], dim=-1)
        doc_logits = self.doc_head(doc_repr)
        evidence_path_logits = self.evidence_path_head(pair_hidden).squeeze(-1)
        return {
            "doc_logits": doc_logits,
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "evidence_path_logits": evidence_path_logits,
            "node_attention": node_attention,
            "edge_attention": flat_edge_attention.view_as(edge_logits),
        }


class VSEARiskNet(nn.Module):
    """Visual Semantic and Element Association-driven Risk Network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = VisualSemanticEncoder(config)
        self.graph_builder = RiskGraphBuilder(config)
        self.reasoner = VisualSemanticConsistencyReasoner(config)
        self.aggregator = RiskEvidenceAggregator(config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        node_hidden = self.encoder(
            batch["text_features"],
            batch["visual_features"],
            batch["layout_features"],
            batch["role_ids"],
        )
        node_hidden = node_hidden * batch["node_mask"].unsqueeze(-1).float()
        graph_outputs = self.graph_builder(
            node_hidden,
            batch["layout_features"],
            batch["node_mask"],
            batch["relation_mask"],
        )
        reasoned_nodes = self.reasoner(
            node_hidden,
            graph_outputs["relation_probs"],
            graph_outputs["edge_logits"],
            graph_outputs["active_mask"],
            batch["node_mask"],
        )
        aggregated = self.aggregator(
            reasoned_nodes,
            graph_outputs["pair_hidden"],
            graph_outputs["edge_logits"],
            batch["node_mask"],
            graph_outputs["pair_mask"],
        )
        return {**graph_outputs, **aggregated, "node_hidden": reasoned_nodes}
