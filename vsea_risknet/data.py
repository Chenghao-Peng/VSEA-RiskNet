"""Dataset utilities for VSEA-RiskNet.

The real task expects OCR boxes, element roles, document labels, and optional
node/edge evidence labels. A synthetic generator is included only for smoke
validation and cross-validation when no private bill dataset is available.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ModelConfig


def stable_hash(value: str) -> int:
    """Return a deterministic integer hash for reproducible feature hashing."""
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def hash_text_features(text: str, dim: int) -> torch.Tensor:
    """Encode text with signed feature hashing.

    This lightweight encoder keeps the repository self-contained. For final
    experiments, it can be replaced by a frozen LayoutLM, BERT, or document VLM
    encoder while keeping the downstream graph modules unchanged.
    """
    vector = torch.zeros(dim, dtype=torch.float32)
    normalized = (text or "").lower().replace("\n", " ").strip()
    if not normalized:
        return vector

    tokens = normalized.split()
    char_tokens = [normalized[i : i + 3] for i in range(max(1, len(normalized) - 2))]
    for token in tokens + char_tokens:
        index = stable_hash(token) % dim
        sign = 1.0 if (stable_hash("sign::" + token) % 2 == 0) else -1.0
        vector[index] += sign
    norm = vector.norm(p=2).clamp_min(1.0)
    return vector / norm


def encode_layout(bbox: Sequence[float], image_size: Sequence[float]) -> torch.Tensor:
    """Encode a bounding box as normalized layout features."""
    width, height = float(image_size[0]), float(image_size[1])
    width = max(width, 1.0)
    height = max(height, 1.0)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0
    area = box_w * box_h
    aspect = box_w / max(box_h, 1.0)
    return torch.tensor(
        [
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
            box_w / width,
            box_h / height,
            cx / width,
            cy / height,
            area / (width * height),
            math.log1p(aspect),
        ],
        dtype=torch.float32,
    )


def fallback_visual_features(
    bbox: Sequence[float], image_size: Sequence[float], dim: int
) -> torch.Tensor:
    """Build deterministic proxy visual features from geometry.

    Real projects should pass ROI visual features extracted from bill crops.
    This proxy enables reproducible unit tests without exposing private images.
    """
    layout = encode_layout(bbox, image_size)
    base = torch.cat(
        [
            layout,
            torch.sin(layout[: min(6, layout.numel())] * math.pi),
            torch.cos(layout[: min(6, layout.numel())] * math.pi),
        ]
    )
    if base.numel() >= dim:
        return base[:dim].clone()
    repeats = int(math.ceil(dim / base.numel()))
    return base.repeat(repeats)[:dim].clone()


def normalize_vector(values: Optional[Sequence[float]], dim: int) -> torch.Tensor:
    """Convert a vector-like object into a fixed-size float tensor."""
    if values is None:
        return torch.zeros(dim, dtype=torch.float32)
    tensor = torch.tensor(list(values), dtype=torch.float32)
    if tensor.numel() >= dim:
        return tensor[:dim]
    return torch.cat([tensor, torch.zeros(dim - tensor.numel(), dtype=torch.float32)])


def relation_type_from_roles(role_i: str, role_j: str) -> str:
    """Infer a relation type from business roles."""
    pair = {role_i, role_j}
    business_pairs = [
        {"amount", "total"},
        {"item", "total"},
        {"tax", "total"},
        {"payer", "payee"},
        {"payer", "seal"},
        {"payee", "seal"},
        {"qr", "bill_no"},
        {"account", "payer"},
        {"account", "payee"},
    ]
    if any(pair == target for target in business_pairs):
        return "business"
    if role_i == role_j or "other" not in pair:
        return "semantic"
    return "spatial"


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load bill samples from a JSONL file."""
    items: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class BillRiskDataset(Dataset):
    """PyTorch dataset for bill risk identification."""

    def __init__(self, samples: Sequence[Dict[str, Any]], config: ModelConfig):
        self.samples = list(samples)
        self.config = config
        self.role_to_id = config.role_to_id
        self.relation_to_id = config.relation_to_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        image_size = sample.get("image_size", [1600, 1000])
        nodes = sample.get("ocr_boxes") or sample.get("nodes") or []
        max_nodes = self.config.max_nodes
        num_nodes = min(len(nodes), max_nodes)

        text_features = torch.zeros(max_nodes, self.config.text_dim, dtype=torch.float32)
        visual_features = torch.zeros(max_nodes, self.config.visual_dim, dtype=torch.float32)
        layout_features = torch.zeros(max_nodes, self.config.layout_dim, dtype=torch.float32)
        role_ids = torch.full((max_nodes,), self.role_to_id.get("other", 0), dtype=torch.long)
        node_labels = torch.zeros(max_nodes, dtype=torch.float32)
        node_mask = torch.zeros(max_nodes, dtype=torch.bool)

        roles: List[str] = []
        for node_idx, node in enumerate(nodes[:max_nodes]):
            text = str(node.get("text", ""))
            bbox = node.get("bbox", [0, 0, 1, 1])
            role = str(node.get("role", "other")).lower()
            roles.append(role)
            role_ids[node_idx] = self.role_to_id.get(role, self.role_to_id.get("other", 0))
            text_features[node_idx] = hash_text_features(text, self.config.text_dim)
            layout_features[node_idx] = encode_layout(bbox, image_size)
            if "visual_feat" in node:
                visual_features[node_idx] = normalize_vector(node.get("visual_feat"), self.config.visual_dim)
            else:
                visual_features[node_idx] = fallback_visual_features(
                    bbox, image_size, self.config.visual_dim
                )
            node_labels[node_idx] = float(node.get("node_label", 0))
            node_mask[node_idx] = True

        edge_labels = torch.zeros(max_nodes, max_nodes, dtype=torch.float32)
        relation_types = torch.zeros(max_nodes, max_nodes, dtype=torch.long)
        relation_mask = torch.zeros(max_nodes, max_nodes, dtype=torch.bool)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                relation_mask[i, j] = True
                role_i = roles[i] if i < len(roles) else "other"
                role_j = roles[j] if j < len(roles) else "other"
                relation_name = relation_type_from_roles(role_i, role_j)
                relation_types[i, j] = self.relation_to_id.get(relation_name, 0)

        for edge in sample.get("edges", []):
            src = int(edge.get("src", edge.get("source", -1)))
            dst = int(edge.get("dst", edge.get("target", -1)))
            if 0 <= src < max_nodes and 0 <= dst < max_nodes and src != dst:
                edge_labels[src, dst] = float(edge.get("label", edge.get("edge_label", 1)))
                relation_name = str(edge.get("type", edge.get("relation", "business"))).lower()
                relation_types[src, dst] = self.relation_to_id.get(relation_name, relation_types[src, dst])
                relation_mask[src, dst] = True

        label = int(sample.get("label", sample.get("risk_label", 0)))
        return {
            "text_features": text_features,
            "visual_features": visual_features,
            "layout_features": layout_features,
            "role_ids": role_ids,
            "node_labels": node_labels,
            "edge_labels": edge_labels,
            "relation_types": relation_types,
            "node_mask": node_mask,
            "relation_mask": relation_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def collate_bill_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of encoded samples into a batch dictionary."""
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


def make_bill_node(
    text: str,
    role: str,
    x: float,
    y: float,
    w: float,
    h: float,
    evidence: int = 0,
) -> Dict[str, Any]:
    """Create a synthetic OCR node."""
    return {
        "text": text,
        "role": role,
        "bbox": [x, y, x + w, y + h],
        "node_label": evidence,
    }


def generate_synthetic_samples(
    n_samples: int = 72, seed: int = 42, num_classes: int = 4
) -> List[Dict[str, Any]]:
    """Generate synthetic bill-risk samples for execution validation.

    The synthetic data is intentionally simple and should never be reported as
    the paper dataset. It exists only to verify code paths, tensor shapes, loss
    computation, and cross-validation reproducibility.
    """
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []
    companies = ["Aster Trading", "Northwind Co", "Lotus Supply", "Blue River Ltd"]
    for idx in range(n_samples):
        label = idx % max(1, num_classes)
        rng.shuffle(companies)
        payer = companies[0]
        payee = companies[1]
        total = round(rng.uniform(120.0, 960.0), 2)
        tax = round(total * 0.06, 2)
        item_total = round(total - tax, 2)
        y0 = 60.0
        nodes: List[Dict[str, Any]] = [
            make_bill_node(f"Bill No BN-{idx:05d}", "bill_no", 50, y0, 220, 32),
            make_bill_node(f"Date 2026-06-{1 + idx % 28:02d}", "date", 320, y0, 180, 32),
            make_bill_node(f"Payer {payer}", "payer", 50, y0 + 70, 360, 36),
            make_bill_node(f"Payee {payee}", "payee", 50, y0 + 120, 360, 36),
            make_bill_node(f"Item service fee {item_total:.2f}", "item", 70, y0 + 220, 440, 36),
            make_bill_node(f"Tax {tax:.2f}", "tax", 620, y0 + 220, 180, 36),
            make_bill_node(f"Total {total:.2f}", "total", 620, y0 + 320, 220, 44),
            make_bill_node(f"Seal {payee}", "seal", 920, y0 + 310, 140, 140),
            make_bill_node(f"QR BN-{idx:05d}", "qr", 920, y0 + 90, 150, 150),
            make_bill_node("Remark approved reimbursement", "other", 50, y0 + 420, 520, 32),
        ]
        edges: List[Dict[str, Any]] = []

        if label == 1:
            nodes[4]["text"] = f"Item service fee {item_total + 57.0:.2f} AMOUNT_MISMATCH"
            nodes[4]["node_label"] = 1
            nodes[6]["node_label"] = 1
            edges.append({"src": 4, "dst": 6, "type": "business", "label": 1})
            edges.append({"src": 5, "dst": 6, "type": "business", "label": 1})
        elif label == 2:
            nodes[7]["text"] = f"Seal {companies[2]} ENTITY_CONFLICT"
            nodes[2]["node_label"] = 1
            nodes[3]["node_label"] = 1
            nodes[7]["node_label"] = 1
            edges.append({"src": 2, "dst": 7, "type": "business", "label": 1})
            edges.append({"src": 3, "dst": 7, "type": "business", "label": 1})
        elif label == 3:
            nodes[8]["text"] = f"QR INVALID-{idx:05d} QR_CONFLICT"
            nodes[0]["node_label"] = 1
            nodes[8]["node_label"] = 1
            edges.append({"src": 0, "dst": 8, "type": "business", "label": 1})
            edges.append({"src": 8, "dst": 7, "type": "business", "label": 1})
        else:
            edges.append({"src": 4, "dst": 6, "type": "business", "label": 0})
            edges.append({"src": 3, "dst": 7, "type": "business", "label": 0})
            edges.append({"src": 0, "dst": 8, "type": "business", "label": 0})

        # Add ordinary semantic/spatial edges so the model sees non-risk relations.
        edges.extend(
            [
                {"src": 2, "dst": 3, "type": "semantic", "label": 0},
                {"src": 4, "dst": 5, "type": "spatial", "label": 0},
                {"src": 5, "dst": 6, "type": "business", "label": 0 if label != 1 else 1},
            ]
        )
        samples.append({"image_size": [1200, 900], "ocr_boxes": nodes, "edges": edges, "label": label})
    rng.shuffle(samples)
    return samples


def write_jsonl(samples: Iterable[Dict[str, Any]], path: str | Path) -> None:
    """Write samples to a JSONL file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
