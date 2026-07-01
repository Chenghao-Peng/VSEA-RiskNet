# 🧾 VSEA-RiskNet: Visual Semantic and Element Association-Driven Bill Risk Identification

<p align="center">
  <b>Risk-oriented document reasoning · Visual-semantic element graph · Evidence-level auditing</b>
</p>

This repository provides a clean PyTorch implementation of **VSEA-RiskNet**, a bill risk identification framework driven by **visual semantics** and **business-element association**. The code follows a compact research-prototype style and is designed for easy extension to private reimbursement, settlement, auditing, and financial risk-control datasets.

---

## ✨ Highlights

- **Element-centered representation.** OCR boxes are reorganized into bill element nodes with text, visual, layout, and role features.
- **Risk-oriented graph construction.** Spatial, semantic, and business-consistency edges are modeled explicitly instead of treating OCR boxes as a flat sequence.
- **Visual-semantic consistency reasoning.** Relation-risk scores modulate graph message passing so conflict evidence is propagated along auditable paths.
- **Evidence-aware training.** Document classification, node evidence, and relation evidence are jointly optimized.
- **Reproducible validation.** The repository includes a three-cycle cross-validation runner and synthetic smoke-test data generation.

---

## 🧠 Method Overview

VSEA-RiskNet converts a bill image and its OCR boxes into a structured element graph:

```text
Bill Image + OCR Boxes
        ↓
Visual Semantic Encoding
        ↓
Element Nodes
        ↓
Risk-oriented Element Graph
        ↓
Visual-Semantic Consistency Reasoning
        ↓
Risk Evidence Aggregation
        ↓
Risk Category + Risk Confidence + Evidence Nodes/Edges
```

The implementation uses lightweight hashed text features and geometry-derived visual proxies by default so that the code can run without private images or pretrained weights. In formal experiments, these modules can be replaced with LayoutLM, Donut, PaddleOCR features, ROI visual encoders, or document VLM embeddings while keeping the downstream graph reasoning modules unchanged.

---

## 📁 Repository Structure

```text
vsea_risknet_final/
├── README.md
├── requirements.txt
├── validation_report.md
├── validation_runs/
│   ├── crossval_results.json
│   └── synthetic_validation_samples.jsonl
└── vsea_risknet/
    ├── config.py          # Model, training, and cross-validation configs
    ├── data.py            # JSONL loader, feature encoding, synthetic smoke data
    ├── model.py           # VSEA-RiskNet architecture
    ├── losses.py          # Classification, evidence, and relation losses
    ├── utils.py           # Metrics, reproducibility, serialization helpers
    ├── train.py           # Training and evaluation routines
    └── cross_validate.py  # K-fold CV and three revision-cycle runner
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run synthetic cross-validation smoke test

```bash
PYTHONPATH=. python -m vsea_risknet.cross_validate \
  --output outputs/crossval \
  --folds 3 \
  --epochs 2 \
  --samples 36 \
  --three-cycles
```

### 3. Train on your own bill-risk dataset

```bash
PYTHONPATH=. python -m vsea_risknet.train \
  --data data/bill_risk_train.jsonl \
  --output outputs/train \
  --epochs 80 \
  --seed 42
```

---

## 🗂️ Expected JSONL Data Format

Each line is one bill sample. The fields `visual_feat` and evidence labels are optional, but recommended for reproducing evidence-level results.

```json
{
  "id": "sample_0001",
  "image_size": [1600, 1000],
  "label": 1,
  "ocr_boxes": [
    {
      "text": "Total 371.70",
      "bbox": [1040, 720, 1210, 770],
      "role": "total",
      "visual_feat": [0.12, 0.44, 0.03],
      "node_label": 1
    },
    {
      "text": "Item service fee 428.70",
      "bbox": [120, 460, 680, 510],
      "role": "item",
      "node_label": 1
    }
  ],
  "edges": [
    {"src": 1, "dst": 0, "type": "business", "label": 1}
  ]
}
```

### Label convention

- `label = 0`: normal bill
- `label > 0`: risk category, such as amount anomaly, entity inconsistency, seal/QR anomaly, or a custom class
- `node_label = 1`: evidence element node
- `edge.label = 1`: evidence relation edge

---

## 📊 Metrics

The validation code reports five metrics aligned with the manuscript setting:

| Metric | Meaning |
|---|---|
| Accuracy | Overall document-level classification correctness |
| Macro-F1 | Class-balanced risk category performance |
| Risk-F1 | Binary normal-vs-risk detection ability |
| AUC | Risk-confidence ranking quality |
| Evidence-F1 | Node/edge evidence localization agreement |

---

## 🔬 Three-Cycle Revision Protocol

The command `--three-cycles` evaluates three conservative presets:

1. **Light baseline**: lower hidden dimension and weaker evidence supervision.
2. **Evidence-balanced**: stronger evidence loss and relation supervision.
3. **Relation-refined**: stronger relation-risk modulation and slightly lower edge threshold.

The runner selects the best cycle by:

```text
Selection Score = Macro-F1 + 0.25 × Evidence-F1
```

This protocol is intended for fast development feedback. For paper-level reporting, use the real ERBD/SBD datasets, fixed splits, repeated seeds, and full training epochs.

---

## ⚠️ Notes on Reproducibility

The included synthetic data is **only a smoke-test dataset**. It verifies tensor shapes, graph construction, loss computation, metric computation, and cross-validation execution. It must not be reported as experimental evidence for the manuscript. To reproduce formal results, prepare the real bill images, OCR outputs, element annotations, document labels, and evidence-node/edge labels.

---

## 📌 Citation Placeholder

```bibtex
@article{vsea_risknet,
  title   = {Visual Semantic and Element Association-Driven Bill Risk Identification},
  author  = {Anonymous},
  journal = {Manuscript under review},
  year    = {2026}
}
```
