# Validation Report: Three Cross-Validation Revision Cycles
This report records a functional cross-validation pass executed on the built-in synthetic smoke-test samples. The run verifies code execution, tensor compatibility, graph reasoning, loss computation, and metric logging. Because the private ERBD/SBD datasets were not provided, these numbers are not manuscript reproduction results.
## Run Setting
- Dataset: built-in synthetic validation samples
- Samples: 36
- Folds: 3
- Epochs per fold: 2
- Revision cycles: 3
- Selected cycle: `cycle_3_relation_refined`
## Cycle Summary
| Cycle | Accuracy | Macro-F1 | Risk-F1 | AUC | Evidence-F1 | Selection Score |
|---|---:|---:|---:|---:|---:|---:|
| cycle_1_light_baseline | 30.56±3.93 | 18.59±6.32 | 85.71±0.00 | 51.85±18.39 | 6.39±5.62 | 20.19 |
| cycle_2_evidence_balanced | 25.00±0.00 | 10.00±0.00 | 57.14±40.41 | 51.85±8.00 | 18.67±2.80 | 14.67 |
| cycle_3_relation_refined | 30.56±3.93 | 18.81±6.23 | 63.81±30.98 | 54.32±15.52 | 11.03±8.39 | 21.57 |

## Applied Revision Logic
1. Cycle 1 tested a light baseline to confirm the graph pipeline and document/evidence losses were stable.
2. Cycle 2 increased evidence and relation supervision to test whether localization improved under stronger evidence constraints.
3. Cycle 3 refined relation-risk propagation by increasing relation modulation and lowering the edge threshold, then selected the best preset by Macro-F1 plus 0.25 times Evidence-F1.

## Final Code Decision
The final repository keeps the implementation configurable and sets the default training configuration to the selected relation-refined preset. Users can still reproduce the three-cycle protocol with `python -m vsea_risknet.cross_validate --three-cycles` and override settings for full-scale training.
