# V4 Phase 1 — fixed independent cartoon holdout

Hard-OOD FAR is the primary endpoint. All detectors/candidate were chosen on validation only.

| Model | Input | ID closed acc | ID acc incl. reject | Easy FAR | **Hard FAR** | AUROC | FPR@95TPR | ID FRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3_baseline | 128 | 95.89% | 89.04% | 12.00% | 16.20% | 94.24% | 59.45% | 6.85% |
| mobilenet_frozen | 224 | 93.15% | 89.04% | 14.00% | 25.86% | 94.71% | 36.31% | 5.48% |
| mobilenet_partial_ft | 224 | 93.15% | 84.93% | 13.33% | 28.04% | 90.54% | 64.54% | 12.33% |
| mobilenet_frozen_128_control | 128 | 95.89% | 90.41% | 20.00% | 35.51% | 92.32% | 45.65% | 6.85% |
| MobileNet FullFT | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

Validation-selected candidate: `mobilenet_frozen`. Conclusion **C**. Production unchanged.

ID closed accuracy excludes rejection; ID acc incl. reject counts rejected target images as errors. AUROC/AUPR positive OOD. FPR@95TPR: OOD acceptance at at least95% ID acceptance on test ROC, diagnostic only, not threshold tuning.

The benchmark is cartoon/3D emoji art, not all commercial characters/plush photographs. Same rendered source styles remain. V3 received historical OE; new MobileNets use ID CE. Resolution control does not remove training-objective/history confounding.
