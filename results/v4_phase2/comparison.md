# V4 Phase 2 — controlled cartoon OE experiment

All thresholds/checkpoints were locked before the new final-test inference. Lower FAR is better.

| Model | Real cartoon FAR | Emoji FAR | Easy FAR | ID closed | ID incl. reject | ID rejected /73 | closed / accepted errors | AUROC | FPR@95TPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3_baseline | 62.16% | 16.20% | 12.00% | 95.89% | 89.04% | 5 | 3 / 3 | 92.87% | 64.46% |
| v3_recalibrated | 8.78% | 33.96% | 12.00% | 95.89% | 93.15% | 2 | 3 / 3 | 97.22% | 7.43% |
| random_seed42 | 9.46% | 33.64% | 4.67% | 95.89% | 90.41% | 4 | 3 / 3 | 96.25% | 21.97% |
| random_seed43 | 12.16% | 23.05% | 6.67% | 95.89% | 89.04% | 6 | 3 / 2 | 94.67% | 53.96% |
| random_seed44 | 5.41% | 26.17% | 2.67% | 97.26% | 93.15% | 3 | 2 / 2 | 96.97% | 12.44% |
| mined_seed42 | 12.16% | 20.25% | 5.33% | 97.26% | 89.04% | 6 | 2 / 2 | 94.46% | 50.73% |
| mined_seed43 | 7.43% | 24.61% | 5.33% | 95.89% | 93.15% | 3 | 3 / 2 | 97.55% | 6.62% |
| mined_seed44 | 3.38% | 27.73% | 3.33% | 97.26% | 93.15% | 3 | 2 / 2 | 96.92% | 13.89% |

Across-seed mean ± sample SD (three actual training seeds; not test sampling uncertainty):

- random: Real FAR 9.01% ± 3.40%; Emoji FAR 27.62% ± 5.44%; ID incl. reject 90.87% ± 2.09%.
- mined: Real FAR 7.66% ± 4.40%; Emoji FAR 24.20% ± 3.76%; ID incl. reject 91.78% ± 2.37%.

Group-bootstrap intervals, paired per fixed test image and averaged over seeds (negative is improvement):

- hard mined_minus_random: -3.43%, 95% CI [-5.42%, -1.53%], groups=121.
- hard mined_minus_v3: 8.00%, 95% CI [4.64%, 11.46%], groups=121.
- cartoon mined_minus_random: -1.35%, 95% CI [-5.88%, 0.00%], groups=2.
- cartoon mined_minus_v3: -54.50%, 95% CI [-57.31%, -45.10%], groups=2.

Real-cartoon test has only two held-out film/cast groups. This is a genuine source/identity holdout, but does **not** establish robust performance across cartoon brands or art styles. Per-image sample size overstates independent evidence. No seed or model was selected using test. A_recalibrated isolates validation expansion from training effects.

Production replacement recommended: **False**. See decision.json and the full experiment README. No production assets changed.
