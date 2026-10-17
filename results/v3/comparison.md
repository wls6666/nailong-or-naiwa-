| System | Method | ID closed accuracy | ID accept | ID end-to-end accuracy | Easy FAR | Hard FAR | AUROC | FPR@95TPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| custom_posthoc | v2_softmax | 94.52% | 95.89% | 93.15% | 2.00% | 74.50% | 78.94% | 38.13% |
| custom_posthoc | msp | 94.52% | 94.52% | 91.78% | 74.67% | 94.63% | 69.27% | 89.97% |
| custom_posthoc | p_other | 94.52% | 97.26% | 91.78% | 1.33% | 72.48% | 78.99% | 33.11% |
| custom_posthoc | mahalanobis | 94.52% | 94.52% | 90.41% | 16.00% | 55.03% | 92.67% | 45.82% |
| custom_posthoc | energy | 94.52% | 98.63% | 93.15% | 7.33% | 83.22% | 77.42% | 39.13% |
| custom_posthoc | distance_energy | 94.52% | 93.15% | 89.04% | 6.00% | 53.69% | 90.22% | 45.82% |
| custom_posthoc | distance_energy_p_other | 94.52% | 93.15% | 89.04% | 0.67% | 50.34% | 87.04% | 45.82% |
| custom_energy_oe | msp | 95.89% | 97.26% | 93.15% | 66.67% | 71.14% | 86.54% | 59.20% |
| custom_energy_oe | mahalanobis | 95.89% | 94.52% | 90.41% | 29.33% | 46.98% | 93.96% | 52.51% |
| custom_energy_oe | energy | 95.89% | 97.26% | 93.15% | 16.67% | 12.75% | 96.52% | 8.70% |
| custom_energy_oe | distance_energy | 95.89% | 93.15% | 89.04% | 12.00% | 4.70% | 95.49% | 47.49% |
| mobilenet_frozen | msp | 91.78% | 90.41% | 84.93% | 84.67% | 87.92% | 59.68% | 98.33% |
| mobilenet_frozen | mahalanobis | 91.78% | 97.26% | 89.04% | 12.67% | 60.40% | 94.75% | 26.42% |
| mobilenet_frozen | energy | 91.78% | 93.15% | 86.30% | 90.67% | 85.91% | 58.35% | 88.63% |
| mobilenet_frozen | distance_energy | 91.78% | 94.52% | 90.41% | 28.67% | 39.60% | 90.24% | 42.81% |
