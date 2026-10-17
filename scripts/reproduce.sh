#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/naiwa-mpl
export TORCH_HOME="$PWD/data/torch_cache"
# Install requirements and clone the original target data first (see README).
# Frozen V2 checkpoint is committed. This script preserves V1/V2 reports.
python training/prepare_other_dataset.py
python scripts/verify_other_data.py
python training/prepare_hard_ood.py
python training/ood/audit.py
python training/ood/extract.py --splits train val
python training/calibrate_ood.py --output results/v3/phase_a
python training/mine_hard_negatives.py
python training/train_ood.py
python training/ood/extract.py --splits train val --checkpoint checkpoints/best_model_ood.pth --output results/v3/phase_b_features
python training/calibrate_ood.py --features results/v3/phase_b_features --output results/v3/phase_b
python training/compare_mobilenet.py
python training/ood/extract.py --splits train val --checkpoint checkpoints/mobilenet_ood.pth --output results/v3/mobilenet_features
python training/calibrate_ood.py --features results/v3/mobilenet_features --output results/v3/mobilenet
# Freeze model + all parameters BEFORE reading test images/model outputs.
python training/ood/select_model.py
python training/ood/extract.py --splits test
python training/ood/extract.py --splits test --checkpoint checkpoints/best_model_ood.pth --output results/v3/phase_b_features
python training/ood/extract.py --splits test --checkpoint checkpoints/mobilenet_ood.pth --output results/v3/mobilenet_features
python training/ood/evaluate.py --output results/v3/phase_a
python training/ood/evaluate.py --features results/v3/phase_b_features --calibration results/v3/phase_b/calibration.json --output results/v3/phase_b
python training/ood/evaluate.py --features results/v3/mobilenet_features --calibration results/v3/mobilenet/calibration.json --output results/v3/mobilenet
python training/ood/report.py
python training/export_ood.py
python training/verify_ood_onnx.py
# Browser QA: python scripts/verify_ood_browser.py --chromium ''
# Node math QA (after browser QA): node scripts/verify_decision.mjs
# Optional real WASM timing: python scripts/benchmark_ood.py
# Final check (after browser QA): python scripts/check_artifacts.py
