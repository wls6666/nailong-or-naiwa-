#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python training/prepare_other_dataset.py
python scripts/verify_other_data.py
python training/dataset.py
python training/train.py
python training/evaluate.py
python training/calibrate_rejection.py
python training/evaluate.py --calibrated
python training/plot_history.py
python training/export_onnx.py
python training/verify_onnx.py
python training/evaluate_extra_other.py
# Optional browser verification needs requirements-dev.txt + Chromium.
# python scripts/verify_browser.py --chromium ''
# node scripts/verify_decision.mjs
# python scripts/check_artifacts.py
