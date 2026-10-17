#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python training/dataset.py
python training/train.py
python training/evaluate.py
python training/plot_history.py
python training/export_onnx.py
python training/verify_onnx.py
