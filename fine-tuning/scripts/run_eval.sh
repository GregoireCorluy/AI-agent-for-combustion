#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python src/evaluate_model.py --config configs/qlora.yaml --split test
python src/evaluate_model.py --config configs/qlora.yaml --split test --base-only
