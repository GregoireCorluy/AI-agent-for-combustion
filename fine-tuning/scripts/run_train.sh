#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python src/validate_dataset.py --config configs/qlora.yaml
RUN_NAME="$(date -u +%Y%m%dT%H%M%S)-$$"
echo "Training run: $RUN_NAME"
accelerate launch --multi_gpu --num_processes 3 src/train.py --config configs/qlora.yaml --run-name "$RUN_NAME"
