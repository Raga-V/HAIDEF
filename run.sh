#!/usr/bin/env bash
set -e

python scripts/dataset_loader.py
python scripts/train_baseline.py --data-dir data/processed --model-dir models/baseline
python scripts/evaluate.py --model-dir models/baseline --data-dir data/processed

echo "Run complete."
