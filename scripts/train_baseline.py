"""Placeholder baseline training script.

Usage: python scripts/train_baseline.py --data-dir data/processed --model-dir models/baseline
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data/processed")
parser.add_argument("--model-dir", default="models/baseline")
args = parser.parse_args()

print(f"Train baseline with data from {args.data_dir} and save to {args.model_dir}")
# TODO: implement training pipeline (tokenize, dataset -> DataLoader, model, train loop)

