"""Evaluation placeholder script.

Usage: python scripts/evaluate.py --model-dir models/baseline --data-dir data/processed
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", default="models/baseline")
parser.add_argument("--data-dir", default="data/processed")
args = parser.parse_args()

print(f"Evaluate model at {args.model_dir} on data in {args.data_dir}")
# TODO: compute metrics and save to results/metrics.json and confusion matrices
