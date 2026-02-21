# HAIDEF

Project scaffold for HAIDEF: dataset processing, training baseline and LoRA, and evaluation.

Structure:

- `data/processed/train` and `data/processed/test`: processed datasets ready for training/eval.
- `scripts/`: helper scripts to run dataset generation, training, and evaluation.
- `models/`: saved model artifacts (baseline and LoRA).
- `results/`: metrics and confusion matrices.

Quick start

1. Create a Python environment and install requirements:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. Run dataset extraction:

```bash
python scripts/dataset_loader.py
```

3. Train a baseline (placeholder):

```bash
python scripts/train_baseline.py --data-dir data/processed --model-dir models/baseline
```

4. Evaluate:

```bash
python scripts/evaluate.py --model-dir models/baseline --data-dir data/processed
```

Files to edit

- Implement training logic in `scripts/train_baseline.py` and `scripts/train_lora.py`.
- Implement evaluation metrics in `scripts/evaluate.py`.

