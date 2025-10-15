
# finbert_training

Modular package to fine-tune and evaluate FinBERT for financial sentiment (Bullish/Neutral/Bearish), with synthetic data generation utilities.

## Structure
```
finbert_training/
├── README.md
├── __init__.py
├── config.py
├── utils.py
├── dataset_preparation.py
├── model_finetune.py
├── evaluation.py
└── mlm_adaptation.py   # optional
```
Outputs (created when you run the scripts):
```
data/
  synthetic_train.csv
  synthetic_val.csv
  real_test.csv           # optional
models/
  finetuned_finbert/      # saved model after training
reports/
  metrics.json
```

## Dependencies
- Python 3.9+
- transformers>=4.41
- datasets>=2.20
- accelerate>=0.32
- scikit-learn
- pandas
- numpy
- torch

Install:
```bash
pip install -U transformers datasets accelerate scikit-learn pandas numpy torch
```

## Data Format
CSV with headers: `text,label` where `label ∈ {positive, neutral, negative}`.

Example:
```
text,label
"Reliance shares surge 4% after strong Q2 results",positive
"Infosys announces Q2 earnings",neutral
"Tata Motors profit declines 10% YoY, misses estimates",negative
```

## Quickstart

### 1) Generate synthetic data
```bash
python dataset_preparation.py --out_dir data --synth_per_class 800 --seed 42
```

### 2) Fine-tune FinBERT
```bash
python model_finetune.py   --train_csv data/synthetic_train.csv   --val_csv data/synthetic_val.csv   --output_dir models/finetuned_finbert   --epochs 3 --batch_size 16 --lr 2e-5 --fp16 false
```

### 3) Evaluate
```bash
python evaluation.py   --test_csv data/real_test.csv   --model_dir models/finetuned_finbert   --baseline_model yiyanghkust/finbert-tone   --report_path reports/metrics.json
```

> If you don't have a real test set yet, temporarily use the val set (not ideal).
