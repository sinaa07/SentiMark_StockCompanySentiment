# evaluation.py
"""
Evaluate fine-tuned model on gold test set and compare with baseline.
"""
import json
import pandas as pd
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import LABEL2ID, ID2LABEL
from config import DATA_DIR, OUTPUT_DIR, MAX_LENGTH


def evaluate_finetuned_model(test_csv_path: str, model_dir: str):
    """
    Evaluate fine-tuned model on manually labeled test set.
    
    Args:
        test_csv_path: Path to CSV with text and manual_label columns
        model_dir: Path to fine-tuned model directory
    """
    # Check if files exist
    test_path = Path(test_csv_path)
    model_path = Path(model_dir)
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_csv_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print("="*70)
    print("Fine-tuned Model Evaluation")
    print("="*70)
    print(f"Test file: {test_csv_path}")
    print(f"Model: {model_dir}\n")
    
    # Load test data
    print("Loading test data...")
    df = pd.read_csv(test_csv_path)
    
    if 'text' not in df.columns or 'manual_label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'manual_label' columns")
    
    print(f"✅ Loaded {len(df)} test samples")
    
    # Load fine-tuned model and tokenizer
    print("\nLoading fine-tuned model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}\n")
    
    # Run predictions
    print("Running predictions...")
    predictions = []
    
    with torch.no_grad():
        for text in df['text']:
            # Tokenize
            inputs = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            outputs = model(**inputs)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=1).item()
            pred_label = ID2LABEL[pred_id]
            
            predictions.append(pred_label)
    
    df['finetuned_label'] = predictions
    
    print(f"✅ Generated predictions for {len(df)} samples")
    print(f"\nFine-tuned label distribution:")
    print(df['finetuned_label'].value_counts())
    
    # Save results
    output_path = test_path.parent / "real_test_finetuned_predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved predictions to: {output_path}")
    
    # Compute metrics
    print("\n" + "="*70)
    print("EVALUATION METRICS - Fine-tuned Model")
    print("="*70)
    
    print("\nClassification Report:")
    print("-"*70)
    print(classification_report(
        df['manual_label'],
        df['finetuned_label'],
        target_names=['positive', 'neutral', 'negative'],
        zero_division=0
    ))
    
    print("\nConfusion Matrix:")
    print("-"*70)
    print("              Predicted")
    print("              positive  neutral  negative")
    cm = confusion_matrix(
        df['manual_label'],
        df['finetuned_label'],
        labels=['positive', 'neutral', 'negative']
    )
    for i, label in enumerate(['positive', 'neutral', 'negative']):
        print(f"Actual {label:8s}  {cm[i][0]:8d}  {cm[i][1]:7d}  {cm[i][2]:8d}")
    
    print("\n" + "="*70)


def compare_with_baseline(test_csv_path: str):
    """
    Compare fine-tuned model with baseline FinBERT.
    
    Args:
        test_csv_path: Path to CSV with manual_label, finbert_label, finetuned_label
    """
    test_path = Path(test_csv_path)
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_csv_path}")
    
    df = pd.read_csv(test_csv_path)
    
    # Check if both baseline and fine-tuned predictions exist
    if 'finbert_label' not in df.columns or 'finetuned_label' not in df.columns:
        print("⚠️  Missing baseline or fine-tuned predictions for comparison")
        return
    
    print("\n" + "="*70)
    print("BASELINE vs FINE-TUNED COMPARISON")
    print("="*70)
    
    from sklearn.metrics import accuracy_score, f1_score
    
    # Baseline metrics
    baseline_acc = accuracy_score(df['manual_label'], df['finbert_label'])
    baseline_f1 = f1_score(df['manual_label'], df['finbert_label'], average='weighted', zero_division=0)
    
    # Fine-tuned metrics
    finetuned_acc = accuracy_score(df['manual_label'], df['finetuned_label'])
    finetuned_f1 = f1_score(df['manual_label'], df['finetuned_label'], average='weighted', zero_division=0)
    
    print("\n📊 Performance Summary:")
    print("-"*70)
    print(f"{'Metric':<20} {'Baseline':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-"*70)
    print(f"{'Accuracy':<20} {baseline_acc:<15.4f} {finetuned_acc:<15.4f} {finetuned_acc-baseline_acc:+.4f}")
    print(f"{'F1 Score (weighted)':<20} {baseline_f1:<15.4f} {finetuned_f1:<15.4f} {finetuned_f1-baseline_f1:+.4f}")
    print("-"*70)
    
    if finetuned_acc > baseline_acc:
        print(f"\n✅ Fine-tuning improved accuracy by {(finetuned_acc-baseline_acc)*100:.2f}%")
    else:
        print(f"\n⚠️  Fine-tuning decreased accuracy by {(baseline_acc-finetuned_acc)*100:.2f}%")
    
        # ---- SAVE RESULTS AS JSON ----
    

    # Get structured classification report
    report_dict = classification_report(
        df['manual_label'],
        df['finetuned_label'],
        target_names=['positive', 'neutral', 'negative'],
        zero_division=0,
        output_dict=True
    )

    # Convert confusion matrix to list for JSON serialization
    cm_list = cm.tolist()

    json_output = {
        "model": "finetuned",
        "test_samples": len(df),
        "classification_report": report_dict,
        "confusion_matrix": cm_list
    }

    json_path = test_path.parent / "evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=4)

    print(f"\n✅ Saved JSON report to: {json_path}")


    print("\n" + "="*70)
    
    comparison_output = {
        "baseline": {
            "accuracy": baseline_acc,
            "f1_score": baseline_f1
        },
        "finetuned": {
            "accuracy": finetuned_acc,
            "f1_score": finetuned_f1
        }
    }

    json_path = test_path.parent / "comparison_summary.json"
    with open(json_path, "w") as f:
        json.dump(comparison_output, f, indent=4)

    print(f"✅ Saved comparison JSON to: {json_path}")



def main():
    """CLI entrypoint for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model on gold test set"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Path to test CSV file (default: DATA_DIR/real_test.csv)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Fine-tuned model directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with baseline (requires real_test_with_predictions.csv)"
    )
    
    args = parser.parse_args()
    
    # Use default path if not specified
    if args.test_csv is None:
        test_csv_path = Path(DATA_DIR) / "real_test.csv"
    else:
        test_csv_path = Path(args.test_csv)
    
    try:
        # Evaluate fine-tuned model
        evaluate_finetuned_model(str(test_csv_path), args.model_dir)
        
        # Compare with baseline if requested
        if args.compare:
            comparison_csv = Path(DATA_DIR) / "real_test_finetuned_predictions.csv"
            compare_with_baseline(str(comparison_csv))
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()