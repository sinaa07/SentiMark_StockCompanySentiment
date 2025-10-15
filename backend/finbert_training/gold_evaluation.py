# gold_evaluation.py
"""
Evaluate base FinBERT on manually labeled gold test set.
Establishes baseline performance before fine-tuning.
"""

import pandas as pd
import argparse
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from app.core.finbert_client import FinBERTClient
from utils import LABEL2ID, ID2LABEL
from config import DATA_DIR


def evaluate_gold(test_csv_path: str):
    """
    Evaluate base FinBERT on manually labeled test set.
    
    Args:
        test_csv_path: Path to CSV with columns: text, manual_label
        
    Raises:
        FileNotFoundError: If test CSV not found
    """
    # Check if file exists
    test_path = Path(test_csv_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_csv_path}")
    
    print("="*70)
    print("Gold Test Set Evaluation - Base FinBERT")
    print("="*70)
    print(f"Test file: {test_csv_path}\n")
    
    # Load test CSV
    print("Loading test data...")
    df = pd.read_csv(test_csv_path)
    
    # Validate columns
    if 'text' not in df.columns or 'manual_label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'manual_label' columns")
    
    print(f"✅ Loaded {len(df)} test samples")
    print(f"\nManual label distribution:")
    print(df['manual_label'].value_counts())
    
    # Remove empty texts
    initial_len = len(df)
    df = df[df['text'].notna() & (df['text'].str.strip() != '')].copy()
    
    if len(df) < initial_len:
        print(f"⚠️  Removed {initial_len - len(df)} samples with empty text")
    
    # Initialize FinBERT client
    print("\nInitializing base FinBERT model...")
    client = FinBERTClient(
        model_name="yiyanghkust/finbert-tone",
        batch_size=16
    )
    
    # Prepare articles for FinBERT client
    print("\nRunning FinBERT predictions...")
    articles = []
    for idx, row in df.iterrows():
        articles.append({
            'id': str(idx),
            'text': row['text'],
            'metadata': {}
        })
    
    # Run FinBERT analysis
    results = client.analyze(articles)
    
    # Map results back to DataFrame
    sentiment_map = {result['id']: result['sentiment'] for result in results}
    df['finbert_label'] = df.index.astype(str).map(sentiment_map)
    
    # Remove rows where prediction failed
    df = df[df['finbert_label'].notna()].copy()
    
    print(f"✅ Generated predictions for {len(df)} samples")
    print(f"\nFinBERT label distribution:")
    print(df['finbert_label'].value_counts())
    
    # Save results with predictions
    output_path = test_path.parent / "real_test_with_predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved predictions to: {output_path}")
    
    # Compute evaluation metrics
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
    print("\nClassification Report:")
    print("-"*70)
    print(classification_report(
        df['manual_label'],
        df['finbert_label'],
        target_names=['positive', 'neutral', 'negative'],
        zero_division=0
    ))
    
    print("\nConfusion Matrix:")
    print("-"*70)
    print("              Predicted")
    print("              positive  neutral  negative")
    cm = confusion_matrix(
        df['manual_label'],
        df['finbert_label'],
        labels=['positive', 'neutral', 'negative']
    )
    for i, label in enumerate(['positive', 'neutral', 'negative']):
        print(f"Actual {label:8s}  {cm[i][0]:8d}  {cm[i][1]:7d}  {cm[i][2]:8d}")
    
    print("\n" + "="*70)
    print("Baseline evaluation complete!")
    print("="*70)


def main():
    """CLI entrypoint for gold test evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate base FinBERT on manually labeled test set"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Path to test CSV file (default: DATA_DIR/real_test.csv)"
    )
    
    args = parser.parse_args()
    
    # Use default path if not specified
    if args.test_csv is None:
        test_csv_path = Path(DATA_DIR) / "real_test.csv"
    else:
        test_csv_path = Path(args.test_csv)
    
    try:
        evaluate_gold(str(test_csv_path))
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()