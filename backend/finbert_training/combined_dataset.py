# combined_dataset.py
"""
Merge synthetic and real (pseudo-labeled) datasets into combined train/val/test splits.
"""

import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from app.core.finbert_client import FinBERTClient


def load_synthetic(data_dir: str = "./data") -> pd.DataFrame:
    """
    Load synthetic train and val datasets.
    
    Args:
        data_dir: Directory containing synthetic CSV files
        
    Returns:
        DataFrame with text and label columns
        
    Raises:
        FileNotFoundError: If synthetic files not found
    """
    data_path = Path(data_dir)
    train_path = data_path / "synthetic_train.csv"
    val_path = data_path / "synthetic_val.csv"
    
    # Check if files exist
    if not train_path.exists():
        raise FileNotFoundError(f"Synthetic train file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Synthetic val file not found: {val_path}")
    
    print(f"Loading synthetic datasets from {data_dir}...")
    
    # Load both files
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Concatenate
    synth_df = pd.concat([train_df, val_df], ignore_index=True)
    
    print(f"✅ Loaded {len(synth_df)} synthetic samples")
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"   Class distribution:\n{synth_df['label'].value_counts()}")
    
    return synth_df


def load_real(data_dir: str = "./data", auto_label: bool = True) -> pd.DataFrame:
    """
    Load real news export and optionally apply pseudo-labels using FinBERT.
    
    Args:
        data_dir: Directory containing news_export.csv
        auto_label: If True, run FinBERT to assign pseudo-labels
        
    Returns:
        DataFrame with text and label columns
        
    Raises:
        FileNotFoundError: If news export file not found
    """
    data_path = Path(data_dir)
    news_path = data_path / "news_export.csv"
    
    if not news_path.exists():
        raise FileNotFoundError(f"News export file not found: {news_path}")
    
    print(f"\nLoading real news from {news_path}...")
    
    # Load news export
    real_df = pd.read_csv(news_path)
    
    print(f"✅ Loaded {len(real_df)} real news articles")
    
    # Drop rows with empty text
    initial_len = len(real_df)
    real_df = real_df[real_df['text'].notna() & (real_df['text'].str.strip() != '')].copy()
    
    if len(real_df) < initial_len:
        print(f"⚠️  Removed {initial_len - len(real_df)} articles with empty text")
    
    if auto_label:
        print("\nApplying pseudo-labels using FinBERT...")
        
        # Initialize FinBERT client
        client = FinBERTClient(
            model_name="yiyanghkust/finbert-tone",
            batch_size=16
        )
        
        # Prepare articles in the format expected by FinBERT client
        articles = []
        for idx, row in real_df.iterrows():
            articles.append({
                'id': str(idx),
                'text': row['text'],
                'metadata': {}
            })
        
        # Run FinBERT analysis
        results = client.analyze(articles)
        
        # Map results back to DataFrame
        sentiment_map = {result['id']: result['sentiment'] for result in results}
        real_df['label'] = real_df.index.astype(str).map(sentiment_map)
        
        # Remove any rows where labeling failed
        real_df = real_df[real_df['label'].notna()].copy()
        
        print(f"✅ Pseudo-labeled {len(real_df)} articles")
        print(f"   Label distribution:\n{real_df['label'].value_counts()}")
    
    # Keep only text and label columns
    real_df = real_df[['text', 'label']].copy()
    
    return real_df


def combine_and_split(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    data_dir: str = "./data",
    test_ratio: float = 0.1,
    val_ratio: float = 0.1
):
    """
    Combine synthetic and real datasets, then split into train/val/test.
    
    Args:
        synth_df: Synthetic dataset DataFrame
        real_df: Real (pseudo-labeled) dataset DataFrame
        data_dir: Directory to save combined datasets
        test_ratio: Ratio of data for test set (default: 0.1)
        val_ratio: Ratio of remaining data for validation (default: 0.1)
    """
    print("\n" + "="*70)
    print("Combining and Splitting Datasets")
    print("="*70)
    
    # Concatenate synthetic and real data
    combined_df = pd.concat([synth_df, real_df], ignore_index=True)
    
    print(f"Combined dataset size: {len(combined_df)}")
    print(f"  - Synthetic: {len(synth_df)}")
    print(f"  - Real: {len(real_df)}")
    print(f"\nCombined class distribution:\n{combined_df['label'].value_counts()}")
    
    # Shuffle combined data
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # First split: test set
    train_val_df, test_df = train_test_split(
        combined_df,
        test_size=test_ratio,
        stratify=combined_df['label'],
        random_state=42
    )
    
    # Second split: train and validation from remaining data
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        stratify=train_val_df['label'],
        random_state=42
    )
    
    print(f"\nSplit sizes:")
    print(f"  - Train: {len(train_df)} ({len(train_df)/len(combined_df)*100:.1f}%)")
    print(f"  - Val:   {len(val_df)} ({len(val_df)/len(combined_df)*100:.1f}%)")
    print(f"  - Test:  {len(test_df)} ({len(test_df)/len(combined_df)*100:.1f}%)")
    
    # Save to CSV
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    train_path = data_path / "combined_train.csv"
    val_path = data_path / "combined_val.csv"
    test_path = data_path / "combined_test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✅ Saved combined datasets:")
    print(f"   Train: {train_path}")
    print(f"   Val:   {val_path}")
    print(f"   Test:  {test_path}")
    
    print(f"\nTrain class distribution:\n{train_df['label'].value_counts()}")
    print(f"\nVal class distribution:\n{val_df['label'].value_counts()}")
    print(f"\nTest class distribution:\n{test_df['label'].value_counts()}")


def main():
    """CLI entrypoint for combining datasets."""
    parser = argparse.ArgumentParser(
        description="Combine synthetic and real datasets with pseudo-labeling"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing datasets (default: ./data)"
    )
    parser.add_argument(
        "--auto_label",
        action="store_true",
        default=True,
        help="Apply pseudo-labels to real data using FinBERT (default: True)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio from remaining data (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Combined Dataset Preparation")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Auto-label real data: {args.auto_label}")
    print(f"Test ratio: {args.test_ratio}")
    print(f"Val ratio: {args.val_ratio}")
    print("="*70 + "\n")
    
    try:
        # Load synthetic data
        synth_df = load_synthetic(data_dir=args.data_dir)
        
        # Load real data with pseudo-labeling
        real_df = load_real(
            data_dir=args.data_dir,
            auto_label=args.auto_label
        )
        
        # Combine and split
        combine_and_split(
            synth_df=synth_df,
            real_df=real_df,
            data_dir=args.data_dir,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio
        )
        
        print("\n✅ Combined dataset preparation complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()