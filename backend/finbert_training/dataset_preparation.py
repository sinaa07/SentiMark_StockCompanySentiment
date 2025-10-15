# dataset_preparation.py
"""
Generate synthetic financial sentiment data using templates.
Balanced dataset for Mac training: ~400-500 samples per class.
"""

import random
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import set_seed

# Indian companies for realistic templates
COMPANIES = [
    "Reliance Industries", "TCS", "Infosys", "HDFC Bank", "ICICI Bank",
    "Wipro", "HCL Technologies", "Bharti Airtel", "ITC", "Axis Bank",
    "Kotak Mahindra Bank", "Maruti Suzuki", "Bajaj Finance", "Asian Paints",
    "Hindustan Unilever", "Tech Mahindra", "Adani Ports", "JSW Steel",
    "Titan Company", "UltraTech Cement", "SBI", "Mahindra & Mahindra",
    "Sun Pharma", "Dr. Reddy's", "Cipla", "Tata Motors", "Power Grid"
]

TARGETS = ["Tech Solutions", "Digital Services", "Capital Ltd", "Financial Corp", 
           "Industrial Group", "Energy Solutions", "Logistics Inc"]

# Template definitions
POSITIVE_TEMPLATES = [
    "{COMPANY} reports a {X}% rise in quarterly revenue.",
    "Shares of {COMPANY} surge after strong Q{PERIOD} results.",
    "{COMPANY} posts record profits of INR {AMT} crore.",
    "Analysts upgrade {COMPANY} on improved growth outlook.",
    "{COMPANY} completes successful acquisition of {TARGET}.",
    "{COMPANY} announces {X}% increase in dividend payout.",
    "{COMPANY} beats market expectations with {X}% profit growth.",
    "Investors cheer as {COMPANY} secures major contract worth INR {AMT} crore.",
    "{COMPANY} stock hits 52-week high on strong demand.",
    "{COMPANY} reports robust performance with revenue of INR {AMT} crore in Q{PERIOD}."
]

NEGATIVE_TEMPLATES = [
    "{COMPANY} reports a {X}% decline in net profit.",
    "Shares of {COMPANY} fall after weak Q{PERIOD} results.",
    "{COMPANY} faces regulatory investigation over irregularities.",
    "Analysts downgrade {COMPANY} citing poor performance.",
    "{COMPANY} posts losses of INR {AMT} crore this quarter.",
    "{COMPANY} stock plunges {X}% amid concerns over debt levels.",
    "Regulatory action against {COMPANY} triggers sell-off.",
    "{COMPANY} misses earnings estimates with {X}% drop in profit.",
    "Investors exit {COMPANY} after disappointing Q{PERIOD} guidance.",
    "{COMPANY} faces legal challenges impacting market confidence."
]

NEUTRAL_TEMPLATES = [
    "{COMPANY} announces leadership changes in key divisions.",
    "{COMPANY} schedules board meeting for Q{PERIOD}.",
    "Shares of {COMPANY} trade flat amid mixed market sentiment.",
    "{COMPANY} introduces a new product line in the domestic market.",
    "{COMPANY} completes acquisition of {TARGET}.",
    "{COMPANY} announces organizational restructuring.",
    "Market participants await {COMPANY} quarterly results.",
]


def fill_template(template: str) -> str:
    """
    Fill a template with random values.
    
    Args:
        template: Template string with placeholders
        
    Returns:
        Filled template string
    """
    text = template
    
    # Replace placeholders
    if "{COMPANY}" in text:
        text = text.replace("{COMPANY}", random.choice(COMPANIES))
    
    if "{TARGET}" in text:
        text = text.replace("{TARGET}", random.choice(TARGETS))
    
    if "{X}" in text:
        # Random percentage based on context
        if "rise" in text or "increase" in text or "growth" in text or "surge" in text:
            pct = random.randint(5, 35)
        elif "decline" in text or "drop" in text or "fall" in text or "plunge" in text:
            pct = random.randint(3, 25)
        else:
            pct = random.randint(5, 20)
        text = text.replace("{X}", str(pct))
    
    if "{AMT}" in text:
        # Random amount in crores
        amt = random.randint(100, 5000)
        text = text.replace("{AMT}", str(amt))
    
    if "{PERIOD}" in text:
        # Quarter number
        period = random.randint(1, 4)
        text = text.replace("{PERIOD}", str(period))
    
    return text


def generate_from_templates(n_per_class: int, seed: int = None) -> pd.DataFrame:
    """
    Generate synthetic dataset from templates.
    
    Args:
        n_per_class: Number of samples to generate per class
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: text, label
    """
    if seed is not None:
        set_seed(seed)
    
    data = []
    
    # Generate positive samples
    print(f"Generating {n_per_class} positive samples...")
    for _ in range(n_per_class):
        template = random.choice(POSITIVE_TEMPLATES)
        text = fill_template(template)
        data.append({"text": text, "label": "positive"})
    
    # Generate negative samples
    print(f"Generating {n_per_class} negative samples...")
    for _ in range(n_per_class):
        template = random.choice(NEGATIVE_TEMPLATES)
        text = fill_template(template)
        data.append({"text": text, "label": "negative"})
    
    # Generate neutral samples
    print(f"Generating {n_per_class} neutral samples...")
    for _ in range(n_per_class):
        template = random.choice(NEUTRAL_TEMPLATES)
        text = fill_template(template)
        data.append({"text": text, "label": "neutral"})
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"\nGenerated {len(df)} total samples")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    return df


def split_and_save(df: pd.DataFrame, out_dir: str, train_ratio: float = 0.8):
    """
    Split dataset into train/val and save to CSV files.
    
    Args:
        df: DataFrame with text and label columns
        out_dir: Output directory path
        train_ratio: Ratio of data for training (rest goes to validation)
    """
    from sklearn.model_selection import train_test_split
    
    # Create output directory if it doesn't exist
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Stratified split to maintain class balance
    train_df, val_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df['label'],
        random_state=42
    )
    
    # Save to CSV
    train_path = out_path / "synthetic_train.csv"
    val_path = out_path / "synthetic_val.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\n✅ Saved datasets:")
    print(f"   Train: {train_path} ({len(train_df)} samples)")
    print(f"   Val:   {val_path} ({len(val_df)} samples)")
    print(f"\nTrain class distribution:\n{train_df['label'].value_counts()}")
    print(f"\nVal class distribution:\n{val_df['label'].value_counts()}")


def main():
    """CLI entrypoint for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic financial sentiment dataset"
    )
    parser.add_argument(
        "--synth_per_class",
        type=int,
        default=150,
        help="Number of synthetic samples per class (default: 150)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for generated datasets (default: ./data)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training data ratio (default: 0.8)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Synthetic Dataset Generation")
    print("="*70)
    print(f"Samples per class: {args.synth_per_class}")
    print(f"Total samples: {args.synth_per_class * 3}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print("="*70 + "\n")
    
    # Generate dataset
    df = generate_from_templates(
        n_per_class=args.synth_per_class,
        seed=args.seed
    )
    
    # Split and save
    split_and_save(
        df=df,
        out_dir=args.output_dir,
        train_ratio=args.train_ratio
    )
    
    print("\n✅ Dataset generation complete!")


if __name__ == "__main__":
    main()