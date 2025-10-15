# model_finetune.py
"""
Fine-tune FinBERT model on combined dataset using HuggingFace Trainer.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import set_seed, LABEL2ID, ID2LABEL
from config import (
    MODEL_NAME,
    DATA_DIR,
    OUTPUT_DIR,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    EVAL_STEPS,
    SAVE_STEPS,
    LOGGING_STEPS,
    MAX_LENGTH,
    SEED
)


def load_dataset(csv_path: str, tokenizer) -> Dataset:
    """
    Load CSV dataset and tokenize for training.
    
    Args:
        csv_path: Path to CSV file with text and label columns
        tokenizer: HuggingFace tokenizer instance
        
    Returns:
        HuggingFace Dataset object ready for training
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Validate columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    
    # Map labels to IDs
    df['label'] = df['label'].map(LABEL2ID)
    
    # Remove any rows with invalid labels
    df = df[df['label'].notna()].copy()
    
    print(f"Loaded {len(df)} samples from {csv_path}")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df[['text', 'label']])
    
    # Apply tokenization
    dataset = dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    dataset = dataset.rename_column('label', 'labels')
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return dataset


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation during training.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary of metric scores
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='weighted',
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def get_trainer(train_dataset, val_dataset, tokenizer, model) -> Trainer:
    """
    Configure and return HuggingFace Trainer.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: Tokenizer instance
        model: Model instance
        
    Returns:
        Configured Trainer object
    """
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        report_to="none"  # Disable wandb/tensorboard
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    return trainer


def train_and_save():
    """
    Main training function: load data, train model, and save.
    """
    print("="*70)
    print("FinBERT Fine-tuning")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("="*70 + "\n")
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    print("✅ Model loaded successfully\n")
    
    # Load datasets
    data_dir = Path(DATA_DIR)
    train_csv = data_dir / "combined_train.csv"
    val_csv = data_dir / "combined_val.csv"
    
    print("Loading and tokenizing datasets...")
    train_dataset = load_dataset(str(train_csv), tokenizer)
    val_dataset = load_dataset(str(val_csv), tokenizer)
    
    print(f"\n✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Val dataset: {len(val_dataset)} samples\n")
    
    # Get trainer
    print("Initializing trainer...")
    trainer = get_trainer(train_dataset, val_dataset, tokenizer, model)
    
    print("✅ Trainer initialized\n")
    
    # Train the model
    print("="*70)
    print("Starting training...")
    print("="*70 + "\n")
    
    trainer.train()
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70 + "\n")
    
    # Evaluate on validation set
    print("Final evaluation on validation set:")
    eval_results = trainer.evaluate()
    
    print("\nValidation Metrics:")
    print("-"*70)
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    
    # Save the fine-tuned model
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving fine-tuned model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n✅ Model and tokenizer saved to: {OUTPUT_DIR}")
    print("\n" + "="*70)
    print("Fine-tuning complete!")
    print("="*70)


def main():
    """CLI entrypoint for model fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune FinBERT on combined dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help=f"Base model name (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR,
        help=f"Data directory (default: {DATA_DIR})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    
    args = parser.parse_args()
    
    # Override config with CLI arguments if provided
    if args.model_name != MODEL_NAME:
        import config
        config.MODEL_NAME = args.model_name
    if args.data_dir != DATA_DIR:
        import config
        config.DATA_DIR = args.data_dir
    if args.output_dir != OUTPUT_DIR:
        import config
        config.OUTPUT_DIR = args.output_dir
    if args.epochs != EPOCHS:
        import config
        config.EPOCHS = args.epochs
    if args.batch_size != BATCH_SIZE:
        import config
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate != LEARNING_RATE:
        import config
        config.LEARNING_RATE = args.learning_rate
    
    try:
        train_and_save()
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()