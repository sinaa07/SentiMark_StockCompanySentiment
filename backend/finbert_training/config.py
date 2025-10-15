# config.py
"""
Configuration constants, hyperparameters, and paths for FinBERT fine-tuning.
"""

# Model configuration
MODEL_NAME = "ProsusAI/finbert"

# Directory paths
DATA_DIR = "./data"
OUTPUT_DIR = "./models/finetuned"

# Training hyperparameters
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

# Random seed for reproducibility
SEED = 42

# Evaluation settings
EVAL_STEPS = 50
SAVE_STEPS = 100
LOGGING_STEPS = 10

# Maximum sequence length for tokenization
MAX_LENGTH = 512