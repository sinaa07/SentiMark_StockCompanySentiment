# utils.py
"""
Shared utility functions and mappings for FinBERT training.
"""

import random
import numpy as np
import torch
import re


def set_seed(seed: int):
    """
    Set random seeds for reproducibility across random, numpy, and torch.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_text(text: str) -> str:
    """
    Clean and normalize text for training.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


# Label mappings for FinBERT sentiment classification
LABEL2ID = {
    'positive': 0,
    'neutral': 1,
    'negative': 2
}

ID2LABEL = {
    0: 'positive',
    1: 'neutral',
    2: 'negative'
}