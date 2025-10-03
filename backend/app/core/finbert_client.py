# finbert_client.py - Local FinBERT Sentiment Analysis Client
import torch
import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinBERTClient:
    """
    Client for local FinBERT sentiment analysis using PyTorch + Transformers.
    Performs batched inference on preprocessed articles and returns sentiment predictions.
    """
    
    def __init__(self, 
                 model_name: str = "yiyanghkust/finbert-tone",
                 device: str = None,
                 batch_size: int = 16):
        """
        Initialize local FinBERT client with model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier or local path
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
            batch_size: Number of texts to process in each batch
        """
        logger.info(f"Initializing FinBERT client with model: {model_name}")
        
        # Set device (auto-detect GPU if available)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            self.batch_size = batch_size
            self.model_name = model_name
            
            # FinBERT label mapping: [positive, neutral, negative]
            self.label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
            
            logger.info(f"FinBERT client initialized successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing FinBERT client: {str(e)}")
            raise
    
    def analyze(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point: Perform sentiment analysis on preprocessed articles.
        
        Args:
            articles: List of preprocessed articles from finbert_preprocessor
                Expected format: [{"id": "...", "text": "...", "metadata": {...}}, ...]
        
        Returns:
            List of sentiment predictions:
            [
                {
                    "id": "unique_id",
                    "sentiment": "positive" | "neutral" | "negative",
                    "scores": {
                        "positive": 0.82,
                        "neutral": 0.14,
                        "negative": 0.04
                    }
                },
                ...
            ]
        """
        if not articles:
            logger.warning("No articles provided for sentiment analysis")
            return []
        
        logger.info(f"Starting sentiment analysis for {len(articles)} articles/chunks")
        
        try:
            results = []
            
            # Process articles in batches
            for i in range(0, len(articles), self.batch_size):
                batch = articles[i:i + self.batch_size]
                batch_results = self._process_batch(batch)
                results.extend(batch_results)
                
                logger.info(f"Processed batch {i//self.batch_size + 1}/{(len(articles)-1)//self.batch_size + 1}")
            
            logger.info(f"Sentiment analysis complete: {len(results)} predictions generated")
            return results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return []
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of articles through FinBERT.
        
        Args:
            batch: List of article dicts with id, text, metadata
        
        Returns:
            List of sentiment predictions for the batch
        """
        try:
            # Extract IDs and texts
            ids = [article['id'] for article in batch]
            texts = [article['text'] for article in batch]
            
            # Skip empty texts
            valid_indices = [i for i, text in enumerate(texts) if text.strip()]
            if not valid_indices:
                logger.warning("Batch contains only empty texts, skipping")
                return []
            
            valid_ids = [ids[i] for i in valid_indices]
            valid_texts = [texts[i] for i in valid_indices]
            
            # Tokenize batch
            tokenized = self._tokenize_batch(valid_texts)
            
            # Run inference
            logits = self._predict_batch(tokenized)
            
            # Post-process results
            predictions = self._postprocess_logits(logits, valid_ids)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return []
    
    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts for FinBERT input.
        
        Args:
            texts: List of text strings
        
        Returns:
            Dictionary of tokenized tensors ready for model input
        """
        try:
            # Tokenize with padding and truncation
            tokenized = self.tokenizer(
                texts,
                padding="longest",  # Pad to longest sequence in batch
                truncation=True,
                max_length=512,  # BERT token limit
                return_tensors="pt"
            )
            
            # Move tensors to device
            tokenized = {key: val.to(self.device) for key, val in tokenized.items()}
            
            return tokenized
            
        except Exception as e:
            logger.error(f"Error tokenizing batch: {str(e)}")
            raise
    
    def _predict_batch(self, tokenized_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run FinBERT forward pass on tokenized batch.
        
        Args:
            tokenized_batch: Dictionary of tokenized tensors
        
        Returns:
            Logits tensor [batch_size, 3]
        """
        try:
            with torch.no_grad():
                # Forward pass
                outputs = self.model(**tokenized_batch)
                logits = outputs.logits  # Shape: [batch_size, 3]
            
            return logits
            
        except Exception as e:
            logger.error(f"Error during model inference: {str(e)}")
            raise
    
    def _postprocess_logits(self, logits: torch.Tensor, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Convert logits to sentiment predictions with probabilities.
        
        Args:
            logits: Raw model output tensor [batch_size, 3]
            ids: List of article IDs corresponding to each prediction
        
        Returns:
            List of sentiment prediction dicts
        """
        try:
            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=1)  # Shape: [batch_size, 3]
            
            # Get predicted class (highest probability)
            predicted_classes = torch.argmax(probabilities, dim=1)
            
            # Convert to CPU and numpy for processing
            probabilities = probabilities.cpu().numpy()
            predicted_classes = predicted_classes.cpu().numpy()
            
            # Build result list
            predictions = []
            for i, article_id in enumerate(ids):
                pred_class = int(predicted_classes[i])
                sentiment_label = self.label_map[pred_class]
                
                # Extract probability scores
                scores = {
                    'positive': float(probabilities[i][0]),
                    'neutral': float(probabilities[i][1]),
                    'negative': float(probabilities[i][2])
                }
                
                prediction = {
                    'id': article_id,
                    'sentiment': sentiment_label,
                    'scores': scores
                }
                
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error post-processing logits: {str(e)}")
            return []


# Test/Example usage
if __name__ == "__main__":
    try:
        logger.info("Starting FinBERT client test")
        
        # Initialize client (downloads model on first run)
        client = FinBERTClient(
            model_name="yiyanghkust/finbert-tone",
            batch_size=8
        )
        
        # Sample preprocessed articles (matching finbert_preprocessor output format)
        sample_articles = [
            {
                'id': 'a9c73f3c4e',
                'text': 'Reliance Industries Reports Strong Q3 Results. Reliance Industries Limited announced robust quarterly results with revenue growth of 15% year-over-year.',
                'metadata': {
                    'title': 'Reliance Industries Reports Strong Q3 Results',
                    'url': 'https://example.com/news/reliance-q3',
                    'published': '2025-10-03T09:00:00Z',
                    'source': 'gemini',
                    'relevance_score': 0.95,
                    'weight': 1.0
                }
            },
            {
                'id': 'b2f84a9d1c',
                'text': 'Market Update: Tech Stocks Show Weakness. Technology stocks declined today amid concerns about rising interest rates and regulatory scrutiny.',
                'metadata': {
                    'title': 'Market Update: Tech Stocks Show Weakness',
                    'url': 'https://example.com/news/tech-decline',
                    'published': '2025-10-03T10:30:00Z',
                    'source': 'rss',
                    'relevance_score': 0.72,
                    'weight': 1.0
                }
            },
            {
                'id': 'c3e95b8a2d',
                'text': 'Company announces quarterly dividend. The board approved a dividend payment in line with expectations.',
                'metadata': {
                    'title': 'Dividend Announcement',
                    'url': 'https://example.com/news/dividend',
                    'published': '2025-10-03T11:00:00Z',
                    'source': 'rss',
                    'relevance_score': 0.68,
                    'weight': 1.0
                }
            }
        ]
        
        # Analyze sentiment
        results = client.analyze(sample_articles)
        
        # Display results
        print(f"\n{'='*70}")
        print(f"Sentiment Analysis Results")
        print(f"{'='*70}")
        print(f"Analyzed {len(results)} articles\n")
        
        for result in results:
            print(f"ID: {result['id']}")
            print(f"Sentiment: {result['sentiment'].upper()}")
            print(f"Scores:")
            print(f"  - Positive: {result['scores']['positive']:.4f}")
            print(f"  - Neutral:  {result['scores']['neutral']:.4f}")
            print(f"  - Negative: {result['scores']['negative']:.4f}")
            print(f"{'-'*70}\n")
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in test execution: {str(e)}")
        import traceback
        traceback.print_exc()