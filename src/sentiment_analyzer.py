import requests
import time
import logging
from typing import List, Dict
import streamlit as st
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

class FinBERTAPIAnalyzer:
    def __init__(self):
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        self.api_url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        self.client = InferenceClient(token=self.hf_token) if self.hf_token else None
        
    def analyze_single(self, text: str) -> Dict:
        """
        Analyze sentiment of single text using Hugging Face API
        Returns: {"label": "positive/negative/neutral", "confidence": float}
        """
        if not text or len(text.strip()) < 3:
            return {"label": "neutral", "confidence": 0.0}
        
        # Try with InferenceClient first (more reliable)
        if self.client:
            try:
                result = self.client.text_classification(
                    text=text,
                    model="ProsusAI/finbert"
                )
                if result and len(result) > 0:
                    return {
                        "label": result[0]["label"].lower(),
                        "confidence": result[0]["score"]
                    }
            except Exception as e:
                logging.warning(f"InferenceClient failed: {e}")
        
        # Fallback to direct API call
        return self._api_call(text)
    
    def _api_call(self, text: str, max_retries: int = 3) -> Dict:
        """Direct API call with retries and error handling"""
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], list):
                            # Format: [[{"label": "POSITIVE", "score": 0.9}]]
                            predictions = result[0]
                        else:
                            # Format: [{"label": "POSITIVE", "score": 0.9}]
                            predictions = result
                        
                        # Find the highest confidence prediction
                        best_prediction = max(predictions, key=lambda x: x['score'])
                        
                        return {
                            "label": best_prediction["label"].lower(),
                            "confidence": best_prediction["score"]
                        }
                
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    wait_time = 2 ** attempt  # Exponential backoff
                    logging.info(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    logging.error(f"API call failed: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        # Return neutral if all attempts fail
        return {"label": "neutral", "confidence": 0.0}
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze multiple texts with rate limiting
        """
        results = []
        
        for i, text in enumerate(texts):
            result = self.analyze_single(text)
            results.append(result)
            
            # Rate limiting: wait between requests to avoid hitting limits
            if i < len(texts) - 1:  # Don't wait after the last request
                time.sleep(0.1)  # 100ms between requests
        
        return results
    
    def get_overall_sentiment(self, articles: List[Dict]) -> Dict:
        """
        Calculate overall sentiment from multiple articles
        """
        if not articles:
            return {
                "overall": "neutral",
                "distribution": {"positive": 0, "negative": 0, "neutral": 0},
                "total_articles": 0
            }
        
        # Count sentiments
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_confidence = 0
        
        for article in articles:
            sentiment = article.get('sentiment', 'neutral')
            confidence = article.get('confidence', 0.0)
            
            sentiment_counts[sentiment] += 1
            total_confidence += confidence
        
        # Calculate percentages
        total_articles = len(articles)
        distribution = {
            sentiment: count / total_articles 
            for sentiment, count in sentiment_counts.items()
        }
        
        # Determine overall sentiment
        overall = max(sentiment_counts, key=sentiment_counts.get)
        
        # Calculate average confidence
        avg_confidence = total_confidence / total_articles if total_articles > 0 else 0.0
        
        return {
            "overall": overall,
            "distribution": distribution,
            "total_articles": total_articles,
            "average_confidence": avg_confidence
        }


# Streamlit cached version
@st.cache_resource
def load_sentiment_analyzer():
    """Load sentiment analyzer with Streamlit caching"""
    return FinBERTAPIAnalyzer()


# Utility functions for easy use
def analyze_news_sentiment(news_articles: List[Dict]) -> List[Dict]:
    """
    Process news articles and add sentiment analysis
    Expected input: [{"headline": str, "summary": str, "url": str, ...}, ...]
    Returns: Same list with added sentiment data
    """
    analyzer = load_sentiment_analyzer()
    
    for article in news_articles:
        # Combine headline and summary for better analysis
        text_to_analyze = article.get('headline', '')
        if article.get('summary'):
            text_to_analyze += f". {article['summary']}"
        
        # Get sentiment
        sentiment_result = analyzer.analyze_single(text_to_analyze)
        
        # Add sentiment data to article
        article['sentiment'] = sentiment_result['label']
        article['confidence'] = sentiment_result['confidence']
    
    return news_articles


# Summary functions moved to summarizer.py