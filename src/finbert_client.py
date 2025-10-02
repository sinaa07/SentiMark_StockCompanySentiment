# finbert_client.py - FinBERT API Integration with Sentiment Aggregation
import asyncio
import aiohttp
import logging
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinBERTClient:
    """
    Client for FinBERT API integration with weighted sentiment aggregation.
    Handles API calls, response processing, and company-level sentiment calculation.
    """
    
    def __init__(self, 
                 api_url: str = "https://api-inference.huggingface.co/models/ProsusAI/finbert",
                 api_token: str = None,
                 rate_limit_delay: float = 1.0,
                 max_retries: int = 3):
        """
        Initialize FinBERT client.
        
        Args:
            api_url: FinBERT API endpoint (default: Hugging Face)
            api_token: API authentication token
            rate_limit_delay: Delay between API calls (seconds)
            max_retries: Maximum retry attempts for failed requests
        """
        self.api_url = api_url
        self.api_token = api_token
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.headers = {}
        
        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"
        self.headers["Content-Type"] = "application/json"
    
    async def analyze_company_sentiment(self, preprocessed_data: Dict[str, Any], 
                                      company_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for company sentiment analysis.
        
        Args:
            preprocessed_data: Output from finbert_preprocessor.prepare_for_finbert()
            company_data: Company information for context and weighting
        
        Returns:
            Complete sentiment analysis results with aggregated scores
        """
        if not preprocessed_data.get('batches'):
            logger.warning("No batches provided for sentiment analysis")
            return self._create_empty_result(company_data)
        
        logger.info(f"Starting sentiment analysis for {company_data.get('symbol', 'Unknown')} "
                   f"with {len(preprocessed_data['batches'])} batches")
        
        try:
            # Process all batches through FinBERT API
            batch_results = await self._process_all_batches(preprocessed_data['batches'])
            
            # Aggregate chunk-level results to article-level
            article_sentiments = self._aggregate_chunks_to_articles(batch_results, preprocessed_data)
            
            # Apply light heuristic to correct obvious misclassifications using titles
            self._apply_title_heuristics(article_sentiments)
            
            # Calculate company-level weighted sentiment
            company_sentiment = self._calculate_company_sentiment(article_sentiments, 
                                                               preprocessed_data, 
                                                               company_data)
            
            # Create final result structure
            final_result = {
                'company': company_data,
                'sentiment_summary': company_sentiment,
                'article_details': article_sentiments,
                'processing_info': {
                    'total_articles': preprocessed_data['articles_processed'],
                    'total_chunks': preprocessed_data['total_chunks'],
                    'batches_processed': len(preprocessed_data['batches']),
                    'articles_analyzed': len(article_sentiments),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'model_used': 'ProsusAI/finbert'
                }
            }
            
            logger.info(f"Sentiment analysis complete for {company_data.get('symbol', 'Unknown')}: "
                       f"{company_sentiment['overall_sentiment']:.3f}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            error_result = self._create_empty_result(company_data)
            error_result['error'] = str(e)
            return error_result
    
    async def _process_all_batches(self, batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all batches through FinBERT API with rate limiting."""
        batch_results = []
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch['texts'])} chunks")
            
            try:
                # Send batch to FinBERT
                api_response = await self._send_to_finbert(batch)
                
                # Process API response
                processed_response = self._process_api_response(api_response, batch)
                batch_results.append(processed_response)
                
                # Rate limiting (except for last batch)
                if i < len(batches) - 1:
                    await asyncio.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"Error processing batch {i+1}: {str(e)}")
                # Continue with next batch, log the failure
                batch_results.append(self._create_error_batch_result(batch, str(e)))
                continue
        
        return batch_results
    
    async def _send_to_finbert(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Send batch to FinBERT API with retry logic."""
        texts = batch['texts']
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    # Prepare payload
                    payload = {"inputs": texts}
                    
                    async with session.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            logger.debug(f"API response received for {len(texts)} texts")
                            return result
                        
                        elif response.status == 429:  # Rate limit
                            wait_time = (attempt + 1) * 2  # Exponential backoff
                            logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt+1}")
                            await asyncio.sleep(wait_time)
                            
                        elif response.status == 503:  # Service unavailable (model loading)
                            wait_time = 20  # Wait for model to load
                            logger.warning(f"Model loading, waiting {wait_time}s before retry {attempt+1}")
                            await asyncio.sleep(wait_time)
                            
                        else:
                            error_text = await response.text()
                            raise Exception(f"API error {response.status}: {error_text}")
            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt+1}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"API call attempt {attempt+1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise e
        
        raise Exception(f"Failed to get API response after {self.max_retries} attempts")
    
    def _process_api_response(self, api_response: List[List[Dict[str, Any]]], 
                            batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process FinBERT API response and map to chunks."""
        processed_chunks = []
        
        try:
            chunk_metadata = batch['chunk_metadata']
            
            for i, chunk_result in enumerate(api_response):
                if i >= len(chunk_metadata):
                    logger.warning(f"API response has more results than chunks")
                    break
                
                metadata = chunk_metadata[i]
                
                # Parse FinBERT scores (expecting 3-class: positive, negative, neutral)
                sentiment_scores = self._parse_finbert_scores(chunk_result)
                
                # Calculate single sentiment score (-1 to +1)
                sentiment_score = sentiment_scores['positive'] - sentiment_scores['negative']
                
                chunk_sentiment = {
                    'chunk_id': metadata['chunk_id'],
                    'text': metadata['text'][:100] + "..." if len(metadata['text']) > 100 else metadata['text'],
                    'sentiment_score': sentiment_score,
                    'raw_scores': sentiment_scores,
                    'confidence': max(sentiment_scores.values()),  # Highest class probability
                    'metadata': metadata['metadata']
                }
                
                processed_chunks.append(chunk_sentiment)
            
            return {
                'batch_id': batch['batch_id'],
                'chunks': processed_chunks,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing API response: {str(e)}")
            return self._create_error_batch_result(batch, str(e))
    
    def _parse_finbert_scores(self, chunk_result: List[Dict[str, Any]]) -> Dict[str, float]:
        """Parse FinBERT API response into standardized scores."""
        scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        try:
            for item in chunk_result:
                label = item.get('label', '').lower()
                score = float(item.get('score', 0.0))
                
                if label in scores:
                    scores[label] = score
                elif label == 'label_0':  # Some models use numeric labels
                    scores['negative'] = score
                elif label == 'label_1':
                    scores['neutral'] = score
                elif label == 'label_2':
                    scores['positive'] = score
            
            return scores
            
        except Exception as e:
            logger.error(f"Error parsing FinBERT scores: {str(e)}")
            return scores
    
    def _aggregate_chunks_to_articles(self, batch_results: List[Dict[str, Any]], 
                                    preprocessed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Aggregate chunk-level sentiment to article-level sentiment."""
        article_groups = {}
        
        # Group chunks by original article (prefer stable link when available)
        for batch_result in batch_results:
            if not batch_result.get('success'):
                continue
                
            for chunk in batch_result['chunks']:
                metadata = chunk['metadata']
                link = metadata.get('link')
                article_key = link if link else f"{metadata['source']}_{metadata['original_title']}"
                
                if article_key not in article_groups:
                    article_groups[article_key] = {
                        'chunks': [],
                        'metadata': metadata
                    }
                
                article_groups[article_key]['chunks'].append(chunk)
        
        # Calculate article-level sentiment
        article_sentiments = []
        
        for article_key, article_data in article_groups.items():
            chunks = article_data['chunks']
            metadata = article_data['metadata']
            
            if not chunks:
                continue
            
            # Average sentiment across chunks of the same article
            avg_sentiment = sum(chunk['sentiment_score'] for chunk in chunks) / len(chunks)
            avg_confidence = sum(chunk['confidence'] for chunk in chunks) / len(chunks)
            
            # Aggregate raw scores
            avg_raw_scores = {
                'positive': sum(chunk['raw_scores']['positive'] for chunk in chunks) / len(chunks),
                'negative': sum(chunk['raw_scores']['negative'] for chunk in chunks) / len(chunks),
                'neutral': sum(chunk['raw_scores']['neutral'] for chunk in chunks) / len(chunks)
            }
            
            article_sentiment = {
                'source': metadata['source'],
                'title': metadata['original_title'],
                'published_date': metadata['published_date'],
                'chunk_count': len(chunks),
                'sentiment_score': avg_sentiment,
                'confidence': avg_confidence,
                'raw_scores': avg_raw_scores,
                'chunks_detail': chunks  # Keep for debugging/transparency
            }
            
            article_sentiments.append(article_sentiment)
        
        # Sort by published date (newest first)
        article_sentiments.sort(key=lambda x: x['published_date'] or '', reverse=True)
        
        return article_sentiments
    
    def _calculate_company_sentiment(self, article_sentiments: List[Dict[str, Any]], 
                                   preprocessed_data: Dict[str, Any],
                                   company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate company-level sentiment using simple unweighted average (no relevance weighting)."""
        if not article_sentiments:
            return self._create_empty_sentiment()

        try:
            overall_sentiment = sum(article['sentiment_score'] for article in article_sentiments) / len(article_sentiments)

            # Confidence: simple average of article confidences without extra weighting
            confidence = sum(article['confidence'] for article in article_sentiments) / len(article_sentiments)

            sentiment_label = self._classify_sentiment(overall_sentiment)

            return {
                'overall_sentiment': round(overall_sentiment, 4),
                'sentiment_label': sentiment_label,
                'confidence': round(confidence, 4),
                'article_count': len(article_sentiments),
                'total_weight': 0.0,
                'sentiment_distribution': self._get_sentiment_distribution(article_sentiments)
            }

        except Exception as e:
            logger.error(f"Error calculating company sentiment: {str(e)}")
            return self._create_empty_sentiment()
    
    def _calculate_recency_weight(self, published_date: str, current_time: datetime) -> float:
        """Calculate recency weight with decay over 3 days."""
        try:
            if not published_date:
                return 0.5  # Default weight for unknown dates
            
            # Parse date (handle different formats)
            if isinstance(published_date, str):
                pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            else:
                pub_date = published_date
            
            # Calculate hours since publication
            hours_diff = (current_time - pub_date).total_seconds() / 3600
            
            # Decay function: full weight for first 24 hours, then exponential decay
            if hours_diff <= 24:
                return 1.0
            elif hours_diff <= 72:  # 3 days
                return 0.5 + 0.5 * (1 - (hours_diff - 24) / 48)  # Linear decay from 1.0 to 0.5
            else:
                return 0.3  # Minimum weight for old articles
                
        except Exception as e:
            logger.error(f"Error calculating recency weight: {str(e)}")
            return 0.5
    
    def _calculate_overall_confidence(self, article_sentiments: List[Dict[str, Any]], 
                                    total_weight: float) -> float:
        """Calculate overall confidence based on agreement and data quality."""
        try:
            if not article_sentiments:
                return 0.0
            
            # Base confidence from individual article confidences
            avg_confidence = sum(article['confidence'] for article in article_sentiments) / len(article_sentiments)
            
            # Agreement factor (how much do articles agree?)
            sentiments = [article['sentiment_score'] for article in article_sentiments]
            sentiment_std = self._calculate_std(sentiments)
            agreement_factor = max(0, 1 - sentiment_std)  # Lower std = higher agreement
            
            # Data quantity factor
            quantity_factor = min(len(article_sentiments) / 5, 1.0)  # Up to 5 articles for full quantity bonus
            
            # Combined confidence
            overall_confidence = (avg_confidence * 0.5 + agreement_factor * 0.3 + quantity_factor * 0.2)
            
            return min(overall_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating overall confidence: {str(e)}")
            return 0.5
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _classify_sentiment(self, sentiment_score: float) -> str:
        """Classify numerical sentiment into label."""
        if sentiment_score >= 0.2:
            return "bullish"
        elif sentiment_score <= -0.2:
            return "bearish"
        else:
            return "neutral"
    
    def _get_sentiment_distribution(self, article_sentiments: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of sentiment classifications across articles."""
        distribution = {"bullish": 0, "neutral": 0, "bearish": 0}
        
        for article in article_sentiments:
            label = self._classify_sentiment(article['sentiment_score'])
            distribution[label] += 1
        
        return distribution

    def _apply_title_heuristics(self, article_sentiments: List[Dict[str, Any]]):
        """Adjust sentiment for headlines with clear directional cues.
        Only small nudges; does not override model outright.
        """
        try:
            positive_cues = [
                'upgrade', 'overweight', 'outperform', 'beats', 'beat estimates', 'record high',
                'surge', 'soar', 'jumps', 'spikes', 'rallies', 'raises guidance', 'hikes target',
                'profit rises', 'revenue grows', 'strong results', 'wins order', 'secures contract'
            ]
            negative_cues = [
                'downgrade', 'underweight', 'underperform', 'misses', 'miss estimates', 'plunge',
                'falls', 'drops', 'sinks', 'tumbles', 'cuts guidance', 'slashes target',
                'loss widens', 'profit falls', 'revenue declines', 'weak results'
            ]
            pct_up_patterns = [r'\bup\s*\d+%\b', r'\bjump\s*\d+%\b', r'\bsoar\s*\d+%\b', r"\bup\s*\d+\.\d+%\b"]
            pct_down_patterns = [r'\bdown\s*\d+%\b', r'\bfall\s*\d+%\b', r'\bplunge\s*\d+%\b', r"\bdown\s*\d+\.\d+%\b"]

            for article in article_sentiments:
                title = (article.get('title') or '').lower()
                if not title:
                    continue

                adjustment = 0.0
                # Cue-based adjustments
                if any(cue in title for cue in positive_cues):
                    adjustment += 0.05
                if any(cue in title for cue in negative_cues):
                    adjustment -= 0.05

                # Percentage move adjustments
                if any(re.search(pat, title) for pat in pct_up_patterns):
                    adjustment += 0.07
                if any(re.search(pat, title) for pat in pct_down_patterns):
                    adjustment -= 0.07

                if adjustment != 0.0:
                    original = article['sentiment_score']
                    article['sentiment_score'] = max(-1.0, min(1.0, article['sentiment_score'] + adjustment))
                    # Slightly boost confidence if cues align with move
                    article['confidence'] = max(0.0, min(1.0, article['confidence'] + abs(adjustment) * 0.3))
                    logger.debug(f"Adjusted sentiment for title cue: '{title[:60]}...' {original:.3f} -> {article['sentiment_score']:.3f}")
        except Exception as e:
            logger.warning(f"Title heuristic adjustment failed: {e}")
    
    def _create_empty_result(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create empty result structure for error cases."""
        return {
            'company': company_data,
            'sentiment_summary': self._create_empty_sentiment(),
            'article_details': [],
            'processing_info': {
                'total_articles': 0,
                'total_chunks': 0,
                'batches_processed': 0,
                'analysis_timestamp': datetime.now().isoformat(),
                'model_used': 'ProsusAI/finbert'
            }
        }
    
    def _create_empty_sentiment(self) -> Dict[str, Any]:
        """Create empty sentiment summary."""
        return {
            'overall_sentiment': 0.0,
            'sentiment_label': 'neutral',
            'confidence': 0.0,
            'article_count': 0,
            'total_weight': 0.0,
            'sentiment_distribution': {"bullish": 0, "neutral": 0, "bearish": 0}
        }
    
    def _create_error_batch_result(self, batch: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Create error result for failed batch."""
        return {
            'batch_id': batch.get('batch_id', -1),
            'chunks': [],
            'success': False,
            'error': error
        }


# Usage example and integration point
async def main():
    """Example usage of FinBERT client."""
    try:
        # Initialize client (replace with your API token)
        client = FinBERTClient(
            api_token="your_huggingface_token_here",  # Replace with actual token
            rate_limit_delay=1.2  # Adjust based on your rate limits
        )
        
        # Mock data from finbert_preprocessor.py
        preprocessed_data = {
            'batches': [
                {
                    'batch_id': 0,
                    'texts': [
                        'Reliance Industries Reports Strong Q3 Results. Reliance Industries Limited announced robust quarterly results with revenue growth of 15% year-over-year.'
                    ],
                    'chunk_metadata': [
                        {
                            'chunk_id': 'Economic_Times_Reliance_Industries_Reports_Strong_Q3_Results_0',
                            'text': 'Reliance Industries Reports Strong Q3 Results. Reliance Industries Limited announced robust quarterly results with revenue growth of 15% year-over-year.',
                            'metadata': {
                                'source': 'Economic Times',
                                'published_date': '2024-01-15T10:30:00',
                                'original_title': 'Reliance Industries Reports Strong Q3 Results',
                                'chunk_position': '1/1'
                            }
                        }
                    ]
                }
            ],
            'total_chunks': 1,
            'articles_processed': 1
        }
        
        company_data = {
            'symbol': 'RELIANCE',
            'name': 'Reliance Industries Limited',
            'sector': 'Oil & Gas'
        }
        
        # Analyze sentiment
        result = await client.analyze_company_sentiment(preprocessed_data, company_data)
        
        print(f"\nSentiment Analysis Results:")
        print(f"Company: {result['company']['name']}")
        print(f"Overall Sentiment: {result['sentiment_summary']['overall_sentiment']:.3f}")
        print(f"Label: {result['sentiment_summary']['sentiment_label']}")
        print(f"Confidence: {result['sentiment_summary']['confidence']:.3f}")
        print(f"Articles Analyzed: {result['sentiment_summary']['article_count']}")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())