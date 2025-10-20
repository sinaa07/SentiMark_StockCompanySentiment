# pipeline.py - Sentiment Analysis Pipeline Orchestrator
import argparse
import json
import logging
from typing import Dict, List, Any
from .user_input_processor import UserInputProcessor
from .news_collector import NewsCollector
from .finbert_preprocessor import FinBERTPreprocessor
from .finbert_client import FinBERTClient
from app.utils import resolve_path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentPipeline:
    """
    End-to-end sentiment analysis pipeline orchestrator.
    Workflow: User Input → News Collection → Preprocessing → Local FinBERT → Aggregation
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize pipeline components.
        
        Args:
            db_path: Path to NSE stocks database
        """
        logger.info("Initializing Sentiment Pipeline...")
        
        try:
            if db_path is None:
                db_path = resolve_path("data/nse_stocks.db")
            
            # Use separate paths for stock DB and cache DB
            stocks_db_path = db_path
            news_cache_db_path = resolve_path("data/news_cache.db")   
            
            # Initialize components
            self.user_input = UserInputProcessor(db_path=stocks_db_path)
            self.news_collector = NewsCollector(database_path=news_cache_db_path)  
            self.preprocessor = FinBERTPreprocessor()

            # ✅ Minimal change: DON'T load FinBERT here
            self.client = None
            
            logger.info("Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}")
            raise
    
    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Execute end-to-end sentiment analysis pipeline.
        
        Args:
            user_query: Company name or symbol from user
        
        Returns:
            Structured JSON result with company info, sentiment summary, and articles
        """
        try:
            logger.info(f"Starting pipeline for query: {user_query}")
            
            # Step 1: Process user input and get company data
            company_data = self.user_input.handle_user_selection(user_query)
            if not company_data or company_data.get("error"):
                logger.warning(f"User input failed: {company_data.get('error', 'Unknown error')}")
                return {
                    "stage": "user_input",
                    "error": company_data.get("error", "Company not found")
                }
            
            logger.info(f"Company identified: {company_data.get('symbol')} - {company_data.get('name')}")
            
            # Step 2: Collect news articles (RSS + Gemini + Cache)
            news_result = self.news_collector.collect_company_news(company_data)
            articles = news_result.get("articles", [])
            
            if not articles:
                logger.warning("No articles found for company")
                return {
                    "stage": "news_collection",
                    "error": "No articles found",
                    "company": company_data
                }
            
            logger.info(f"Collected {len(articles)} articles")
            
            # Step 3: Preprocess articles for FinBERT
            preprocessed_chunks = self.preprocessor.prepare_for_finbert(articles)
            
            if not preprocessed_chunks:
                logger.warning("No valid articles after preprocessing")
                return {
                    "stage": "preprocessing",
                    "error": "No valid articles after preprocessing",
                    "company": company_data
                }
            
            logger.info(f"Preprocessed {len(preprocessed_chunks)} chunks")
            
            # Step 4: Run local FinBERT sentiment analysis
            # ✅ Minimal change: Lazy-load FinBERT client only when needed
            if self.client is None:
                self.client = FinBERTClient()

            predictions = self.client.analyze(preprocessed_chunks)
            
            if not predictions:
                logger.warning("No predictions returned from FinBERT")
                return {
                    "stage": "sentiment_inference",
                    "error": "No predictions returned",
                    "company": company_data
                }
            
            logger.info(f"Generated {len(predictions)} sentiment predictions")
            
            # Step 5: Aggregate results into final structured output
            final_result = self._aggregate_results(company_data, preprocessed_chunks, predictions)
            
            logger.info(f"Pipeline complete: {final_result['sentiment_summary']['sentiment_label']}")
            return final_result
            
        except Exception as e:
            logger.exception("Pipeline execution failed")
            return {
                "stage": "pipeline",
                "error": str(e)
            }
    
    def _aggregate_results(self, 
                          company_data: Dict[str, Any],
                          preprocessed_articles: List[Dict[str, Any]],
                          predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate sentiment predictions with article metadata into final JSON output.
        
        Args:
            company_data: Company information
            preprocessed_articles: List of preprocessed articles with metadata
            predictions: List of sentiment predictions from FinBERT
        
        Returns:
            Structured JSON response for API/frontend consumption
        """
        try:
            # Build prediction lookup by ID
            prediction_map = {pred['id']: pred for pred in predictions}
            
            # Merge predictions with article metadata
            enriched_articles = []
            sentiment_counts = {"bullish": 0, "neutral": 0, "bearish": 0}
            total_positive = 0.0
            total_negative = 0.0
            
            for article in preprocessed_articles:
                article_id = article['id']
                prediction = prediction_map.get(article_id)
                
                if not prediction:
                    logger.warning(f"No prediction found for article ID: {article_id}")
                    continue
                
                metadata = article.get('metadata', {})
                sentiment = prediction['sentiment']
                scores = prediction['scores']
                
                # Map sentiment to bullish/bearish/neutral
                if sentiment == 'positive':
                    sentiment_label = 'bullish'
                elif sentiment == 'negative':
                    sentiment_label = 'bearish'
                else:
                    sentiment_label = 'neutral'
                
                sentiment_counts[sentiment_label] += 1
                total_positive += scores['positive']
                total_negative += scores['negative']
                
                # Build enriched article dict
                enriched_article = {
                    "title": metadata.get('title', 'Unknown'),
                    "url": metadata.get('url', ''),
                    "source": metadata.get('source', 'unknown'),
                    "published": metadata.get('published_date', ''),
                    "sentiment": sentiment_label,
                    "positive": round(scores['positive'], 4),
                    "neutral": round(scores['neutral'], 4),
                    "negative": round(scores['negative'], 4)
                }
                
                enriched_articles.append(enriched_article)
            
            # Calculate overall sentiment metrics
            article_count = len(enriched_articles)
            
            if article_count > 0:
                # Overall sentiment score: (avg positive - avg negative)
                avg_positive = total_positive / article_count
                avg_negative = total_negative / article_count
                overall_score = avg_positive - avg_negative
                
                # Classify overall sentiment
                if overall_score >= 0.2:
                    overall_label = "bullish"
                elif overall_score <= -0.2:
                    overall_label = "bearish"
                else:
                    overall_label = "neutral"
                
                # Calculate confidence (based on score magnitude and agreement)
                confidence = min(abs(overall_score) + 0.3, 1.0)  # Simple heuristic
                
            else:
                overall_score = 0.0
                overall_label = "neutral"
                confidence = 0.0
            
            # Build final structured response
            final_result = {
                "company": {
                    "symbol": company_data.get('symbol', 'UNKNOWN'),
                    "name": company_data.get('company_name', 'Unknown Company'),
                    "sector": company_data.get('sector', 'Unknown')
                },
                "article_count": article_count,
                "sentiment_summary": {
                    "bullish": sentiment_counts["bullish"],
                    "neutral": sentiment_counts["neutral"],
                    "bearish": sentiment_counts["bearish"],
                    "overall_score": round(overall_score, 4),
                    "sentiment_label": overall_label,
                    "confidence": round(confidence, 4)
                },
                "articles": enriched_articles
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error aggregating results: {str(e)}")
            return {
                "company": company_data,
                "article_count": 0,
                "sentiment_summary": {
                    "bullish": 0,
                    "neutral": 0,
                    "bearish": 0,
                    "overall_score": 0.0,
                    "sentiment_label": "neutral",
                    "confidence": 0.0
                },
                "articles": [],
                "error": str(e)
            }


class PipelineCLI:
    """CLI runner for the SentimentPipeline"""
    
    def __init__(self):
        """Initialize CLI (no API token needed for local FinBERT)"""
        pass
    
    def parse_args(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description="Run sentiment analysis pipeline for NSE companies"
        )
        parser.add_argument(
            "symbols",
            nargs="+",
            help="Company symbols or names (e.g., RELIANCE, TCS, 'Tata Motors')"
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable debug logging"
        )
        parser.add_argument(
            "--db-path",
            default="data/nse_stocks.db",
            help="Path to NSE stocks database"
        )
        return parser.parse_args()
    
    def run(self):
        """Execute CLI workflow"""
        args = self.parse_args()
        
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize pipeline
        try:
            pipeline = SentimentPipeline(db_path=args.db_path)
        except Exception as e:
            print(f"Failed to initialize pipeline: {str(e)}")
            return
        
        # Process each symbol
        results = {}
        for symbol in args.symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {symbol}")
            logger.info(f"{'='*60}\n")
            
            result = pipeline.run(symbol)
            results[symbol] = result
        
        # Print final results as JSON
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60 + "\n")
        print(json.dumps(results, indent=2))


# Test/Example usage
if __name__ == "__main__":
    # Run CLI if called directly
    PipelineCLI().run()