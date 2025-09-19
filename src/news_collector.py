"""
News Collector Module for NSE Stock Market Sentiment Analysis
Handles news collection, caching, and preparation for FinBERT processing

Dependencies: rss_manager.py, content_filter.py, news_database.py, finbert_preprocessor.py
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sqlite3
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsCollector:
    """
    Main news collection orchestrator that coordinates between
    user input processing and FinBERT preparation
    """
    
    def __init__(self, database_path: str = "nse_stocks.db", cache_expiry_days: int = 3):
        self.db_path = database_path
        self.cache_expiry_days = cache_expiry_days
        self.max_articles_per_company = 15
        self.min_articles_required = 3
        
        # Initialize supporting modules (will be imported when available)
        self.rss_manager = None
        self.content_filter = None
        self.news_database = None
        self.finbert_preprocessor = None
        
        self._initialize_modules()
        self._ensure_news_tables()
    
    def _initialize_modules(self):
        """Initialize supporting modules - placeholder for now"""
        try:
            # These imports will work when the modules are created
            # from rss_manager import RSSManager
            # from content_filter import ContentFilter
            # from news_database import NewsDatabase
            # from finbert_preprocessor import FinBERTPreprocessor
            
            self.rss_manager = RSSManager()
            self.content_filter = ContentFilter()
            self.news_database = NewsDatabase(self.db_path)
            self.finbert_preprocessor = FinBERTPreprocessor()
            
            logger.info("Supporting modules initialized successfully")
        except ImportError as e:
            logger.warning(f"Some modules not yet available: {e}")
            # For MVP, we'll implement basic functionality inline
    
    def _ensure_news_tables(self):
        """Create news cache tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # News cache table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS news_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_symbol TEXT NOT NULL,
                        article_title TEXT NOT NULL,
                        article_content TEXT,
                        article_url TEXT,
                        source_name TEXT,
                        published_date TEXT,
                        relevance_score REAL DEFAULT 0.0,
                        cached_date TEXT NOT NULL,
                        expires_date TEXT NOT NULL,
                        FOREIGN KEY (company_symbol) REFERENCES companies (symbol)
                    )
                ''')
                
                # Create indexes for performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_news_symbol_date 
                    ON news_cache(company_symbol, expires_date)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_news_published_date 
                    ON news_cache(published_date DESC)
                ''')
                
                conn.commit()
                logger.info("News cache tables created successfully")
                
        except sqlite3.Error as e:
            logger.error(f"Database error creating news tables: {e}")
            raise
    
    def collect_company_news(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main orchestrator function that collects news for a company
        
        Args:
            company_data: Output from user_input_processor.py containing:
                - symbol: Company ticker symbol
                - company_name: Full company name
                - search_terms: List of search keywords
                - validated: Boolean confirmation
                - Other metadata
        
        Returns:
            Dict containing news articles ready for FinBERT processing
        """
        start_time = time.time()
        company_symbol = company_data.get('symbol', '').upper()
        company_name = company_data.get('company_name', '')
        
        logger.info(f"Starting news collection for {company_symbol} ({company_name})")
        
        try:
            # Step 1: Check cache first
            cached_articles = self.check_cache_first(company_symbol)
            if cached_articles:
                logger.info(f"Found {len(cached_articles)} cached articles for {company_symbol}")
                processing_time = time.time() - start_time
                return self._format_final_output(company_data, cached_articles, processing_time, from_cache=True)
            
            # Step 2: Fetch fresh news if cache miss
            logger.info(f"Cache miss for {company_symbol}, fetching fresh news...")
            fresh_articles = self.fetch_fresh_news(company_data)
            
            # Step 3: Store in cache for future requests
            if fresh_articles:
                self.store_news_cache(company_symbol, fresh_articles)
                logger.info(f"Cached {len(fresh_articles)} articles for {company_symbol}")
            
            processing_time = time.time() - start_time
            return self._format_final_output(company_data, fresh_articles, processing_time, from_cache=False)
            
        except Exception as e:
            logger.error(f"Error collecting news for {company_symbol}: {e}")
            # Return empty result with error info
            return {
                'company_data': company_data,
                'articles': [],
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'from_cache': False,
                'ready_for_finbert': False
            }
    
    def check_cache_first(self, company_symbol: str) -> Optional[List[Dict[str, Any]]]:
        """
        Check if we have valid cached news for the company
        
        Args:
            company_symbol: NSE ticker symbol
            
        Returns:
            List of cached articles if found and valid, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                current_time = datetime.now().isoformat()
                
                cursor.execute('''
                    SELECT * FROM news_cache 
                    WHERE company_symbol = ? 
                    AND expires_date > ?
                    ORDER BY published_date DESC
                ''', (company_symbol, current_time))
                
                cached_rows = cursor.fetchall()
                
                if not cached_rows:
                    return None
                
                # Convert to list of dictionaries
                articles = []
                for row in cached_rows:
                    articles.append({
                        'title': row['article_title'],
                        'content': row['article_content'],
                        'url': row['article_url'],
                        'source': row['source_name'],
                        'published_date': row['published_date'],
                        'relevance_score': row['relevance_score']
                    })
                
                logger.info(f"Found {len(articles)} cached articles for {company_symbol}")
                return articles
                
        except sqlite3.Error as e:
            logger.error(f"Database error checking cache for {company_symbol}: {e}")
            return None
    
    def fetch_fresh_news(self, company_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch fresh news from RSS sources when cache miss occurs
        
        Args:
            company_data: Company information from user input processor
            
        Returns:
            List of relevant articles
        """
        company_symbol = company_data.get('symbol', '')
        company_name = company_data.get('company_name', '')
        search_terms = company_data.get('search_terms', [company_name])
        
        logger.info(f"Fetching fresh news for {company_symbol}")
        
        # For MVP implementation - basic RSS fetching
        # This will be replaced by proper RSS manager module
        articles = self._fetch_rss_articles_mvp(search_terms)
        
        # Filter for relevance
        relevant_articles = self._filter_articles_mvp(articles, company_data)
        
        # Sort by recency and limit count
        relevant_articles.sort(key=lambda x: x.get('published_date', ''), reverse=True)
        relevant_articles = relevant_articles[:self.max_articles_per_company]
        
        logger.info(f"Found {len(relevant_articles)} relevant articles for {company_symbol}")
        return relevant_articles
    
    def _fetch_rss_articles_mvp(self, search_terms: List[str]) -> List[Dict[str, Any]]:
        """
        MVP implementation of RSS fetching
        This is a placeholder that will be replaced by proper RSS manager
        """
        # Placeholder RSS sources for MVP
        rss_sources = [
            {
                'name': 'Economic Times Business',
                'url': 'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
                'priority': 1
            },
            {
                'name': 'Moneycontrol',
                'url': 'https://www.moneycontrol.com/rss/business.xml',
                'priority': 2
            },
            {
                'name': 'Business Standard',
                'url': 'https://www.business-standard.com/rss/markets-106.rss',
                'priority': 3
            }
        ]
        
        all_articles = []
        
        # For MVP, return mock articles
        # In actual implementation, this will fetch from RSS sources
        logger.info("MVP: Using placeholder RSS fetching")
        
        # Mock articles for testing
        mock_articles = [
            {
                'title': f'Sample financial news about {search_terms[0] if search_terms else "company"}',
                'content': 'Sample content for testing purposes. This would contain actual news content.',
                'url': 'https://example.com/news/1',
                'source': 'Economic Times',
                'published_date': datetime.now().isoformat(),
                'raw_content': 'Full article content would be here'
            }
        ]
        
        all_articles.extend(mock_articles)
        return all_articles
    
    def _filter_articles_mvp(self, articles: List[Dict[str, Any]], company_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        MVP implementation of simple article filtering
        """
        company_name = company_data.get('company_name', '').lower()
        company_symbol = company_data.get('symbol', '').lower()
        search_terms = [term.lower() for term in company_data.get('search_terms', [])]
        
        relevant_articles = []
        
        for article in articles:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            
            # Simple keyword matching
            is_relevant = False
            
            # Check if company name or symbol in title (high relevance)
            if company_name in title or company_symbol in title:
                is_relevant = True
                article['relevance_score'] = 0.9
            
            # Check search terms in title
            elif any(term in title for term in search_terms):
                is_relevant = True
                article['relevance_score'] = 0.8
            
            # Check company name in content (lower relevance)
            elif company_name in content or company_symbol in content:
                is_relevant = True
                article['relevance_score'] = 0.6
            
            if is_relevant:
                relevant_articles.append(article)
        
        return relevant_articles
    
    def store_news_cache(self, company_symbol: str, articles: List[Dict[str, Any]]):
        """
        Store fetched articles in cache for future requests
        
        Args:
            company_symbol: NSE ticker symbol
            articles: List of articles to cache
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cached_date = datetime.now().isoformat()
                expires_date = (datetime.now() + timedelta(days=self.cache_expiry_days)).isoformat()
                
                # Clear existing cache for this company first
                cursor.execute('DELETE FROM news_cache WHERE company_symbol = ?', (company_symbol,))
                
                # Insert new articles
                for article in articles:
                    cursor.execute('''
                        INSERT INTO news_cache 
                        (company_symbol, article_title, article_content, article_url, 
                         source_name, published_date, relevance_score, cached_date, expires_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        company_symbol,
                        article.get('title', ''),
                        article.get('content', ''),
                        article.get('url', ''),
                        article.get('source', ''),
                        article.get('published_date', ''),
                        article.get('relevance_score', 0.0),
                        cached_date,
                        expires_date
                    ))
                
                conn.commit()
                logger.info(f"Cached {len(articles)} articles for {company_symbol}")
                
        except sqlite3.Error as e:
            logger.error(f"Database error storing cache for {company_symbol}: {e}")
    
    def _format_final_output(self, company_data: Dict[str, Any], articles: List[Dict[str, Any]], 
                           processing_time: float, from_cache: bool) -> Dict[str, Any]:
        """
        Format the final output for FinBERT processing
        """
        return {
            'company_data': company_data,
            'articles': articles,
            'article_count': len(articles),
            'success': len(articles) >= self.min_articles_required,
            'processing_time': round(processing_time, 2),
            'from_cache': from_cache,
            'cache_expiry_days': self.cache_expiry_days,
            'ready_for_finbert': len(articles) > 0,
            'finbert_ready_data': self._prepare_finbert_input(articles, company_data),
            'metadata': {
                'collection_timestamp': datetime.now().isoformat(),
                'min_articles_required': self.min_articles_required,
                'max_articles_limit': self.max_articles_per_company
            }
        }
    
    def _prepare_finbert_input(self, articles: List[Dict[str, Any]], company_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare articles for FinBERT sentiment analysis
        This is a basic implementation - will be enhanced in finbert_preprocessor.py
        """
        finbert_input = {
            'company_symbol': company_data.get('symbol', ''),
            'company_name': company_data.get('company_name', ''),
            'articles_for_analysis': [],
            'total_articles': len(articles),
            'ready_for_processing': True
        }
        
        for i, article in enumerate(articles):
            processed_article = {
                'id': f"{company_data.get('symbol', '')}_{i}",
                'title': article.get('title', ''),
                'content': article.get('content', '')[:2000],  # Limit content length for FinBERT
                'source': article.get('source', ''),
                'published_date': article.get('published_date', ''),
                'relevance_score': article.get('relevance_score', 0.0)
            }
            finbert_input['articles_for_analysis'].append(processed_article)
        
        return finbert_input
    
    def cleanup_expired_cache(self):
        """
        Background cleanup job to remove expired cache entries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                current_time = datetime.now().isoformat()
                
                # Delete expired entries
                cursor.execute('DELETE FROM news_cache WHERE expires_date < ?', (current_time,))
                deleted_count = cursor.rowcount
                
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired cache entries")
                
                return deleted_count
                
        except sqlite3.Error as e:
            logger.error(f"Database error during cache cleanup: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cached news data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total cached articles
                cursor.execute('SELECT COUNT(*) FROM news_cache')
                total_articles = cursor.fetchone()[0]
                
                # Unique companies with cache
                cursor.execute('SELECT COUNT(DISTINCT company_symbol) FROM news_cache')
                unique_companies = cursor.fetchone()[0]
                
                # Articles by source
                cursor.execute('''
                    SELECT source_name, COUNT(*) as count 
                    FROM news_cache 
                    GROUP BY source_name 
                    ORDER BY count DESC
                ''')
                source_stats = dict(cursor.fetchall())
                
                return {
                    'total_cached_articles': total_articles,
                    'companies_with_cache': unique_companies,
                    'articles_by_source': source_stats,
                    'cache_expiry_days': self.cache_expiry_days
                }
                
        except sqlite3.Error as e:
            logger.error(f"Database error getting cache stats: {e}")
            return {'error': str(e)}


# Example usage and testing
if __name__ == "__main__": 
    # Initialize news collector
    collector = NewsCollector()
    
    # Example input from user_input_processor.py
    sample_company_data = {
        "symbol": "ADANIPORTS",
        "company_name": "Adani Ports and Special Economic Zone Limited",
        "search_terms": ["ADANIPORTS", "Adani Ports", "Adani SEZ"],
        "series": "EQ",
        "isin": "INE742F01042",
        "validated": True,
        "timestamp": "2025-09-11T09:45:00Z",
        "source": "nse_database",
        "ready_for_news_scraper": True
    }
    
    print("Testing News Collector...")
    print("=" * 50)
    
    # Test news collection
    result = collector.collect_company_news(sample_company_data)
    
    print(f"Collection Success: {result['success']}")
    print(f"Articles Found: {result['article_count']}")
    print(f"Processing Time: {result['processing_time']} seconds")
    print(f"From Cache: {result['from_cache']}")
    print(f"Ready for FinBERT: {result['ready_for_finbert']}")
    
    # Test cache stats
    print("\nCache Statistics:")
    stats = collector.get_cache_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test cache cleanup
    print(f"\nCleanup Result: {collector.cleanup_expired_cache()} entries removed")
    
    print("\nNews Collector testing completed!")