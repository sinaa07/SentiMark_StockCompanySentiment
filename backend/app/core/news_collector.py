"""
News Collector Module for NSE Stock Market Sentiment Analysis
Handles news collection, caching, and preparation for FinBERT processing

Dependencies: rss_manager.py, content_filter.py, news_database.py
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sqlite3
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from .llm_web_searcher import LLMWebSearcher  
from app.utils import resolve_path
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsCollector:
    """
    Main news collection orchestrator that coordinates between
    user input processing and downstream sentiment analysis
    """
    
    def __init__(self, database_path: str = None, cache_expiry_days: int = 3):
        # Resolve to an absolute path anchored at project root
        # If no path is provided, use default path resolved from project root
        if database_path is None:
            self.db_path = resolve_path("data/news_cache.db")
        elif os.path.isabs(database_path):
            self.db_path = database_path
        else:
            self.db_path = resolve_path(database_path)
            
        self.cache_expiry_days = cache_expiry_days
        self.max_articles_per_company = 15
        self.min_articles_required = 1
        
        # Initialize supporting modules (will be imported when available)
        self.rss_manager = None
        self.content_filter = None
        self.news_database = None
        
        self._initialize_modules()
        self.llm_searcher = LLMWebSearcher()
        self._ensure_news_tables()
    
    def _initialize_modules(self):
        """Initialize supporting modules"""
        try:
            # Import RSS Manager
            from .rss_manager import RSSManager
            self.rss_manager = RSSManager()
            
            #from content_filter import ContentFilter
            #self.content_filter = ContentFilter()
            #from news_database import NewsDatabase
            #self.news_database = NewsDatabase(self.db_path)
            
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
            Dict containing news articles ready for downstream processing
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
                'ready_for_processing': False
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
                
                # Convert to list of dictionaries, normalized to RSS schema
                # Expected by downstream filtering: title, description, link, published, source
                articles = []
                for row in cached_rows:
                    articles.append({
                        'title': row['article_title'],
                        'description': row['article_content'] or '',
                        'link': row['article_url'] or '',
                        'source': row['source_name'] or '',
                        'published': row['published_date'] or '',
                        'relevance_score': row['relevance_score']
                    })
                
                logger.info(f"Found {len(articles)} cached articles for {company_symbol}")
                return articles
                
        except sqlite3.Error as e:
            logger.error(f"Database error checking cache for {company_symbol}: {e}")
            return None
    
    def fetch_fresh_news(self, company_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch fresh news from RSS sources and Gemini LLM when cache miss occurs.
        """
        company_symbol = company_data.get('symbol', '')
        company_name = company_data.get('company_name', '')
        search_terms = company_data.get('search_terms', [company_name])
        
        logger.info(f"Fetching fresh news for {company_symbol}")
    
        # Check if RSS manager is available
        if not self.rss_manager:
            logger.error("RSS Manager not initialized")
            return []
    
        try:
            # Fetch from RSS sources using RSS Manager
            rss_result = self.rss_manager.fetch_all_rss_feeds()
            raw_articles = rss_result.get('articles', []) if rss_result.get('success', False) else []
            logger.info(f"Fetched {len(raw_articles)} raw articles from RSS sources")
    
            # Map RSS field names to expected format
            rss_articles = self.map_rss_articles_to_expected_format(raw_articles)
            filtered_rss = self._filter_articles_basic(rss_articles, company_data)
    
            # Deduplicate RSS internally
            unique_rss = {}
            for art in filtered_rss:
                key = (art['title'].lower(), art['link'])
                if key not in unique_rss:
                    unique_rss[key] = art
            filtered_rss = list(unique_rss.values())[:self.max_articles_per_company]
            for art in filtered_rss:
                art['weight'] = 1.0
    
            all_articles = filtered_rss
    
            # Fetch from Gemini LLM Searcher (MVP)
            try:
                llm_articles = self.llm_searcher.search_news(company_name, company_symbol)
                if llm_articles:
                    logger.info(f"Gemini fetched {len(llm_articles)} articles")
                    # Convert Gemini articles into RSS-like structure for consistency
                    for art in llm_articles:
                        all_articles.append({
                            'title': art['title'],
                            'description': art['summary'],
                            'link': art['url'],
                            'published': art['published'].isoformat(),
                            'source': art['source'],
                            'relevance_score': art['relevance_score'],
                            'weight': art['weight']
                        })
            except Exception as e:
                logger.error(f"Gemini LLM fetch failed: {e}")
    
            # Deduplicate (Gemini > RSS priority)
            deduped_articles = self._deduplicate_articles(all_articles)
    
            logger.info(f"Found {len(deduped_articles)} total articles after merging RSS + Gemini")
            return deduped_articles
    
        except Exception as e:
            logger.error(f"Error fetching fresh news for {company_symbol}: {e}")
            return []

    def map_rss_articles_to_expected_format(self, raw_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        mapped_articles = []
        for article in raw_articles:
            try:
                mapped_article = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),  # Keep original
                    'link': article.get('link', ''),               # Keep original  
                    'published': article.get('published', ''),     # Keep original
                    'source': article.get('source', ''),
                    'author': article.get('author', ''),
                    'relevance_score': 0.0
                }
                if mapped_article['title'] and mapped_article['link']:
                    mapped_articles.append(mapped_article)
            except Exception as e:
                logger.debug(f"Error mapping article: {e}")
                continue
        
        return mapped_articles

    def _filter_articles_basic(self, articles, company_data):
        """Fixed basic filtering with better matching logic"""
        company_terms = company_data.get("search_terms", [])
        company_name = company_data.get('company_name', '')
        company_symbol = company_data.get('symbol', '')
        
        # Normalize all search terms to lowercase
        search_terms = []
        
        # Add search terms from company_data
        if company_terms:
            search_terms.extend([term.lower() for term in company_terms if isinstance(term, str)])
        
        # Add company name and variations
        if company_name:
            search_terms.append(company_name.lower())
            # Add company name without common suffixes
            clean_name = self._clean_company_name(company_name).lower()
            if clean_name and clean_name != company_name.lower():
                search_terms.append(clean_name)
        
        # Add symbol
        if company_symbol:
            search_terms.append(company_symbol.lower())
        
        # Remove duplicates
        search_terms = list(set(search_terms))
        
        logger.info(f"Filtering with search terms: {search_terms}")
        
        relevant_articles = []
        
        for article in articles:
            title = article.get("title", "").lower()
            description = article.get("description", "").lower()
            article_text = f"{title} {description}"
            
            is_relevant = False
            matched_terms = []
            
            # Check for matches with any search term
            for term in search_terms:
                if term in article_text:
                    is_relevant = True
                    matched_terms.append(term)
                    break
                
                # Also check for partial matches (words within the term)
                term_words = term.split()
                if len(term_words) > 1:
                    # Check if all words of the term appear in the article
                    if all(word in article_text for word in term_words):
                        is_relevant = True
                        matched_terms.append(term)
                        break
            
            if is_relevant:
                article["relevance_score"] = 0.7
                article["matched_terms"] = matched_terms
                relevant_articles.append(article)
                logger.debug(f"Matched article: {title[:60]}... (terms: {matched_terms})")
        
        logger.info(f"Found {len(relevant_articles)} relevant articles out of {len(articles)} total")
        return relevant_articles

    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate articles by URL. Gemini articles take priority over RSS.
        """
        seen_urls = set()
        deduped = []
    
        # Sort so that Gemini articles come first
        sorted_articles = sorted(articles, key=lambda a: 0 if a.get("source") == "gemini" else 1)
    
        for art in sorted_articles:
            url = art.get('link') or art.get('url')
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped.append(art)
    
        return deduped

    def _clean_company_name(self, company_name: str) -> str:
        """Remove common company suffixes - this method is missing from news_collector.py"""
        if not company_name:
            return ""
        
        clean_name = company_name.strip()
        suffixes = [' Limited', ' Ltd', ' Pvt', ' Private', ' Company', ' Corp', ' Corporation', ' Inc']
        
        for suffix in suffixes:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)].strip()
                break
        
        return clean_name

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
                
                # Insert new articles (map from RSS schema to DB columns)
                for article in articles:
                    cursor.execute('''
                        INSERT INTO news_cache 
                        (company_symbol, article_title, article_content, article_url, 
                         source_name, published_date, relevance_score, cached_date, expires_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        company_symbol,
                        article.get('title', ''),
                        article.get('description', ''),
                        article.get('link', ''),
                        article.get('source', ''),
                        article.get('published', ''),
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
        Format the final output for downstream processing
        """
        return {
            'company_data': company_data,
            'articles': articles,
            'article_count': len(articles),
            'success': len(articles) >= self.min_articles_required,
            'processing_time': round(processing_time, 2),
            'from_cache': from_cache,
            'cache_expiry_days': self.cache_expiry_days,
            'ready_for_processing': len(articles) > 0,
            'metadata': {
                'collection_timestamp': datetime.now().isoformat(),
                'min_articles_required': self.min_articles_required,
                'max_articles_limit': self.max_articles_per_company
            }
        }
    
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
        "timestamp": "2025-09-22T09:45:00Z",
        "source": "nse_database",
        "ready_for_news_scraper": True
    }
    
    print("Testing News Collector with RSS Manager Integration...")
    print("=" * 60)
    
    # Test news collection
    result = collector.collect_company_news(sample_company_data)
    
    print(f"Collection Success: {result['success']}")
    print(f"Articles Found: {result['article_count']}")
    print(f"Processing Time: {result['processing_time']} seconds")
    print(f"From Cache: {result['from_cache']}")
    print(f"Ready for Processing: {result['ready_for_processing']}")
    
    # Show sample articles if found
    if result.get('articles'):
        print(f"\nSample Articles:")
        for i, article in enumerate(result['articles'][:3], 1):
            print(f"{i}. {article.get('title', 'No Title')[:80]}...")
            print(f"   Source: {article.get('source', 'Unknown')}")
            print(f"   Relevance: {article.get('relevance_score', 0.0)}")
    
    # Test cache stats
    print("\nCache Statistics:")
    stats = collector.get_cache_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test cache cleanup
    print(f"\nCleanup Result: {collector.cleanup_expired_cache()} entries removed")
    
    print("\nNews Collector with RSS Manager integration testing completed!")