import requests
import feedparser
import logging
import time
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
import json
from bs4 import BeautifulSoup

def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    # remove scripts and styles
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)
   
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RSSManager:
    """RSS feed manager for Indian financial news sources"""
    
    def __init__(self, request_timeout=10, max_workers=3, retry_attempts=2):
        self.request_timeout = request_timeout
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        
        # RSS source configuration
        self.rss_sources = self.get_rss_source_config()
        
        # Request session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive'
        })
        
        logger.info(f"RSSManager initialized with {len(self.rss_sources)} sources, timeout={request_timeout}s")
    
    def get_rss_source_config(self) -> Dict[str, Dict[str, Any]]:
        """
        RSS source configuration with URLs and parsing settings
        Returns dictionary of source configurations
        """
        return {
            'economic_times': {
                'name': 'Economic Times Business',
                'url': 'https://economictimes.indiatimes.com/rssfeedstopstories.cms',
                'backup_url': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
                'format': 'xml',
                'encoding': 'utf-8',
                'priority': 1,
                'active': True,
                'description_field': 'summary',
                'date_field': 'published'
            },
            #'business_standard': {
            #    'name': 'Business Standard',
            #    'url': 'https://www.business-standard.com/rss/markets-106.rss',
            #    'backup_url': 'https://www.business-standard.com/rss/finance-103.rss',
            #    'format': 'xml',
            #    'encoding': 'utf-8',
            #    'priority': 2,
            #    'active': True,
            #    'description_field': 'summary',
            #    'date_field': 'published'
            #},
            'livemint': {
                'name': 'LiveMint Markets',
                'url': 'https://www.livemint.com/rss/markets',
                'backup_url': 'https://www.livemint.com/rss/companies',
                'format': 'xml',
                'encoding': 'utf-8',
                'priority': 3,
                'active': True,
                'description_field': 'summary',
                'date_field': 'published'
            },
            'hindu_businessline': {
                'name': 'Hindu BusinessLine',
                'url': 'https://www.thehindubusinessline.com/markets/stock-markets/feeder/default.rss',
                'backup_url': 'https://www.thehindubusinessline.com/companies/feeder/default.rss',
                'format': 'xml',
                'encoding': 'utf-8',
                'priority': 4,
                'active': True,
                'description_field': 'summary',
                'date_field': 'published'
            },
            #'financial_express': {
            #    'name': 'Financial Express',
            #    'url': 'https://www.financialexpress.com/market/rss/',
            #    'backup_url': 'https://www.financialexpress.com/industry/rss/',
            #    'format': 'xml',
            #    'encoding': 'utf-8',
            #    'priority': 5,
            #    'active': True,
            #    'description_field': 'summary',
            #    'date_field': 'published'
            #},
            'ndtv_business': {
                'name': 'NDTV Business',
                'url': 'https://feeds.feedburner.com/ndtvprofit-latest',
                'backup_url': 'https://www.ndtv.com/business/rss',
                'format': 'xml',
                'encoding': 'utf-8',
                'priority': 6,
                'active': True,
                'description_field': 'summary',
                'date_field': 'published'
            },
            'zee_business': {
                'name': 'Zee Business',
                'url': 'https://zeenews.india.com/rss/business.xml',
                'backup_url': 'https://zeenews.india.com/rss/stock-market.xml',
                'format': 'xml',
                'encoding': 'utf-8',
                'priority': 7,
                'active': True,
                'description_field': 'summary',
                'date_field': 'published'
            },
            'moneycontrol': {
                'name': 'Moneycontrol Markets',
                'url': 'https://www.moneycontrol.com/rss/business.xml',
                'backup_url': 'https://www.moneycontrol.com/rss/marketsnews.xml',
                'format': 'xml',
                'encoding': 'utf-8',
                'priority': 8,
                'active': False,  # Disabled due to 403 errors
                'description_field': 'summary',
                'date_field': 'published'
            },
            'news18_business': {
                'name': 'News18 Business',
                'url': 'https://www.news18.com/rss/business.xml',
                'backup_url': 'https://www.news18.com/rss/india.xml',
                'format': 'xml',
                'encoding': 'utf-8',
                'priority': 9,
                'active': True,
                'description_field': 'summary',
                'date_field': 'published'
            },
            #'outlook_money': {
            #    'name': 'Outlook Money',
            #    'url': 'https://www.outlookindia.com/rss/business',
            #    'backup_url': 'https://www.outlookindia.com/business/rss',
            #    'format': 'xml',
            #    'encoding': 'utf-8',
            #    'priority': 10,
            #    'active': True,
            #    'description_field': 'summary',
            #    'date_field': 'published'
            #}
        }
    
    def fetch_all_rss_feeds(self) -> Dict[str, Any]:
        """
        Main function: Parallel fetch from all RSS sources
        Returns combined results with success/failure tracking
        """
        logger.info("Starting parallel RSS fetch from all sources")
        start_time = time.time()
        
        # Filter active sources
        active_sources = {
            source_id: config for source_id, config in self.rss_sources.items()
            if config.get('active', True)
        }
        
        if not active_sources:
            logger.error("No active RSS sources configured")
            return self._create_empty_result("No active sources")
        
        # Parallel execution
        results = {}
        failed_sources = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all fetch tasks
            future_to_source = {
                executor.submit(self._fetch_single_rss_feed, source_id, config): source_id
                for source_id, config in active_sources.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_source, timeout=self.request_timeout + 5):
                source_id = future_to_source[future]
                
                try:
                    source_result = future.result()
                    
                    if source_result.get('success', False):
                        results[source_id] = source_result
                        logger.info(f"Successfully fetched {len(source_result.get('articles', []))} articles from {source_id}")
                    else:
                        failed_sources.append(source_id)
                        logger.warning(f"Failed to fetch from {source_id}: {source_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    failed_sources.append(source_id)
                    logger.error(f"Exception fetching from {source_id}: {str(e)}")
        
        # Handle failures and compile final results
        final_result = self._compile_fetch_results(results, failed_sources)
        
        execution_time = time.time() - start_time
        final_result['execution_time_seconds'] = round(execution_time, 2)
        
        logger.info(f"RSS fetch completed in {execution_time:.2f}s. Success: {len(results)}/{len(active_sources)} sources")
        
        # Handle failures if any
        if failed_sources:
            failure_result = self.handle_rss_failures(failed_sources)
            final_result['failure_handling'] = failure_result
        
        return final_result
    
    def _fetch_single_rss_feed(self, source_id: str, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch RSS feed from a single source with retry logic
        """
        source_name = source_config.get('name', source_id)
        primary_url = source_config.get('url')
        backup_url = source_config.get('backup_url')
        
        # Try primary URL first
        urls_to_try = [primary_url]
        if backup_url:
            urls_to_try.append(backup_url)
        
        last_error = None
        
        for attempt in range(self.retry_attempts):
            for url_index, url in enumerate(urls_to_try):
                try:
                    logger.debug(f"Fetching {source_name} from {url} (attempt {attempt + 1})")
                    
                    response = self.session.get(
                        url,
                        timeout=self.request_timeout,
                        allow_redirects=True
                    )
                    
                    response.raise_for_status()
                    
                    # Parse the RSS content
                    articles = self.parse_rss_content(response.content, source_config)
                    
                    if articles:
                        return {
                            'success': True,
                            'source_id': source_id,
                            'source_name': source_name,
                            'articles': articles,
                            'url_used': url,
                            'is_backup_url': url_index > 0,
                            'fetch_timestamp': datetime.now(timezone.utc).isoformat(),
                            'article_count': len(articles)
                        }
                    else:
                        last_error = "No articles parsed from RSS feed"
                        
                except requests.exceptions.Timeout:
                    last_error = f"Timeout after {self.request_timeout}s"
                    logger.warning(f"{source_name} timeout on attempt {attempt + 1}")
                    
                except requests.exceptions.RequestException as e:
                    last_error = f"Request error: {str(e)}"
                    logger.warning(f"{source_name} request error: {str(e)}")
                    
                except Exception as e:
                    last_error = f"Parsing error: {str(e)}"
                    logger.error(f"{source_name} parsing error: {str(e)}")
            
            # Wait before retry (exponential backoff)
            if attempt < self.retry_attempts - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        
        # All attempts failed
        return {
            'success': False,
            'source_id': source_id,
            'source_name': source_name,
            'error': last_error or "Unknown error",
            'attempts': self.retry_attempts,
            'urls_tried': urls_to_try
        }
    
    def parse_rss_content(self, rss_content: bytes, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract articles from RSS XML/JSON content
        Returns list of standardized article dictionaries
        """
        try:
            source_name = source_config.get('name', 'Unknown')
            
            # Use feedparser for robust RSS/Atom parsing
            feed = feedparser.parse(rss_content)
            
            if feed.bozo and feed.bozo_exception:
                logger.warning(f"RSS parsing warning for {source_name}: {feed.bozo_exception}")
            
            if not hasattr(feed, 'entries') or not feed.entries:
                logger.warning(f"No entries found in RSS feed for {source_name}")
                return []
            
            articles = []
            
            for entry in feed.entries:
                try:
                    article = self._extract_article_data(entry, source_config)
                    if article:
                        articles.append(article)
                        
                except Exception as e:
                    logger.debug(f"Error parsing individual article from {source_name}: {str(e)}")
                    continue
            
            logger.debug(f"Parsed {len(articles)} articles from {source_name}")
            return articles
            
        except Exception as e:
            logger.error(f"Error parsing RSS content: {str(e)}")
            return []
    
    def _extract_article_data(self, entry: Any, source_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract standardized article data from RSS entry
        """
        try:
            # Extract basic fields
            title = self._safe_get_text(entry, 'title', '').strip()
            link = self._safe_get_text(entry, 'link', '').strip()
            
            if not title or not link:
                return None
    
            # Extract description/summary
            description_field = source_config.get('description_field', 'summary')
            description = self._safe_get_text(entry, description_field, '')
    
            # Fallback description fields
            if not description:
                for field in ['content_encoded', 'summary', 'description', 'content']:
                    description = self._safe_get_text(entry, field, '')
                    if description:
                        break
    
            # Clean description HTML (always run, even if already plain)
            description = clean_html(description)
    
            # Extract and parse date
            date_field = source_config.get('date_field', 'published')
            published_date = self._extract_publish_date(entry, date_field)
    
            # Extract author if available
            author = self._safe_get_text(entry, 'author', '')
    
            # Clean title too
            title = clean_html(title)
    
            # Create standardized article object
            article = {
                'title': title,
                'description': description,
                'link': link,
                'published': published_date,
                'author': author,
                'source': source_config.get('name', 'Unknown'),
                'source_id': source_config.get('source_id', ''),
                
                # Additional metadata
                'guid': self._safe_get_text(entry, 'guid', ''),
                'category': self._extract_categories(entry),
                'content_length': len(description),
                'extraction_timestamp': datetime.now(timezone.utc).isoformat(),
                
                # Raw entry for debugging (optional)
                'raw_entry_keys': list(entry.keys()) if hasattr(entry, 'keys') else []
            }
            
            return article
            
        except Exception as e:
            logger.debug(f"Error extracting article data: {str(e)}")
            return None

    def _safe_get_text(self, entry: Any, field: str, default: str = '') -> str:
        """
        Safely extract text content from RSS entry field
        """
        try:
            value = getattr(entry, field, None)
            
            if value is None:
                return default
            
            # Handle different content formats
            if isinstance(value, str):
                return value
            elif isinstance(value, list) and value:
                return str(value[0])
            elif hasattr(value, 'value'):
                return str(value.value)
            elif hasattr(value, 'text'):
                return str(value.text)
            else:
                return str(value)
                
        except Exception:
            return default
    
    def _extract_publish_date(self, entry: Any, date_field: str = 'published') -> str:
        """
        Extract and standardize publication date
        """
        try:
            # Try different date fields
            date_fields = [date_field, 'published', 'updated', 'pubDate']
            
            for field in date_fields:
                date_value = getattr(entry, field, None)
                
                if date_value:
                    # Handle feedparser's parsed time
                    if hasattr(entry, f"{field}_parsed") and getattr(entry, f"{field}_parsed"):
                        parsed_time = getattr(entry, f"{field}_parsed")
                        if parsed_time:
                            dt = datetime(*parsed_time[:6], tzinfo=timezone.utc)
                            return dt.isoformat()
                    
                    # Handle string dates
                    if isinstance(date_value, str):
                        return date_value
            
            # Default to current time if no date found
            return datetime.now(timezone.utc).isoformat()
            
        except Exception:
            return datetime.now(timezone.utc).isoformat()
    
    def _extract_categories(self, entry: Any) -> List[str]:
        """
        Extract categories/tags from RSS entry
        """
        try:
            categories = []
            
            # Check tags field
            if hasattr(entry, 'tags') and entry.tags:
                for tag in entry.tags:
                    if hasattr(tag, 'term'):
                        categories.append(tag.term)
            
            # Check category field
            if hasattr(entry, 'category'):
                if isinstance(entry.category, str):
                    categories.append(entry.category)
                elif isinstance(entry.category, list):
                    categories.extend(entry.category)
            
            return categories
            
        except Exception:
            return []
    
    def _compile_fetch_results(self, successful_results: Dict[str, Any], failed_sources: List[str]) -> Dict[str, Any]:
        """
        Compile results from all RSS sources into final response
        """
        all_articles = []
        source_stats = {}
        
        # Collect articles from successful sources
        for source_id, result in successful_results.items():
            articles = result.get('articles', [])
            all_articles.extend(articles)
            
            source_stats[source_id] = {
                'source_name': result.get('source_name', source_id),
                'article_count': len(articles),
                'success': True,
                'url_used': result.get('url_used', ''),
                'is_backup_url': result.get('is_backup_url', False)
            }
        
        # Add failed source stats
        for source_id in failed_sources:
            source_config = self.rss_sources.get(source_id, {})
            source_stats[source_id] = {
                'source_name': source_config.get('name', source_id),
                'article_count': 0,
                'success': False,
                'error': 'Failed to fetch'
            }
        
        return {
            'success': len(successful_results) > 0,
            'total_articles': len(all_articles),
            'articles': all_articles,
            'sources_attempted': len(self.rss_sources),
            'sources_successful': len(successful_results),
            'sources_failed': len(failed_sources),
            'source_stats': source_stats,
            'fetch_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def handle_rss_failures(self, failed_sources: List[str]) -> Dict[str, Any]:
        """
        Handle RSS source failures - continue with available sources
        Returns failure analysis and recommendations
        """
        logger.info(f"Handling failures for {len(failed_sources)} sources: {failed_sources}")
        
        total_sources = len(self.rss_sources)
        successful_sources = total_sources - len(failed_sources)
        
        # Determine severity
        if successful_sources == 0:
            severity = 'critical'
            recommendation = 'No RSS sources available. Check network connection and source URLs.'
        elif successful_sources < total_sources / 2:
            severity = 'high'
            recommendation = 'More than half of RSS sources failed. Content may be limited.'
        else:
            severity = 'low'
            recommendation = 'Some RSS sources failed but majority are working. Content should be sufficient.'
        
        failure_analysis = {
            'severity': severity,
            'failed_sources': failed_sources,
            'successful_sources': successful_sources,
            'total_sources': total_sources,
            'success_rate': round(successful_sources / total_sources * 100, 1),
            'recommendation': recommendation,
            'failed_source_details': []
        }
        
        # Add details for each failed source
        for source_id in failed_sources:
            source_config = self.rss_sources.get(source_id, {})
            failure_analysis['failed_source_details'].append({
                'source_id': source_id,
                'source_name': source_config.get('name', source_id),
                'primary_url': source_config.get('url', ''),
                'backup_url': source_config.get('backup_url', ''),
                'priority': source_config.get('priority', 99)
            })
        
        return failure_analysis
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """
        Create empty result structure for error cases
        """
        return {
            'success': False,
            'total_articles': 0,
            'articles': [],
            'sources_attempted': 0,
            'sources_successful': 0,
            'sources_failed': 0,
            'error': reason,
            'fetch_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def test_single_source(self, source_id: str) -> Dict[str, Any]:
        """
        Test a single RSS source for debugging
        """
        if source_id not in self.rss_sources:
            return {'error': f'Source {source_id} not configured'}
        
        source_config = self.rss_sources[source_id]
        logger.info(f"Testing RSS source: {source_id}")
        
        return self._fetch_single_rss_feed(source_id, source_config)
    
    def get_source_status(self) -> Dict[str, Any]:
        """
        Get status of all configured RSS sources
        """
        return {
            'total_sources': len(self.rss_sources),
            'active_sources': len([s for s in self.rss_sources.values() if s.get('active', True)]),
            'sources': {
                source_id: {
                    'name': config.get('name', source_id),
                    'url': config.get('url', ''),
                    'active': config.get('active', True),
                    'priority': config.get('priority', 99)
                }
                for source_id, config in self.rss_sources.items()
            }
        }
    
    def close(self):
        """
        Clean up resources
        """
        if hasattr(self, 'session'):
            self.session.close()
        logger.info("RSSManager closed")

# Convenience function for quick usage
def create_rss_manager(timeout=10, max_workers=3, retry_attempts=2) -> RSSManager:
    """Factory function to create RSSManager instance"""
    return RSSManager(timeout, max_workers, retry_attempts)

# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize RSS manager
        rss_manager = RSSManager()
        
        print("=== RSS Manager Test ===")
        
        # Test source configuration
        print("\n1. Source Configuration:")
        status = rss_manager.get_source_status()
        print(json.dumps(status, indent=2))
        
        # Test single source
        print("\n2. Testing Single Source (Economic Times):")
        single_test = rss_manager.test_single_source('economic_times')
        if single_test.get('success'):
            print(f"   Success: {single_test.get('article_count', 0)} articles")
            if single_test.get('articles'):
                print(f"   Sample: {single_test['articles'][0].get('title', 'N/A')[:80]}...")
        else:
            print(f"   Failed: {single_test.get('error', 'Unknown error')}")
        
        # Test parallel fetch from all sources
        print("\n3. Parallel Fetch Test:")
        all_results = rss_manager.fetch_all_rss_feeds()
        
        if all_results.get('success'):
            print(f"   Total Articles: {all_results.get('total_articles', 0)}")
            print(f"   Successful Sources: {all_results.get('sources_successful', 0)}/{all_results.get('sources_attempted', 0)}")
            print(f"   Execution Time: {all_results.get('execution_time_seconds', 0)}s")
            
            # Show sample articles
            articles = all_results.get('articles', [])
            if articles:
                print("\n   Sample Articles:")
                for i, article in enumerate(articles[:3], 1):
                    print(f"   {i}. {article.get('title', 'N/A')[:60]}...")
                    print(f"      Source: {article.get('source', 'N/A')}")
                    print(f"      Published: {article.get('published', 'N/A')}")
        else:
            print(f"   Failed: {all_results.get('error', 'Unknown error')}")
        
        # Clean up
        rss_manager.close()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Check network connection and RSS URLs.")