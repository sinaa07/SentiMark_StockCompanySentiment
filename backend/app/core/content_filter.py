import re
import logging
from typing import List, Dict, Optional, Any, Set
from datetime import datetime, timezone, timedelta
import hashlib
import unicodedata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContentFilter:
    """Simplified content filter focused on deduplication and basic cleanup"""
    
    def __init__(self, relevance_threshold=0.05, max_articles_per_company=50, max_days_old=7):
        """Initialize with very permissive threshold for now"""
        self.relevance_threshold = relevance_threshold  # Much lower threshold
        self.max_articles_per_company = max_articles_per_company
        self.max_days_old = max_days_old
        
        # Simplified stop words for cleanup
        self.stop_words = self._load_stop_words()
        
        logger.info(f"ContentFilter initialized: threshold={relevance_threshold}, max_articles={max_articles_per_company}")
    
    def _load_stop_words(self) -> Set[str]:
        """Load common stop words to ignore during matching"""
        return {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'have', 'has', 'had', 'will', 'would', 'could', 'should',
            'a', 'an', 'this', 'that', 'these', 'those', 'ltd', 'limited',
            'pvt', 'private', 'company', 'corp', 'corporation', 'inc'
        }
    
    def filter_company_articles(self, articles: List[Dict[str, Any]], company_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Simplified filtering: Basic relevance check + deduplication + sorting
        Maintains same input/output format for pipeline compatibility
        """
        try:
            if not articles or not company_data:
                logger.warning("Empty articles or company_data provided")
                return []
            
            logger.info(f"Processing {len(articles)} articles for {company_data.get('company_name', 'Unknown')}")
            
            # Step 1: Basic relevance filtering (very permissive)
            search_terms = [t.lower() for t in company_data.get('search_terms', []) if isinstance(t, str)]
            relevant_articles = []
            for article in articles:
                relevance_score = self.check_article_relevance(
                    article, 
                    company_data.get('company_name', ''), 
                    company_data.get('symbol', '')
                )
                # Boost if any search term appears in title/description (substring match)
                try:
                    text = f"{article.get('title', '')} {article.get('description', '')}".lower()
                    if search_terms and any(term in text for term in search_terms):
                        relevance_score = min(1.0, relevance_score + 0.25)
                except Exception:
                    pass
                
                if relevance_score >= self.relevance_threshold:
                    article['relevance_score'] = relevance_score
                    # Add company identifiers for compatibility
                    article['company_identifiers_found'] = self._find_basic_matches(article, company_data)
                    relevant_articles.append(article)
            
            logger.info(f"Found {len(relevant_articles)} relevant articles (threshold: {self.relevance_threshold})")
            
            if not relevant_articles:
                return []
            
            # Step 2: Remove duplicates
            deduplicated_articles = self.deduplicate_articles(relevant_articles)
            
            # Step 3: Filter by date
            date_filtered_articles = self._filter_by_date(deduplicated_articles)
            
            # Step 4: Sort by relevance and recency
            sorted_articles = self.sort_by_recency(date_filtered_articles)
            
            # Step 5: Limit number of articles
            final_articles = sorted_articles[:self.max_articles_per_company]
            
            logger.info(f"Final result: {len(final_articles)} articles for {company_data.get('symbol', 'Unknown')}")
            
            return final_articles
            
        except Exception as e:
            logger.error(f"Error in filter_company_articles: {str(e)}")
            return []
    
    def check_article_relevance(self, article: Dict[str, Any], company_name: str, ticker: str) -> float:
        """
        Simplified relevance check - very permissive, focuses on basic company matching
        """
        try:
            if not article:
                return 0.0
            
            # Get article text content
            title = self._clean_text(article.get('title', ''))
            description = self._clean_text(article.get('description', ''))
            article_text = f"{title} {description}".lower()
            
            if not article_text.strip():
                return 0.0
            
            relevance_score = 0.0
            
            # 1. Direct ticker matching (if available)
            if ticker and ticker.lower() in article_text:
                relevance_score += 0.8  # High confidence
                logger.debug(f"Direct ticker match: {ticker}")
            
            # 2. Company name matching (various forms)
            if company_name:
                name_score = self._simple_company_match(article_text, company_name)
                relevance_score += name_score * 0.6
            
            # 3. Basic business context (very broad)
            if self._has_business_context(article_text):
                relevance_score += 0.1
            
            # 4. Not obviously irrelevant (weather, sports, etc.)
            if not self._is_obviously_irrelevant(article_text):
                relevance_score += 0.05
            
            # Ensure minimum score for any article with company mention
            if ticker and ticker.lower() in article_text:
                relevance_score = max(relevance_score, 0.1)
            elif company_name and company_name.lower() in article_text:
                relevance_score = max(relevance_score, 0.08)
            
            return min(relevance_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {str(e)}")
            return 0.0
    
    def _simple_company_match(self, article_text: str, company_name: str) -> float:
        """Simplified company name matching"""
        if not company_name:
            return 0.0
        
        company_lower = company_name.lower()
        
        # Exact match
        if company_lower in article_text:
            return 1.0
        
        # Clean version without suffixes
        clean_name = self._clean_company_name(company_name).lower()
        if clean_name and clean_name in article_text:
            return 0.8
        
        # First word match (for common abbreviations)
        first_word = company_lower.split()[0] if company_lower.split() else ""
        if len(first_word) > 3 and first_word in article_text:
            return 0.3
        
        return 0.0
    
    def _clean_company_name(self, company_name: str) -> str:
        """Remove common company suffixes"""
        if not company_name:
            return ""
        
        clean_name = company_name.strip()
        suffixes = [' Ltd', ' Limited', ' Pvt', ' Private', ' Company', ' Corp', ' Corporation', ' Inc']
        
        for suffix in suffixes:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)].strip()
                break
        
        return clean_name
    
    def _has_business_context(self, article_text: str) -> bool:
        """Check for basic business/financial context"""
        business_terms = [
            'revenue', 'profit', 'loss', 'earnings', 'shares', 'stock', 'market',
            'business', 'company', 'financial', 'investment', 'growth', 'results'
        ]
        return any(term in article_text for term in business_terms)
    
    def _is_obviously_irrelevant(self, article_text: str) -> bool:
        """Check for obviously irrelevant content"""
        irrelevant_terms = [
            'weather', 'rainfall', 'temperature', 'sports', 'cricket', 'football',
            'movie', 'film', 'celebrity', 'entertainment', 'recipe', 'cooking'
        ]
        return any(term in article_text for term in irrelevant_terms)
    
    def _find_basic_matches(self, article: Dict[str, Any], company_data: Dict[str, Any]) -> List[str]:
        """Find basic company matches for compatibility with existing pipeline"""
        article_text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        found_matches = []
        
        # Check symbol
        symbol = company_data.get('symbol', '')
        if symbol and symbol.lower() in article_text:
            found_matches.append(f"symbol:{symbol}")
        
        # Check company name
        company_name = company_data.get('company_name', '')
        if company_name and company_name.lower() in article_text:
            found_matches.append(f"company:{company_name}")
        
        return found_matches
    
    def deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles - keeping the deduplication logic as it's solid"""
        try:
            if not articles:
                return []
            
            logger.debug(f"Deduplicating {len(articles)} articles")
            
            seen_hashes = set()
            seen_urls = set()
            seen_titles = {}
            unique_articles = []
            
            for article in articles:
                # Generate content hash
                content_hash = self._generate_article_hash(article)
                url = self._clean_url(article.get('link', ''))
                title = self._clean_text(article.get('title', ''))
                title_normalized = self._normalize_title(title)
                
                is_duplicate = False
                
                # Check for duplicates
                if content_hash and content_hash in seen_hashes:
                    is_duplicate = True
                elif url and url in seen_urls:
                    is_duplicate = True
                elif title_normalized:
                    for existing_title_norm, existing_article in seen_titles.items():
                        similarity = self._calculate_title_similarity(title_normalized, existing_title_norm)
                        if similarity > 0.85:
                            is_duplicate = True
                            # Keep the one with higher relevance score
                            if article.get('relevance_score', 0) > existing_article.get('relevance_score', 0):
                                unique_articles.remove(existing_article)
                                seen_titles[title_normalized] = article
                                unique_articles.append(article)
                            break
                
                if not is_duplicate:
                    if content_hash:
                        seen_hashes.add(content_hash)
                    if url:
                        seen_urls.add(url)
                    if title_normalized:
                        seen_titles[title_normalized] = article
                    unique_articles.append(article)
            
            logger.info(f"Deduplication: {len(articles)} -> {len(unique_articles)} articles")
            return unique_articles
            
        except Exception as e:
            logger.error(f"Error in deduplicate_articles: {str(e)}")
            return articles
    
    def _generate_article_hash(self, article: Dict[str, Any]) -> str:
        """Generate hash for article content comparison"""
        try:
            title = self._clean_text(article.get('title', ''))
            description = self._clean_text(article.get('description', ''))[:200]
            content = f"{title}|{description}".lower()
            return hashlib.md5(content.encode()).hexdigest() if content.strip() else ""
        except Exception:
            return ""
    
    def _clean_url(self, url: str) -> str:
        """Clean and normalize URL for comparison"""
        if not url:
            return ""
        url = re.sub(r'[?&](utm_|ref=|source=|campaign=)[^&]*', '', url)
        url = re.sub(r'[/#]+$', '', url)
        return url.lower().strip()
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for similarity comparison"""
        if not title:
            return ""
        normalized = re.sub(r'\s+', ' ', title.lower().strip())
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        words = normalized.split()
        significant_words = [w for w in words if len(w) > 2 and w not in self.stop_words]
        return ' '.join(significant_words)
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate Jaccard similarity between titles"""
        if not title1 or not title2:
            return 0.0
        
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def sort_by_recency(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort by relevance score then recency"""
        try:
            if not articles:
                return []
            
            logger.debug(f"Sorting {len(articles)} articles")
            
            for article in articles:
                parsed_date = self._parse_article_date(article.get('published', ''))
                article['_parsed_date'] = parsed_date
                article['_relevance_score'] = article.get('relevance_score', 0.0)
            
            sorted_articles = sorted(
                articles,
                key=lambda x: (x['_relevance_score'], x['_parsed_date']),
                reverse=True
            )
            
            # Clean up temporary fields
            for article in sorted_articles:
                article.pop('_parsed_date', None)
                article.pop('_relevance_score', None)
            
            return sorted_articles
            
        except Exception as e:
            logger.error(f"Error sorting articles: {str(e)}")
            return articles
    
    def _parse_article_date(self, date_str: str) -> datetime:
        """Parse publication date - simplified version"""
        if not date_str:
            return datetime.min.replace(tzinfo=timezone.utc)
        
        try:
            # Handle ISO format with Z
            if 'T' in date_str and 'Z' in date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            # Handle ISO format
            if 'T' in date_str:
                return datetime.fromisoformat(date_str)
            
            # Try common formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                try:
                    return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            # Default to now if parsing fails
            return datetime.now(timezone.utc)
            
        except Exception:
            return datetime.now(timezone.utc)
    
    def _filter_by_date(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter articles by date"""
        try:
            if not articles:
                return []
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.max_days_old)
            filtered_articles = []
            
            for article in articles:
                article_date = self._parse_article_date(article.get('published', ''))
                if article_date >= cutoff_date:
                    filtered_articles.append(article)
            
            logger.info(f"Date filtering: {len(articles)} -> {len(filtered_articles)} articles")
            return filtered_articles
            
        except Exception as e:
            logger.error(f"Error filtering by date: {str(e)}")
            return articles
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        return text
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get filter statistics for compatibility"""
        return {
            'configuration': {
                'relevance_threshold': self.relevance_threshold,
                'max_articles_per_company': self.max_articles_per_company,
                'max_days_old': self.max_days_old
            },
            'keyword_categories': {'simplified': 1},  # Placeholder for compatibility
            'total_keywords': 0,  # No longer using keyword lists
            'stop_words_count': len(self.stop_words)
        }

# Keep original factory function for compatibility
def create_content_filter(relevance_threshold=0.05, max_articles=50, max_days=30) -> ContentFilter: #change to 7
    """Factory function to create ContentFilter instance"""
    return ContentFilter(relevance_threshold, max_articles, max_days)

# Simplified test section
if __name__ == "__main__":
    try:
        # Initialize with very permissive settings
        content_filter = ContentFilter(relevance_threshold=0.05)
        
        print("=== Simplified Content Filter Test ===")
        
        # Test with recent dates
        from datetime import datetime, timezone
        recent_date = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        sample_articles = [
            {
                'title': 'TCS Q3 Results: Revenue grows 12% YoY, profit jumps 15%',
                'description': 'Tata Consultancy Services reported strong quarterly results.',
                'link': 'https://example.com/tcs-results-1',
                'published': recent_date,  # Use current date
                'source': 'Economic Times'
            },
            {
                'title': 'Weather update: Heavy rainfall expected in Mumbai',
                'description': 'Meteorological department warning for Mumbai.',
                'link': 'https://example.com/weather-news',
                'published': recent_date,  # Use current date
                'source': 'Moneycontrol'
            }
        ]
        
        company_data = {
            'symbol': 'TCS',
            'company_name': 'Tata Consultancy Services Limited'
        }
        
        print(f"\nTesting with {len(sample_articles)} articles")
        
        # Test filtering
        filtered_articles = content_filter.filter_company_articles(sample_articles, company_data)
        print(f"Filtered articles: {len(filtered_articles)}")
        
        for i, article in enumerate(filtered_articles, 1):
            print(f"{i}. {article['title']}")
            print(f"   Score: {article.get('relevance_score', 0):.3f}")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()