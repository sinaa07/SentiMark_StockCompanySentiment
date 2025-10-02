import re
import logging
from typing import List, Dict, Optional, Any, Set
from datetime import datetime, timezone, timedelta
import hashlib
from collections import defaultdict
import unicodedata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContentFilter:
    """Content filter for company-specific news article relevance and deduplication"""
    
    def __init__(self, relevance_threshold=0.2, max_articles_per_company=50, max_days_old=7):
        self.relevance_threshold = relevance_threshold
        self.max_articles_per_company = max_articles_per_company
        self.max_days_old = max_days_old
        
        # Financial keywords for relevance scoring
        self.financial_keywords = self._load_financial_keywords()
        
        # Stop words for better matching
        self.stop_words = self._load_stop_words()
        
        logger.info(f"ContentFilter initialized: threshold={relevance_threshold}, max_articles={max_articles_per_company}")
    
    def _load_financial_keywords(self) -> Dict[str, List[str]]:
        """
        Load financial keywords for relevance scoring
        Categorized by importance/weight
        """
        return {
            'high_weight': [
                'earnings', 'results', 'profit', 'loss', 'revenue', 'turnover',
                'quarterly', 'annual', 'financial', 'performance', 'growth',
                'dividend', 'bonus', 'split', 'ipo', 'listing', 'merger',
                'acquisition', 'takeover', 'buyback', 'rights issue'
            ],
            'medium_weight': [
                'shares', 'stock', 'market', 'trading', 'volume', 'price',
                'valuation', 'investment', 'investor', 'fund', 'portfolio',
                'analyst', 'rating', 'target', 'recommendation', 'upgrade',
                'downgrade', 'outlook', 'forecast', 'guidance', 'expansion'
            ],
            'low_weight': [
                'company', 'business', 'corporate', 'management', 'board',
                'director', 'ceo', 'chairman', 'announcement', 'news',
                'update', 'launch', 'product', 'service', 'contract',
                'agreement', 'deal', 'partnership', 'collaboration'
            ]
        }
    
    def _load_stop_words(self) -> Set[str]:
        """
        Load common stop words to ignore during matching
        """
        return {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'have', 'has', 'had', 'will', 'would', 'could', 'should',
            'a', 'an', 'this', 'that', 'these', 'those', 'ltd', 'limited',
            'pvt', 'private', 'company', 'corp', 'corporation', 'inc'
        }
    
    def filter_company_articles(self, articles: List[Dict[str, Any]], company_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main filtering function: Filter articles for company relevance
        Returns list of relevant articles sorted by relevance and recency
        """
        try:
            if not articles or not company_data:
                logger.warning("Empty articles or company_data provided")
                return []
            
            logger.info(f"Filtering {len(articles)} articles for {company_data.get('company_name', 'Unknown')}")
            
            # Extract company identifiers
            company_identifiers = self._extract_company_identifiers(company_data)
            
            # Filter for relevance
            relevant_articles = []
            for article in articles:
                relevance_score = self.check_article_relevance(article, company_data.get('company_name', ''), 
                                                             company_data.get('symbol', ''))
                
                if relevance_score >= self.relevance_threshold:
                    article['relevance_score'] = relevance_score
                    article['company_identifiers_found'] = self._find_matching_identifiers(article, company_identifiers)
                    relevant_articles.append(article)
            
            logger.info(f"Found {len(relevant_articles)} relevant articles (threshold: {self.relevance_threshold})")
            
            if not relevant_articles:
                return []
            
            # Remove duplicates
            deduplicated_articles = self.deduplicate_articles(relevant_articles)
            
            # Filter by date
            date_filtered_articles = self._filter_by_date(deduplicated_articles)
            
            # Sort by relevance and recency
            sorted_articles = self.sort_by_recency(date_filtered_articles)
            
            # Limit number of articles
            final_articles = sorted_articles[:self.max_articles_per_company]
            
            logger.info(f"Final filtered result: {len(final_articles)} articles for {company_data.get('symbol', 'Unknown')}")
            
            return final_articles
            
        except Exception as e:
            logger.error(f"Error in filter_company_articles: {str(e)}")
            return []
    
    def check_article_relevance(self, article: Dict[str, Any], company_name: str, ticker: str) -> float:
        """
        Check article relevance using basic string matching
        Returns relevance score between 0.0 and 1.0
        """
        try:
            if not article or not company_name:
                return 0.0
            
            # Get article text content
            title = self._clean_text(article.get('title', ''))
            description = self._clean_text(article.get('description', ''))
            
            # Combine all text for analysis
            article_text = f"{title} {description}".lower()
            
            if not article_text.strip():
                return 0.0
            
            relevance_score = 0.0
            
            # 1. Direct symbol/ticker matching (highest weight)
            if ticker and ticker.lower() in article_text:
                relevance_score += 0.4
                logger.debug(f"Direct ticker match found: {ticker}")
            
            # 2. Company name matching
            company_score = self._calculate_company_name_score(article_text, company_name)
            relevance_score += company_score * 0.3
            
            # 3. Financial keywords matching
            keyword_score = self._calculate_keyword_score(article_text)
            relevance_score += keyword_score * 0.2
            
            # 4. Context relevance (title vs description weight)
            context_score = self._calculate_context_score(title, description, company_name, ticker)
            relevance_score += context_score * 0.1
            
            # Normalize score to [0, 1]
            relevance_score = min(relevance_score, 1.0)
            
            logger.debug(f"Article relevance score: {relevance_score:.3f} for '{title[:50]}...'")
            
            return relevance_score
            
        except Exception as e:
            logger.error(f"Error calculating article relevance: {str(e)}")
            return 0.0
    
    def _extract_company_identifiers(self, company_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract all possible identifiers for the company
        """
        identifiers = {
            'symbols': [],
            'company_names': [],
            'search_terms': []
        }
        
        # Extract symbols
        symbol = company_data.get('symbol', '').strip()
        nse_symbol = company_data.get('nse_symbol', '').strip()
        
        if symbol:
            identifiers['symbols'].append(symbol.upper())
        if nse_symbol and nse_symbol != symbol:
            identifiers['symbols'].append(nse_symbol.upper())
        
        # Extract company names
        company_name = company_data.get('company_name', '').strip()
        if company_name:
            identifiers['company_names'].append(company_name)
            
            # Add cleaned version without suffixes
            cleaned_name = self._clean_company_name(company_name)
            if cleaned_name != company_name:
                identifiers['company_names'].append(cleaned_name)
        
        # Extract search terms if available
        search_terms = company_data.get('search_terms', [])
        if isinstance(search_terms, list):
            identifiers['search_terms'].extend(search_terms)
        
        return identifiers
    
    def _clean_company_name(self, company_name: str) -> str:
        """
        Clean company name by removing common suffixes
        """
        if not company_name:
            return ""
        
        clean_name = company_name.strip()
        
        # Remove common suffixes
        suffixes = [
            ' Ltd', ' Limited', ' Pvt', ' Private', ' Company',
            ' Corp', ' Corporation', ' Inc', ' Incorporated'
        ]
        
        for suffix in suffixes:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)].strip()
                break
        
        return clean_name
    
    def _find_matching_identifiers(self, article: Dict[str, Any], identifiers: Dict[str, List[str]]) -> List[str]:
        """
        Find which company identifiers are present in the article
        """
        article_text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        found_identifiers = []
        
        # Check symbols
        for symbol in identifiers.get('symbols', []):
            if symbol.lower() in article_text:
                found_identifiers.append(f"symbol:{symbol}")
        
        # Check company names
        for name in identifiers.get('company_names', []):
            if name.lower() in article_text:
                found_identifiers.append(f"company:{name}")
        
        # Check search terms
        for term in identifiers.get('search_terms', []):
            if term.lower() in article_text:
                found_identifiers.append(f"search_term:{term}")
        
        return found_identifiers
    
    def _calculate_company_name_score(self, article_text: str, company_name: str) -> float:
        """
        Calculate relevance score based on company name matching
        """
        if not company_name:
            return 0.0
        
        company_name_lower = company_name.lower()
        cleaned_name = self._clean_company_name(company_name).lower()
        
        score = 0.0
        
        # Exact company name match
        if company_name_lower in article_text:
            score += 1.0
        
        # Cleaned company name match
        elif cleaned_name and cleaned_name in article_text:
            score += 0.8
        
        # Partial matching (first two words)
        else:
            words = cleaned_name.split()
            if len(words) >= 2:
                first_two = ' '.join(words[:2])
                if first_two in article_text:
                    score += 0.5
            
            # Individual word matching
            word_matches = 0
            significant_words = [w for w in words if len(w) > 3 and w not in self.stop_words]
            
            for word in significant_words:
                if word in article_text:
                    word_matches += 1
            
            if significant_words:
                score += (word_matches / len(significant_words)) * 0.3
        
        return min(score, 1.0)
    
    def _calculate_keyword_score(self, article_text: str) -> float:
        """
        Calculate relevance score based on financial keywords
        """
        total_score = 0.0
        total_weight = 0.0
        
        for category, keywords in self.financial_keywords.items():
            if category == 'high_weight':
                weight = 3.0
            elif category == 'medium_weight':
                weight = 2.0
            else:  # low_weight
                weight = 1.0
            
            found_keywords = 0
            for keyword in keywords:
                if keyword.lower() in article_text:
                    found_keywords += 1
            
            if keywords:
                category_score = (found_keywords / len(keywords)) * weight
                total_score += category_score
                total_weight += weight
        
        return (total_score / total_weight) if total_weight > 0 else 0.0
    
    def _calculate_context_score(self, title: str, description: str, company_name: str, ticker: str) -> float:
        """
        Calculate contextual relevance (title gets higher weight)
        """
        title_lower = title.lower()
        description_lower = description.lower()
        
        score = 0.0
        
        # Company mentions in title (higher weight)
        if ticker and ticker.lower() in title_lower:
            score += 0.6
        elif company_name and company_name.lower() in title_lower:
            score += 0.4
        
        # Company mentions in description
        if ticker and ticker.lower() in description_lower:
            score += 0.3
        elif company_name and company_name.lower() in description_lower:
            score += 0.2
        
        return min(score, 1.0)
    
    def deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate articles from multiple sources
        Uses title similarity and URL matching
        """
        try:
            if not articles:
                return []
            
            logger.debug(f"Deduplicating {len(articles)} articles")
            
            # Track seen articles
            seen_hashes = set()
            seen_urls = set()
            seen_titles = {}
            unique_articles = []
            
            for article in articles:
                # Generate content hash
                content_hash = self._generate_article_hash(article)
                
                # Check URL duplicates
                url = self._clean_url(article.get('link', ''))
                
                # Check title similarity
                title = self._clean_text(article.get('title', ''))
                title_normalized = self._normalize_title(title)
                
                is_duplicate = False
                
                # Check against various duplicate criteria
                if content_hash in seen_hashes:
                    is_duplicate = True
                    logger.debug(f"Content hash duplicate: {title[:50]}...")
                
                elif url and url in seen_urls:
                    is_duplicate = True
                    logger.debug(f"URL duplicate: {title[:50]}...")
                
                elif title_normalized:
                    # Check for similar titles
                    for existing_title_norm, existing_article in seen_titles.items():
                        similarity = self._calculate_title_similarity(title_normalized, existing_title_norm)
                        if similarity > 0.85:  # 85% similarity threshold
                            is_duplicate = True
                            logger.debug(f"Title similarity duplicate: {title[:50]}...")
                            
                            # Keep the one with higher relevance score
                            if article.get('relevance_score', 0) > existing_article.get('relevance_score', 0):
                                # Replace the existing article
                                unique_articles.remove(existing_article)
                                seen_titles[title_normalized] = article
                                unique_articles.append(article)
                            break
                
                if not is_duplicate:
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
            return articles  # Return original articles if deduplication fails
    
    def _generate_article_hash(self, article: Dict[str, Any]) -> str:
        """
        Generate hash for article content comparison
        """
        try:
            # Use title and first part of description for hash
            title = self._clean_text(article.get('title', ''))
            description = self._clean_text(article.get('description', ''))[:200]  # First 200 chars
            
            content = f"{title}|{description}".lower()
            return hashlib.md5(content.encode()).hexdigest()
            
        except Exception:
            return ""
    
    def _clean_url(self, url: str) -> str:
        """
        Clean and normalize URL for comparison
        """
        if not url:
            return ""
        
        # Remove common tracking parameters
        url = re.sub(r'[?&](utm_|ref=|source=|campaign=)[^&]*', '', url)
        
        # Remove trailing slashes and fragments
        url = re.sub(r'[/#]+$', '', url)
        
        return url.lower().strip()
    
    def _normalize_title(self, title: str) -> str:
        """
        Normalize title for similarity comparison
        """
        if not title:
            return ""
        
        # Convert to lowercase and remove extra spaces
        normalized = re.sub(r'\s+', ' ', title.lower().strip())
        
        # Remove common punctuation
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Remove stop words
        words = normalized.split()
        significant_words = [w for w in words if len(w) > 2 and w not in self.stop_words]
        
        return ' '.join(significant_words)
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate similarity between two normalized titles
        """
        if not title1 or not title2:
            return 0.0
        
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def sort_by_recency(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort articles by relevance score and recency (newest first for FinBERT)
        """
        try:
            if not articles:
                return []
            
            logger.debug(f"Sorting {len(articles)} articles by relevance and recency")
            
            # Parse dates and add sort keys
            for article in articles:
                parsed_date = self._parse_article_date(article.get('published', ''))
                article['_parsed_date'] = parsed_date
                article['_relevance_score'] = article.get('relevance_score', 0.0)
            
            # Sort by relevance (desc) then by recency (desc)
            sorted_articles = sorted(
                articles,
                key=lambda x: (x['_relevance_score'], x['_parsed_date']),
                reverse=True
            )
            
            # Remove temporary sort keys
            for article in sorted_articles:
                if '_parsed_date' in article:
                    del article['_parsed_date']
                if '_relevance_score' in article:
                    del article['_relevance_score']
            
            return sorted_articles
            
        except Exception as e:
            logger.error(f"Error sorting articles: {str(e)}")
            return articles
    
    def _parse_article_date(self, date_str: str) -> datetime:
        """
        Parse article publication date string to datetime object
        """
        if not date_str:
            return datetime.min.replace(tzinfo=timezone.utc)
        
        try:
            # Try ISO format first
            if 'T' in date_str and 'Z' in date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            # Try ISO format without Z
            if 'T' in date_str:
                return datetime.fromisoformat(date_str)
            
            # Try common date formats
            common_formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%Y-%m-%dT%H:%M:%S%z'
            ]
            
            for fmt in common_formats:
                try:
                    return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            # Default to current time if parsing fails
            logger.debug(f"Could not parse date: {date_str}")
            return datetime.now(timezone.utc)
            
        except Exception as e:
            logger.debug(f"Date parsing error: {str(e)}")
            return datetime.now(timezone.utc)
    
    def _filter_by_date(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter articles by date (keep only recent articles)
        """
        try:
            if not articles:
                return []
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.max_days_old)
            
            filtered_articles = []
            for article in articles:
                article_date = self._parse_article_date(article.get('published', ''))
                
                if article_date >= cutoff_date:
                    filtered_articles.append(article)
                else:
                    logger.debug(f"Filtered old article: {article.get('title', '')[:50]}...")
            
            logger.info(f"Date filtering: {len(articles)} -> {len(filtered_articles)} articles (max {self.max_days_old} days old)")
            return filtered_articles
            
        except Exception as e:
            logger.error(f"Error filtering by date: {str(e)}")
            return articles
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content
        """
        if not text:
            return ""
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        return text
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """
        Get filter configuration and statistics
        """
        return {
            'configuration': {
                'relevance_threshold': self.relevance_threshold,
                'max_articles_per_company': self.max_articles_per_company,
                'max_days_old': self.max_days_old
            },
            'keyword_categories': {
                category: len(keywords) 
                for category, keywords in self.financial_keywords.items()
            },
            'total_keywords': sum(len(keywords) for keywords in self.financial_keywords.values()),
            'stop_words_count': len(self.stop_words)
        }

# Convenience function for quick usage
def create_content_filter(relevance_threshold=0.6, max_articles=50, max_days=7) -> ContentFilter:
    """Factory function to create ContentFilter instance"""
    return ContentFilter(relevance_threshold, max_articles, max_days)

# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize content filter
        content_filter = ContentFilter()
        
        print("=== Content Filter Test ===")
        
        # Test configuration
        print("\n1. Filter Configuration:")
        stats = content_filter.get_filter_stats()
        print(f"   Relevance Threshold: {stats['configuration']['relevance_threshold']}")
        print(f"   Max Articles: {stats['configuration']['max_articles_per_company']}")
        print(f"   Max Days Old: {stats['configuration']['max_days_old']}")
        print(f"   Total Keywords: {stats['total_keywords']}")
        
        # Sample articles for testing
        sample_articles = [
            {
                'title': 'TCS Q3 Results: Revenue grows 12% YoY, profit jumps 15%',
                'description': 'Tata Consultancy Services reported strong quarterly results with revenue growth of 12% year-on-year and profit increase of 15%. The company maintained its market leadership position.',
                'link': 'https://example.com/tcs-results-1',
                'published': '2025-01-15T10:30:00Z',
                'source': 'Economic Times'
            },
            {
                'title': 'Indian IT sector shows resilience amid global challenges',
                'description': 'The Indian IT sector, led by companies like TCS, Infosys, and Wipro, continues to show strong performance despite global economic uncertainties.',
                'link': 'https://example.com/it-sector-news',
                'published': '2025-01-14T14:20:00Z',
                'source': 'Business Standard'
            },
            {
                'title': 'Weather update: Heavy rainfall expected in Mumbai',
                'description': 'The meteorological department has issued a warning for heavy rainfall in Mumbai and surrounding areas for the next 48 hours.',
                'link': 'https://example.com/weather-news',
                'published': '2025-01-15T08:00:00Z',
                'source': 'Moneycontrol'
            },
            {
                'title': 'TCS announces major expansion plans, to hire 50000 employees',
                'description': 'Tata Consultancy Services has announced significant expansion plans with a focus on emerging technologies. The company plans to hire 50,000 new employees over the next year.',
                'link': 'https://example.com/tcs-expansion',
                'published': '2025-01-13T16:45:00Z',
                'source': 'Economic Times'
            }
        ]
        
        # Sample company data
        company_data = {
            'symbol': 'TCS',
            'nse_symbol': 'TCS',
            'company_name': 'Tata Consultancy Services Limited',
            'search_terms': ['TCS', 'Tata Consultancy Services', 'Tata Consultancy']
        }
        
        print(f"\n2. Testing with {len(sample_articles)} sample articles for {company_data['company_name']}")
        
        # Test individual article relevance
        print("\n3. Individual Article Relevance Scores:")
        for i, article in enumerate(sample_articles, 1):
            score = content_filter.check_article_relevance(
                article, 
                company_data['company_name'], 
                company_data['symbol']
            )
            print(f"   {i}. Score: {score:.3f} - {article['title'][:60]}...")
        
        # Test complete filtering pipeline
        print("\n4. Complete Filtering Pipeline:")
        filtered_articles = content_filter.filter_company_articles(sample_articles, company_data)
        
        print(f"   Input Articles: {len(sample_articles)}")
        print(f"   Filtered Articles: {len(filtered_articles)}")
        
        if filtered_articles:
            print("\n   Filtered Results:")
            for i, article in enumerate(filtered_articles, 1):
                print(f"   {i}. {article['title']}")
                print(f"      Relevance: {article.get('relevance_score', 0):.3f}")
                print(f"      Source: {article.get('source', 'Unknown')}")
                print(f"      Identifiers: {', '.join(article.get('company_identifiers_found', []))}")
        
        # Test deduplication
        print("\n5. Deduplication Test:")
        # Add duplicate article
        duplicate_article = sample_articles[0].copy()
        duplicate_article['link'] = 'https://example.com/tcs-results-duplicate'
        duplicate_article['source'] = 'Different Source'
        
        test_articles_with_duplicate = sample_articles + [duplicate_article]
        deduplicated = content_filter.deduplicate_articles(test_articles_with_duplicate)
        
        print(f"   Before deduplication: {len(test_articles_with_duplicate)} articles")
        print(f"   After deduplication: {len(deduplicated)} articles")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()