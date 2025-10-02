"""
LLM Web News Searcher
Purpose: Find recent financial news URLs using Gemini's web search capabilities
Single responsibility: Takes company info, returns recent news URLs with metadata
"""

import google.generativeai as genai
import re
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse
from typing import List, Dict, Optional

class LLMWebSearcher:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        """Initialize the LLM web searcher with Gemini"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.logger = logging.getLogger(__name__)
        
    def search_recent_news(self, company_name: str, ticker: str = None, days_back: int = 2) -> List[Dict]:
        """
        Main function: Search for recent financial news about a company
        
        Args:
            company_name: Full company name (e.g., "Adani Ports")
            ticker: Stock ticker symbol (e.g., "ADANIPORTS") - optional
            days_back: How many days back to search (default: 2)
            
        Returns:
            List of dictionaries with standardized schema: title, description, link, published, source, author, relevance_score
        """
        try:
            # Build search prompt
            prompt = self._build_search_prompt(company_name, ticker, days_back)
            
            # Call Gemini API
            response = self._call_gemini_for_search(prompt)
            
            # Parse the structured response
            articles = self._parse_structured_response(response)
            
            # Validate and clean URLs
            validated_articles = self._validate_and_clean(articles)
            
            self.logger.info(f"Found {len(validated_articles)} recent articles for {company_name}")
            return validated_articles
            
        except Exception as e:
            self.logger.error(f"Error searching news for {company_name}: {str(e)}")
            return []  # Return empty list instead of crashing
    
    def _build_search_prompt(self, company_name: str, ticker: str, days_back: int) -> str:
        """Build the search prompt for Gemini"""
        ticker_text = f" ({ticker})" if ticker else ""
        
        prompt = f"""
        Search the web for recent financial news about {company_name}{ticker_text} from the last {days_back} days.

        Focus only on:
        - Financial news and business updates
        - Stock market news
        - Corporate announcements
        - Regulatory news
        - Earnings and business performance

        Ignore:
        - General news unrelated to business/finance
        - Old news (older than {days_back} days)
        - Duplicate stories

        Return EXACTLY in this format for each article found:

        **Headline: [Article Title]**
        * **URL:** [Full URL]
        * **Content:** [Brief 2-3 sentence summary of the key financial points]

        Find 5-8 most relevant recent articles. Make sure URLs are complete and working.
        """
        
        return prompt
    
    def _call_gemini_for_search(self, prompt: str) -> str:
        """Call Gemini API with web search capabilities"""
        try:
            # Use Gemini's web search capability
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {str(e)}")
            raise
    
    def _parse_structured_response(self, response_text: str) -> List[Dict]:
        """Parse the structured response from Gemini"""
        articles = []
        
        # Split by headline markers
        sections = re.split(r'\*\*Headline:\s*', response_text)
        
        for section in sections[1:]:  # Skip first empty section
            try:
                article = self._extract_article_data(section)
                if article:
                    articles.append(article)
            except Exception as e:
                self.logger.warning(f"Failed to parse article section: {str(e)}")
                continue
                
        return articles
    
    def _extract_article_data(self, section: str) -> Optional[Dict]:
        """Extract article data from a section of text"""
        try:
            # Extract title (everything before first **)
            title_match = re.search(r'^([^*]+)', section)
            title = title_match.group(1).strip() if title_match else ""
            
            # Extract URL
            url_match = re.search(r'\*\*URL:\*\*\s*\[?([^\]\s]+)\]?', section)
            url = url_match.group(1).strip() if url_match else ""
            
            # Extract content/description
            content_match = re.search(r'\*\*Content:\*\*\s*(.+?)(?:\n\n|\*\*|$)', section, re.DOTALL)
            description = content_match.group(1).strip() if content_match else ""
            
            if not url or not title:
                return None
                
            # Extract domain for source
            parsed_url = urlparse(url)
            source = parsed_url.netloc.replace('www.', '').replace('m.', '')
            
            return {
                'title': title,
                'description': description,
                'link': url,
                'published': datetime.now().strftime('%Y-%m-%d'),
                'source': source,
                'author': '',  # LLM search doesn't provide author info
                'relevance_score': 0.0  # Will be set by downstream filtering
            }
            
        except Exception as e:
            self.logger.warning(f"Error extracting article data: {str(e)}")
            return None
    
    def _validate_and_clean(self, articles: List[Dict]) -> List[Dict]:
        """Validate URLs and remove duplicates"""
        valid_articles = []
        seen_urls = set()
        
        for article in articles:
            # Skip if URL already seen
            if article['link'] in seen_urls:
                continue
                
            # Basic URL validation
            if not self._is_valid_news_url(article['link']):
                continue
                
            seen_urls.add(article['link'])
            valid_articles.append(article)
        
        return valid_articles
    
    def _is_valid_news_url(self, url: str) -> bool:
        """Basic validation for news URLs"""
        try:
            parsed = urlparse(url)
            
            # Must have valid scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False
                
            # Common financial news domains (expand as needed)
            financial_domains = [
                'economictimes.com',
                'financialexpress.com',
                'moneycontrol.com',
                'business-standard.com',
                'livemint.com',
                'reuters.com',
                'bloomberg.com',
                'cnbc.com',
                'marketwatch.com',
                'timesofindia.com'
            ]
            
            # Check if it's a known financial news domain
            domain = parsed.netloc.replace('www.', '').replace('m.', '')
            return any(domain.endswith(fin_domain) for fin_domain in financial_domains)
            
        except Exception:
            return False


def create_searcher(api_key: str = None, model_name: str = "gemini-1.5-pro") -> LLMWebSearcher:
    """Factory function to create a web searcher instance"""
    if not api_key:
        # Try to get from environment or config
        import os
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key required")
    
    return LLMWebSearcher(api_key, model_name)


# Simple usage example
if __name__ == "__main__":
    # Example usage
    searcher = create_searcher()
    results = searcher.search_recent_news("Adani Ports", "ADANIPORTS")
    
    for article in results:
        print(f"Title: {article['title']}")
        print(f"Link: {article['link']}")
        print(f"Source: {article['source']}")
        print("-" * 50)