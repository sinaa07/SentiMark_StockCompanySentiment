import os
import json
import logging
import requests
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMWebSearcher:
    """
    LLM-powered web search for financial news articles.
    MVP implementation using Gemini API.
    """
    
    def __init__(self, model: str = "gemini", max_articles: int = 10, date_range_days: int = 3):
        """
        Initialize the LLM web searcher.
        
        Args:
            model: LLM provider to use (currently only "gemini" supported)
            max_articles: Maximum number of articles to return
            date_range_days: How many days back to search
        """
        self.model = model
        self.max_articles = max_articles
        self.date_range_days = date_range_days
        
        # Get API key from environment
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables")
        
        # Credible financial sources whitelist
        self.credible_sources = [
            'economictimes.indiatimes.com',
            'business-standard.com',
            'livemint.com',
            'moneycontrol.com',
            'thehindubusinessline.com',
            'reuters.com',
            'bloomberg.com',
            'financialexpress.com',
            'zeenews.india.com',
            'cnbctv18.com',
            'ndtvprofit.com',
            'news18.com'
        ]
        
        logger.info(f"LLMWebSearcher initialized: model={model}, max_articles={max_articles}, days={date_range_days}")
    
    def build_prompt(self, company_name: str, ticker: Optional[str] = None) -> str:
        """
        Build a structured prompt for the LLM to return financial news in JSON format.
        
        Args:
            company_name: Company name (e.g., "Reliance Industries")
            ticker: Optional NSE ticker symbol (e.g., "RELIANCE")
        
        Returns:
            Formatted prompt string
        """
        ticker_text = f" (ticker {ticker})" if ticker else ""
        sources_list = ", ".join(self.credible_sources)
        
        prompt = f"""You are a financial news search assistant for Indian stock markets. Search the web for the most recent financial news articles from the last {self.date_range_days} days about the company {company_name}{ticker_text}.

        Return ONLY a JSON array of objects. Each object must have these exact fields:
        - "title" (string): Article headline
        - "url" (string): Full article URL
        - "date" (string): Publication date in YYYY-MM-DD format
        - "summary" (string): A detailed 10-sentence factual summary of the article's financial and business content
        
        SUMMARY REQUIREMENTS:
        - Must be around 6 to 7 sentences (full, complete sentences with proper grammar)
        - Focus on factual information: financial figures, percentages, dates, announcements, and concrete events
        - Include specific numbers whenever available (revenue, profit, EPS, stock price, market cap, etc.)
        - Mention NSE/BSE stock performance, trading volumes, and price movements when available
        - Include information about quarterly results, annual reports, board decisions, dividend announcements
        - Reference key stakeholders: promoters, institutional investors, executives, or analysts quoted
        - Mention regulatory filings (SEBI disclosures, BSE/NSE announcements) if applicable
        - Avoid vague statements like "the company is performing well" - be specific with data
        - Do not invent or assume details not present in the article
        - Each sentence should add unique, valuable information for investors
        
        IMPORTANT CONSTRAINTS:
        1. Only include articles published within the **last 7 days** from the current date in current month (October,2025). 
           - Strictly check the published date in the article.
           - If the date is missing or ambiguous, include the article.
        2. Only include articles from these credible financial sources: {sources_list}
        3. Return maximum {self.max_articles} articles
        4. Only recent articles (last {self.date_range_days} days)
        5. Focus on: quarterly earnings, stock price movements, corporate actions, mergers & acquisitions, management changes, regulatory updates, sectoral trends
        6. Do not include any text before or after the JSON array
        7. If no articles found, return empty array: []
        
        The JSON must be well-formed and strictly follow this structure.
        
        Example output format:
        [
          {{
            "title": "Reliance Industries Reports 12% Jump in Q2 Profit, Declares ₹9 Per Share Dividend",
            "url": "https://economictimes.indiatimes.com/markets/stocks/news/reliance-industries-q2-results-profit-dividend/articleshow/123456789.cms",
            "date": "2025-10-01",
            "summary": "Reliance Industries Ltd (NSE: RELIANCE) reported a consolidated net profit of ₹19,323 crore for Q2 FY2026, marking a 12% increase compared to ₹17,265 crore in the same quarter last year. The company's revenue from operations stood at ₹2,35,478 crore, up 8.5% year-on-year, driven by strong performance in the retail and digital services segments. The board of directors declared an interim dividend of ₹9 per equity share, with a record date set for October 15, 2025. Reliance Jio added 15.2 million new subscribers during the quarter, taking its total subscriber base to 485 million, while average revenue per user (ARPU) increased to ₹195 from ₹181 in the previous quarter. The oil-to-chemicals (O2C) business reported an EBITDA of ₹13,845 crore, impacted by softer refining margins that averaged $8.2 per barrel. Reliance Retail's revenue grew 17% to ₹68,345 crore with the addition of 847 new stores, bringing the total store count to 18,040 across India. The company's net debt reduced to ₹1,42,890 crore from ₹1,56,234 crore in the previous quarter, reflecting improved cash flow generation. Chairman Mukesh Ambani announced plans to invest ₹75,000 crore in clean energy projects over the next three years. On NSE, Reliance shares closed at ₹2,847.50, up 4.2% on the day of the announcement, with trading volumes surging to 12.5 million shares. Analysts at ICICI Securities maintained a 'Buy' rating with a revised target price of ₹3,200, citing strong growth momentum across all business verticals."
          }}
        ]"""
                
        return prompt
    
    def call_model(self, prompt: str) -> str:
        """
        Make API call to Gemini and return raw response.
        
        Args:
            prompt: The structured prompt to send
        
        Returns:
            Raw model response as string
        
        Raises:
            Exception: If API call fails
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        
        try:
            # Use Gemini 2.5 Flash - the fastest and latest model available
            GEMINI_MODEL = "gemini-2.5-flash"
            # Correct Gemini API endpoint using v1
            url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent?key={self.api_key}"
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "topK": 64,
                    "topP": 0.95,
                    "maxOutputTokens": 8000,  # Increased to allow full response
                }
            }
            
            logger.debug(f"Calling Gemini API at: {url.replace(self.api_key, '***API_KEY_HIDDEN***')}")
            response = requests.post(url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            
            result = response.json()
            logger.debug("📡 Full Gemini raw response:\n%s", json.dumps(result, indent=2))
            
            # Extract text from Gemini response structure
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                
                # Check for truncated response
                finish_reason = candidate.get('finishReason', '')
                if finish_reason == 'MAX_TOKENS':
                    logger.warning("⚠️ Response was truncated due to MAX_TOKENS. Increase maxOutputTokens.")
                
                logger.debug(f"Candidate keys: {candidate.keys()}")
                
                content = candidate.get('content', {})
                logger.debug(f"Content type: {type(content)}, Content keys: {content.keys() if isinstance(content, dict) else 'not a dict'}")
                
                parts = content.get('parts', [])
                logger.debug(f"Parts: {parts}")
                
                if parts and len(parts) > 0:
                    logger.debug(f"First part keys: {parts[0].keys() if isinstance(parts[0], dict) else 'not a dict'}")
                    if 'text' in parts[0]:
                        text = parts[0]['text']
                        logger.debug(f"✅ Received response text ({len(text)} chars)")
                        return text
                    else:
                        logger.warning(f"No 'text' key in first part. Part content: {parts[0]}")
                else:
                    logger.warning(f"No parts in content. Finish reason: {finish_reason}")
            
            logger.warning("⚠️ Unexpected Gemini response structure")
            logger.warning(f"Response keys: {result.keys()}")
            if 'candidates' in result and result['candidates']:
                logger.warning(f"First candidate: {json.dumps(result['candidates'][0], indent=2)}")
            return ""
            
        except requests.exceptions.Timeout:
            logger.error("Gemini API call timed out")
            raise
        except requests.exceptions.RequestException as e:
            # FIXED: Mask the API key in error logs
            error_msg = str(e)
            if self.api_key and self.api_key in error_msg:
                error_msg = error_msg.replace(self.api_key, "***API_KEY_HIDDEN***")
            logger.error(f"Gemini API request failed: {error_msg}")
            raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Error calling Gemini: {str(e)}")
            raise
    
    def parse_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse and normalize JSON response from LLM into standardized article format.
        
        Args:
            response: Raw response string from LLM
        
        Returns:
            List of normalized article dictionaries
        """
        if not response or not response.strip():
            logger.warning("Empty response from LLM")
            return []
        
        try:
            # Clean response - remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip().lstrip('json').strip()
            
            logger.debug(f"Cleaned response to parse: {cleaned[:500]}...")
            
            # Parse JSON
            raw_articles = json.loads(cleaned)
            
            if not isinstance(raw_articles, list):
                logger.warning("Response is not a JSON array")
                return []
            
            logger.info(f"Parsed {len(raw_articles)} raw articles from LLM")
            
            # Normalize each article
            normalized_articles = []
            for raw in raw_articles:
                try:
                    normalized = self._normalize_article(raw)
                    if normalized:
                        normalized_articles.append(normalized)
                except Exception as e:
                    logger.debug(f"Failed to normalize article: {str(e)}")
                    continue
            
            logger.info(f"Successfully normalized {len(normalized_articles)} articles")
            return normalized_articles
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Raw response (first 1000 chars):\n{response[:1000]}")
            logger.error(f"Cleaned response (first 1000 chars):\n{cleaned[:1000] if 'cleaned' in locals() else 'N/A'}")
            return []
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return []
    
    def _normalize_article(self, raw_article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalize a single article to match pipeline standard format.
        
        Args:
            raw_article: Raw article dict from LLM
        
        Returns:
            Normalized article dict or None if invalid
        """
        # Extract required fields
        title = raw_article.get('title', '').strip()
        url = raw_article.get('url', '').strip()
        summary = raw_article.get('summary', '').strip()
        date_str = raw_article.get('date', '').strip()
        
        # Validate required fields
        if not title or not url:
            logger.debug("Article missing title or url")
            return None
        
        # Validate URL
        if not self._is_valid_url(url):
            logger.debug(f"Invalid URL: {url}")
            return None
        
        # Check if URL is from credible source
        if not self._is_credible_source(url):
            logger.debug(f"URL not from credible source: {url}")
            return None
        
        # Parse date
        published = self._parse_date(date_str)
        
        # Create normalized article matching pipeline standard format
        normalized = {
            "title": title,
            "url": url,
            "published": published,
            "summary": summary,
            "source": "gemini",
            "relevance_score": 1.0,
            "weight": 1.0
        }
        
        return normalized
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _is_credible_source(self, url: str) -> bool:
        """Check if URL is from a credible source."""
        try:
            domain = urlparse(url).netloc.lower()
            # Remove www. prefix
            domain = domain.replace('www.', '')
            
            # Exact match or proper subdomain match
            return any(domain == src or domain.endswith(f".{src}") for src in self.credible_sources)
        except Exception:
            return False
    
    def _parse_date(self, date_str: str) -> datetime:
        """
        Parse date string to datetime object.
        
        Args:
            date_str: Date string (expected format: YYYY-MM-DD)
        
        Returns:
            datetime object (defaults to now if parsing fails)
        """
        if not date_str:
            return datetime.now(timezone.utc)
        
        try:
            # Try YYYY-MM-DD format
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            try:
                # Try ISO format
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except Exception:
                logger.debug(f"Could not parse date: {date_str}")
                return datetime.now(timezone.utc)
    
    def search_news(self, company_name: str, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Main entry point: Search for financial news about a company.
        
        Args:
            company_name: Company name (e.g., "Reliance Industries")
            ticker: Optional NSE ticker symbol (e.g., "RELIANCE")
        
        Returns:
            List of normalized article dictionaries (empty list on failure)
        """
        logger.info(f"Searching news for: {company_name} ({ticker or 'no ticker'})")
        
        try:
            # Build prompt
            prompt = self.build_prompt(company_name, ticker)
            
            # Call model
            response = self.call_model(prompt)
            
            # Parse and normalize
            articles = self.parse_response(response)
            
            logger.info(f"Search completed: {len(articles)} articles found")
            return articles
            
        except Exception as e:
            logger.error(f"Error searching news: {str(e)}")
            return []


# Factory function
def create_llm_searcher(model: str = "gemini", max_articles: int = 20, date_range_days: int = 3) -> LLMWebSearcher:
    """Factory function to create LLMWebSearcher instance."""
    return LLMWebSearcher(model, max_articles, date_range_days)


# Testing
if __name__ == "__main__":
    try:
        # Initialize searcher
        searcher = LLMWebSearcher()
        
        print("=== LLM Web Searcher Test ===\n")
        
        # Test search
        print("Searching for: Reliance Industries (RELIANCE)")
        articles = searcher.search_news("Reliance Industries", "RELIANCE")
        
        if articles:
            print(f"\nFound {len(articles)} articles:\n")
            for i, article in enumerate(articles[:3], 1):
                print(f"{i}. {article['title'][:70]}...")
                print(f"   Source: {article['source']}")
                print(f"   URL: {article['url']}")
                print(f"   Published: {article['published']}")
                print(f"   Summary: {article['summary'][:100]}...\n")
        else:
            print("No articles found or error occurred")
            
    except Exception as e:
        print(f"Error during testing: {e}")