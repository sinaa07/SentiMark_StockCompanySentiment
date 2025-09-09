import yfinance as yf
import json
import logging
import re
import time
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import os

class SearchStatus(Enum):
    """Status codes for search results"""
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    INVALID_INPUT = "invalid_input"
    MULTIPLE_MATCHES = "multiple_matches"
    API_ERROR = "api_error"
    ERROR = "error"

@dataclass
class ProcessedCompanyInput:
    """Processed company information ready for news scraping"""
    ticker: str
    company_name: str
    search_terms: List[str]
    sector: str
    status: SearchStatus
    confidence_score: float = 0.0
    suggestions: List[Dict] = None
    raw_input: str = ""
    error_message: str = ""
    market_cap: str = ""
    exchange: str = "NSE"

class SimpleCache:
    """Simple in-memory cache for yfinance results"""
    
    def __init__(self, ttl_minutes: int = 60):
        self.cache = {}
        self.ttl_minutes = ttl_minutes
        
    def get(self, key: str) -> Optional[Dict]:
        """Get cached data if still valid"""
        if key not in self.cache:
            return None
            
        entry = self.cache[key]
        if time.time() - entry['timestamp'] > (self.ttl_minutes * 60):
            del self.cache[key]
            return None
            
        return entry['data']
    
    def set(self, key: str, data: Dict):
        """Cache data with timestamp"""
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }

class RateLimitedCache(SimpleCache):
    """Enhanced cache with size limits and better cleanup"""
    
    def __init__(self, ttl_minutes: int = 60, max_size: int = 1000):
        super().__init__(ttl_minutes)
        self.max_size = max_size
        
    def set(self, key: str, data: Dict):
        """Cache data with size management"""
        # Clean expired entries first
        self._cleanup_expired()
        
        # If still over limit, remove oldest entries
        if len(self.cache) >= self.max_size:
            # Remove 20% of oldest entries
            remove_count = max(1, self.max_size // 5)
            oldest_keys = sorted(self.cache.keys(), 
                               key=lambda k: self.cache[k]['timestamp'])[:remove_count]
            for old_key in oldest_keys:
                del self.cache[old_key]
        
        # Add new entry
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry['timestamp'] > (self.ttl_minutes * 60)
        ]
        for key in expired_keys:
            del self.cache[key]

class EnhancedRateLimiter:
    """Advanced rate limiter with exponential backoff and jitter"""
    
    def __init__(self):
        self.last_request_time = 0
        self.base_interval = 0.8  # Start with 800ms (more conservative)
        self.current_interval = self.base_interval
        self.max_interval = 10.0  # Max 10 seconds
        self.consecutive_errors = 0
        self.max_retries = 3
        
    def wait_if_needed(self):
        """Wait if needed based on current rate limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.current_interval:
            sleep_time = self.current_interval - time_since_last
            # Add jitter (±20% randomness)
            jitter = random.uniform(-0.2, 0.2) * sleep_time
            actual_sleep = max(0.1, sleep_time + jitter)
            time.sleep(actual_sleep)
        
        self.last_request_time = time.time()
    
    def on_success(self):
        """Call this after successful API request"""
        self.consecutive_errors = 0
        # Gradually reduce interval back to base
        if self.current_interval > self.base_interval:
            self.current_interval = max(
                self.base_interval, 
                self.current_interval * 0.8
            )
    
    def on_error(self, is_rate_limit_error: bool = False):
        """Call this after failed API request"""
        self.consecutive_errors += 1
        
        if is_rate_limit_error or self.consecutive_errors >= 2:
            # Exponential backoff
            self.current_interval = min(
                self.max_interval,
                self.current_interval * (2 ** min(self.consecutive_errors, 4))
            )
    
    def should_retry(self) -> bool:
        """Check if we should retry the request"""
        return self.consecutive_errors < self.max_retries

class YFinanceCompanyResolver:
    """
    Resolves company names to valid tickers using yfinance (100% Free)
    No paid APIs, no external services - only yfinance library
    """
    
    def __init__(self):
        self.cache = RateLimitedCache(ttl_minutes=60, max_size=1000)
        self.rate_limiter = EnhancedRateLimiter()
        self.setup_logging()
        
        # Reduced popular tickers list - only most liquid ones
        self.popular_tickers = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", 
            "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS"
        ]
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _clean_input(self, user_input: str) -> str:
        """Clean and normalize user input"""
        if not user_input:
            return ""
            
        # Remove extra whitespace and clean
        cleaned = re.sub(r'\s+', ' ', user_input.strip())
        
        # Remove common suffixes that might interfere
        suffixes_to_remove = [
            'limited', 'ltd', 'ltd.', 'pvt', 'pvt.', 'private', 'company', 
            'corp', 'corporation', 'inc', 'incorporated', '.ns', '.bo',
            'enterprises', 'industries', 'services'
        ]
        
        cleaned_lower = cleaned.lower()
        for suffix in suffixes_to_remove:
            if cleaned_lower.endswith(f' {suffix}'):
                cleaned = cleaned[:-len(suffix)-1].strip()
                break
        
        return cleaned

    def _generate_ticker_variations_optimized(self, company_name: str) -> List[str]:
        """
        Optimized ticker generation - prioritized and limited variations
        Returns max 8 variations in order of likelihood
        """
        variations = []  # Use list to maintain priority order
        seen = set()  # Track duplicates
        
        clean_name = self._clean_input(company_name)
        words = clean_name.split()
        
        if not words:
            return []
        
        def add_variation(ticker):
            """Helper to add unique variations"""
            if ticker not in seen:
                seen.add(ticker)
                variations.append(ticker)
        
        # Priority 1: Direct ticker input (highest probability)
        ticker_input = clean_name.upper().replace(' ', '')
        add_variation(f"{ticker_input}.NS")
        add_variation(f"{ticker_input}.BO")
        
        # Priority 2: Special patterns for common Indian companies
        special_patterns = {
            'STATE BANK': 'SBIN.NS',
            'HDFC BANK': 'HDFCBANK.NS', 
            'ICICI BANK': 'ICICIBANK.NS',
            'TATA CONSULTANCY': 'TCS.NS',
            'BHARTI AIRTEL': 'BHARTIARTL.NS',
            'RELIANCE': 'RELIANCE.NS',
            'INFOSYS': 'INFY.NS',
            'MARUTI SUZUKI': 'MARUTI.NS',
            'LARSEN TOUBRO': 'LT.NS',
            'HINDUSTAN UNILEVER': 'HINDUNILVR.NS'
        }
        
        name_upper = clean_name.upper()
        for pattern, ticker in special_patterns.items():
            if pattern in name_upper:
                add_variation(ticker)
                # Also add BSE version
                add_variation(ticker.replace('.NS', '.BO'))
                break  # Only match first pattern
        
        # Priority 3: Abbreviations (only if we have multiple words)
        if len(words) >= 2:
            # Clean abbreviation (skip common suffixes)
            clean_words = [w for w in words 
                          if w.lower() not in ['limited', 'ltd', 'pvt', 'private', 'company']]
            if clean_words:
                abbrev = ''.join([w[0].upper() for w in clean_words])
                add_variation(f"{abbrev}.NS")
                add_variation(f"{abbrev}.BO")
        
        # Priority 4: First word (for single distinctive names)
        first_word = words[0].upper()
        if len(first_word) > 2:  # Skip very short words
            add_variation(f"{first_word}.NS")
            add_variation(f"{first_word}.BO")
        
        # Return max 8 variations to limit API calls
        return variations[:8]

    def _validate_ticker_with_yfinance_enhanced(self, ticker: str) -> Optional[Dict]:
        """
        Enhanced ticker validation with better error handling and caching
        """
        # Check cache first (including negative results)
        cache_key = f"ticker_{ticker}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result if cached_result != "NOT_FOUND" else None
        
        retry_count = 0
        
        while retry_count <= self.rate_limiter.max_retries:
            try:
                # Rate limiting with backoff
                self.rate_limiter.wait_if_needed()
                
                # Create yfinance ticker object
                stock = yf.Ticker(ticker)
                
                # Get info with timeout
                info = stock.info
                
                # Validate response quality
                if not self._is_valid_ticker_response(info):
                    # Cache negative result for short time to avoid repeated calls
                    self.cache.set(cache_key, "NOT_FOUND")
                    self.rate_limiter.on_success()  # Not an error, just no data
                    return None
                
                # Extract company information
                company_data = self._extract_company_data(ticker, info)
                
                # Cache positive result
                self.cache.set(cache_key, company_data)
                self.rate_limiter.on_success()
                
                return company_data
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e).lower()
                
                # Check if it's a rate limiting error
                is_rate_limit = any(phrase in error_msg for phrase in 
                                  ['429', 'rate limit', 'too many requests', 'quota'])
                
                self.rate_limiter.on_error(is_rate_limit)
                
                self.logger.warning(f"Error validating ticker {ticker} (attempt {retry_count}): {e}")
                
                if not self.rate_limiter.should_retry():
                    break
                    
                # Wait longer for rate limit errors
                if is_rate_limit:
                    time.sleep(2 ** retry_count)
        
        # Cache negative result after all retries failed
        self.cache.set(cache_key, "NOT_FOUND")
        return None

    def _is_valid_ticker_response(self, info: dict) -> bool:
        """
        Validate if yfinance response contains meaningful data
        """
        if not info:
            return False
        
        # Check for common error indicators
        error_indicators = [
            info.get('symbol') is None,
            info.get('regularMarketPrice') is None and info.get('previousClose') is None,
            info.get('longName') is None and info.get('shortName') is None,
            'messageBoard' not in info and 'website' not in info  # Usually present for real stocks
        ]
        
        # If too many indicators suggest invalid data
        if sum(error_indicators) >= 3:
            return False
        
        # Additional check for recently active trading
        market_cap = info.get('marketCap', 0)
        volume = info.get('volume', 0) or info.get('averageVolume', 0)
        
        # Very basic sanity checks
        if market_cap and market_cap < 1000000:  # Less than 10L market cap
            return False
            
        return True

    def _extract_company_data(self, ticker: str, info: dict) -> Dict:
        """
        Extract and clean company data from yfinance response
        """
        # Get best available name
        company_name = (info.get('longName') or 
                       info.get('shortName') or 
                       info.get('symbol', ticker.split('.')[0]))
        
        # Clean up company name
        if company_name:
            company_name = company_name.strip()
        
        return {
            'ticker': ticker,
            'symbol': info.get('symbol', ticker.replace('.NS', '').replace('.BO', '')),
            'company_name': company_name,
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'exchange': 'NSE' if '.NS' in ticker else 'BSE',
            'market_cap': self._format_market_cap(info.get('marketCap', 0)),
            'currency': info.get('currency', 'INR'),
            'country': info.get('country', 'India'),
            'search_terms': self._generate_search_terms({'company_name': company_name, 
                                                       'symbol': info.get('symbol', '')})
        }

    def _format_market_cap(self, market_cap: int) -> str:
        """Format market cap in Indian format"""
        if not market_cap or market_cap <= 0:
            return "Unknown"
            
        if market_cap >= 1e12:  # 1 trillion
            return f"₹{market_cap/1e12:.1f}T"
        elif market_cap >= 1e10:  # 1000 crores
            return f"₹{market_cap/1e10:.0f}K Cr"
        elif market_cap >= 1e9:   # 100 crores
            return f"₹{market_cap/1e9:.0f} Cr"
        elif market_cap >= 1e7:   # 1 crore
            return f"₹{market_cap/1e7:.1f} Cr"
        else:
            return f"₹{market_cap/1e5:.1f}L"
    
    def _generate_search_terms(self, company_data: Dict) -> List[str]:
        """Generate search terms for news scraping"""
        terms = set()
        
        # Add company name variations
        company_name = company_data.get('company_name', '')
        if company_name:
            terms.add(company_name)
            
            # Remove common suffixes for cleaner search terms
            clean_name = self._clean_input(company_name)
            if clean_name != company_name:
                terms.add(clean_name)
        
        # Add symbol/ticker
        symbol = company_data.get('symbol', '')
        if symbol:
            terms.add(symbol)
        
        # Add short variations
        if company_name:
            words = company_name.split()
            if len(words) >= 2:
                # First two words
                terms.add(' '.join(words[:2]))
                # First word if meaningful
                if len(words[0]) > 3 and words[0].lower() not in ['the', 'and', 'of']:
                    terms.add(words[0])
        
        return list(terms)[:5]  # Limit to 5 search terms
    
    def _calculate_confidence(self, user_input: str, company_data: Dict) -> float:
        """Calculate confidence score for the match"""
        user_lower = user_input.lower()
        company_name_lower = company_data.get('company_name', '').lower()
        symbol_lower = company_data.get('symbol', '').lower()
        
        confidence = 0.0
        
        # Exact symbol match
        if user_lower == symbol_lower:
            confidence = 1.0
        
        # Exact company name match
        elif user_lower == company_name_lower:
            confidence = 0.95
        
        # Symbol contains user input or vice versa
        elif user_lower in symbol_lower or symbol_lower in user_lower:
            confidence = 0.9
        
        # Company name contains user input
        elif user_lower in company_name_lower:
            confidence = 0.8
        
        # User input contains company name
        elif company_name_lower in user_lower:
            confidence = 0.7
        
        # Partial word matching
        else:
            user_words = user_lower.split()
            company_words = company_name_lower.split()
            
            matches = sum(1 for word in user_words 
                         if any(word in comp_word for comp_word in company_words))
            
            if matches > 0:
                confidence = 0.5 + (matches / len(user_words)) * 0.3
        
        return confidence

    def resolve_company_input(self, user_input: str) -> ProcessedCompanyInput:
        """
        Main method: Resolve user input to company information using yfinance
        Optimized resolution with early termination and smarter searching
        """
        # Input validation
        if not user_input or not user_input.strip():
            return ProcessedCompanyInput(
                ticker="",
                company_name="",
                search_terms=[],
                sector="",
                status=SearchStatus.INVALID_INPUT,
                raw_input=user_input,
                error_message="Please enter a company name or ticker symbol"
            )
        
        clean_input = self._clean_input(user_input)
        
        if len(clean_input) < 2:
            return ProcessedCompanyInput(
                ticker="", company_name="", search_terms=[], sector="",
                status=SearchStatus.INVALID_INPUT, raw_input=user_input,
                error_message="Please enter at least 2 characters"
            )
        
        try:
            # Generate prioritized ticker variations (max 8)
            ticker_variations = self._generate_ticker_variations_optimized(clean_input)
            
            self.logger.info(f"Trying {len(ticker_variations)} prioritized variations for '{clean_input}'")
            
            valid_companies = []
            
            # Try variations with early termination
            for i, ticker in enumerate(ticker_variations):
                company_data = self._validate_ticker_with_yfinance_enhanced(ticker)
                
                if company_data:
                    confidence = self._calculate_confidence(clean_input, company_data)
                    company_data['confidence'] = confidence
                    valid_companies.append(company_data)
                    
                    # Early termination for high confidence matches
                    if confidence > 0.9:
                        self.logger.info(f"High confidence match found after {i+1} attempts")
                        break
                    
                    # If we have a decent match and tried enough variations, stop
                    if confidence > 0.7 and i >= 3:
                        self.logger.info(f"Good match found after {i+1} attempts")
                        break
            
            if not valid_companies:
                suggestions = self.get_search_suggestions(clean_input[:3], limit=5)
                return ProcessedCompanyInput(
                    ticker="", company_name=clean_input, search_terms=[clean_input],
                    sector="Unknown", status=SearchStatus.NOT_FOUND, raw_input=user_input,
                    error_message=f"No company found matching '{user_input}'. Please try a different search term.",
                    suggestions=suggestions
                )
            
            # Sort by confidence
            valid_companies.sort(key=lambda x: x['confidence'], reverse=True)
            best_match = valid_companies[0]
            
            # Check if we have multiple good matches
            if len(valid_companies) > 1 and valid_companies[1]['confidence'] > 0.6:
                # Multiple potential matches
                suggestions = []
                for company in valid_companies[:5]:
                    suggestions.append({
                        'ticker': company['symbol'],
                        'company_name': company['company_name'],
                        'sector': company['sector'],
                        'confidence': company['confidence'],
                        'display_name': f"{company['company_name']} ({company['symbol']})"
                    })
                
                return ProcessedCompanyInput(
                    ticker=best_match['ticker'],
                    company_name=best_match['company_name'],
                    search_terms=best_match['search_terms'],
                    sector=best_match['sector'],
                    status=SearchStatus.MULTIPLE_MATCHES,
                    confidence_score=best_match['confidence'],
                    raw_input=user_input,
                    suggestions=suggestions,
                    market_cap=best_match['market_cap'],
                    exchange=best_match['exchange'],
                    error_message=f"Multiple companies found. Best match: {best_match['company_name']}"
                )
            
            # Single good match
            return ProcessedCompanyInput(
                ticker=best_match['ticker'],
                company_name=best_match['company_name'],
                search_terms=best_match['search_terms'],
                sector=best_match['sector'],
                status=SearchStatus.SUCCESS,
                confidence_score=best_match['confidence'],
                raw_input=user_input,
                market_cap=best_match['market_cap'],
                exchange=best_match['exchange']
            )
            
        except Exception as e:
            self.logger.error(f"Error resolving company input '{user_input}': {e}")
            return ProcessedCompanyInput(
                ticker="", company_name="", search_terms=[], sector="",
                status=SearchStatus.ERROR, raw_input=user_input,
                error_message=f"An error occurred while processing your search: {str(e)}"
            )

    def get_search_suggestions(self, partial_input: str, limit: int = 10) -> List[Dict]:
        """
        Get search suggestions for autocomplete (uses popular companies for free tier)
        
        Args:
            partial_input: Partial user input
            limit: Maximum suggestions to return
            
        Returns:
            List of suggestion dictionaries
        """
        if not partial_input or len(partial_input) < 2:
            # Return popular companies
            return self._get_popular_suggestions(limit)
        
        suggestions = []
        partial_lower = partial_input.lower()
        
        # Check popular companies first (free, fast)
        for ticker in self.popular_tickers:
            company_data = self._validate_ticker_with_yfinance_enhanced(ticker)
            if company_data:
                company_name = company_data['company_name'].lower()
                symbol = company_data['symbol'].lower()
                
                # Check if partial input matches
                if (partial_lower in company_name or 
                    partial_lower in symbol or
                    company_name.startswith(partial_lower) or
                    symbol.startswith(partial_lower)):
                    
                    suggestions.append({
                        'ticker': company_data['symbol'],
                        'company_name': company_data['company_name'],
                        'sector': company_data['sector'],
                        'confidence': 0.8,
                        'display_name': f"{company_data['company_name']} ({company_data['symbol']})"
                    })
            
            if len(suggestions) >= limit:
                break
        
        return suggestions[:limit]
    
    def _get_popular_suggestions(self, limit: int = 10) -> List[Dict]:
        """Get popular company suggestions"""
        suggestions = []
        
        for ticker in self.popular_tickers[:limit]:
            company_data = self._validate_ticker_with_yfinance_enhanced(ticker)
            if company_data:
                suggestions.append({
                    'ticker': company_data['symbol'],
                    'company_name': company_data['company_name'],
                    'sector': company_data['sector'],
                    'confidence': 1.0,
                    'display_name': f"{company_data['company_name']} ({company_data['symbol']})"
                })
        
        return suggestions

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize resolver
    resolver = YFinanceCompanyResolver()
    
    # Test cases
    test_inputs = [
        "Reliance Industries",
        "TCS", 
        "hdfc bank",
        "Infosys",
        "sbi",
        "State Bank of India",
        "Bharti Airtel",
        "wipro",
        "adani enterprises",  # Test lesser known company
        "invalid company xyz",  # Should not be found
        "",  # Invalid input
        "R"  # Too short
    ]
    
    print("=== YFINANCE COMPANY RESOLUTION TESTS ===\n")
    
    for test_input in test_inputs:
        print(f"Input: '{test_input}'")
        print("-" * 40)
        
        result = resolver.resolve_company_input(test_input)
        
        print(f"Status: {result.status.value}")
        print(f"Company: {result.company_name}")
        print(f"Ticker: {result.ticker}")
        print(f"Sector: {result.sector}")
        print(f"Market Cap: {result.market_cap}")
        print(f"Exchange: {result.exchange}")
        print(f"Confidence: {result.confidence_score:.2f}")
        
        if result.search_terms:
            print(f"Search Terms: {', '.join(result.search_terms)}")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
            
        if result.suggestions:
            print("Suggestions:")
            for i, suggestion in enumerate(result.suggestions[:3], 1):
                print(f"  {i}. {suggestion['display_name']} (Confidence: {suggestion['confidence']:.2f})")
        
        print("=" * 50)
        print()
    
    # Test autocomplete
    print("=== AUTOCOMPLETE SUGGESTIONS TEST ===")
    partial_inputs = ["rel", "tat", "hdf", "sbi"]
    
    for partial in partial_inputs:
        suggestions = resolver.get_search_suggestions(partial, limit=3)
        print(f"\nPartial input: '{partial}'")
        for suggestion in suggestions:
            print(f"  {suggestion['display_name']}")
    
    print("\n=== INTEGRATION TEST ===")
    # Test the complete flow
    result = resolver.resolve_company_input("Reliance")
    if result.status == SearchStatus.SUCCESS:
        print(f"✅ SUCCESS: Ready to fetch news for {result.company_name}")
        print(f"   Ticker: {result.ticker}")
        print(f"   Search Terms: {result.search_terms}")
        print(f"   This data can now be passed to news_scraper.py")
    else:
        print(f"❌ FAILED: {result.error_message}")