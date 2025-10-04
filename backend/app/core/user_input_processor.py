import time
import re
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import json
from app.services.db_manager import NSEDatabaseManager
from app.utils import resolve_path
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserInputProcessor:
    """Enhanced user input processor with database integration and autocomplete"""
    
    def __init__(self, db_path: str=None, min_chars=2, debounce_ms=300):
        if db_path is None:
            self.db_path = resolve_path("data/nse_stocks.db")
        else:
            self.db_path = resolve_path(db_path)
        self.db_manager = NSEDatabaseManager(self.db_path)
        self.min_chars = min_chars
        self.debounce_delay = debounce_ms / 1000.0  # Convert to seconds
        self.last_query_time = {}
        self.recent_searches = []
        self.max_recent_searches = 10
        
        logger.info(f"UserInputProcessor initialized with min_chars={min_chars}, debounce={debounce_ms}ms")
    
    def process_raw_input(self, user_text: str) -> str:
        """
        Clean and sanitize user input
        Remove special characters, normalize spacing, handle edge cases
        """
        if not user_text:
            return ""
        
        # Convert to string and strip whitespace
        cleaned = str(user_text).strip()
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters but keep alphanumeric, spaces, hyphens, dots
        cleaned = re.sub(r'[^a-zA-Z0-9\s\-\.]', '', cleaned)
        
        # Normalize case - uppercase for better matching
        cleaned = cleaned.upper()
        
        logger.debug(f"Cleaned input: '{user_text}' -> '{cleaned}'")
        return cleaned
    
    def validate_input_length(self, text: str, min_chars: int = None) -> bool:
        """
        Check if input meets minimum character requirement
        """
        min_length = min_chars if min_chars is not None else self.min_chars
        
        if not text or len(text.strip()) < min_length:
            logger.debug(f"Input too short: '{text}' (min: {min_length})")
            return False
        
        return True
    
    def apply_debouncing(self, text: str, user_session: str = "default") -> bool:
        """
        Wait for user to stop typing before processing
        Returns True if should proceed with search, False if should wait
        """
        current_time = time.time()
        
        # Store the current query time for this session
        self.last_query_time[user_session] = current_time
        
        # Wait for debounce delay
        time.sleep(self.debounce_delay)
        
        # Check if user has typed again during the delay
        latest_time = self.last_query_time.get(user_session, 0)
        
        if current_time < latest_time:
            # User typed again, skip this request
            logger.debug(f"Debounced query skipped: '{text}'")
            return False
        
        logger.debug(f"Debounce passed for query: '{text}'")
        return True
    
    # 2. Search Bar Autocomplete
    
    def get_autocomplete_suggestions(self, query: str, limit: int = 10, 
                                   user_session: str = "default") -> List[Dict]:
        """
        Main autocomplete function for search bar
        Handles debouncing, validation, and database integration
        """
        try:
            # Clean the input
            cleaned_query = self.process_raw_input(query)
            
            # Validate minimum length
            if not self.validate_input_length(cleaned_query):
                return []
            
            # Apply debouncing (in production, this would be handled by frontend)
            # For now, we'll skip debouncing in this function to avoid blocking
            logger.debug(f"Getting suggestions for: '{cleaned_query}'")
            
            # Get raw results from database
            raw_results = self.db_manager.search_companies(cleaned_query, limit)
            
            if not raw_results:
                logger.info(f"No results found for query: '{cleaned_query}'")
                return []
            
            # Format for UI
            formatted_suggestions = self.format_suggestions_for_ui(raw_results, cleaned_query)
            
            logger.info(f"Returning {len(formatted_suggestions)} suggestions for '{cleaned_query}'")
            return formatted_suggestions
            
        except Exception as e:
            logger.error(f"Error in get_autocomplete_suggestions: {e}")
            return self.handle_errors("autocomplete_error", {"query": query, "error": str(e)})
    
    def format_suggestions_for_ui(self, raw_results: List[Dict], search_query: str = "") -> List[Dict]:
        """
        Convert database results to UI-friendly format for dropdown display
        """
        formatted_suggestions = []
        
        for result in raw_results:
            try:
                # Create display text
                symbol = result.get('symbol', '')
                company_name = result.get('company_name', '')
                series = result.get('series', '')
                
                # Handle long company names
                max_name_length = 50
                if len(company_name) > max_name_length:
                    display_name = company_name[:max_name_length] + "..."
                else:
                    display_name = company_name
                
                # Create display text
                display_text = f"{symbol} - {display_name}"
                
                # Create subtitle with additional info
                subtitle = f"{series} Series" if series else "Listed Stock"
                
                # Highlight search term in company name (basic implementation)
                highlighted_name = self._highlight_search_term(display_name, search_query)
                
                suggestion = {
                    'display_text': display_text,
                    'highlighted_text': f"{symbol} - {highlighted_name}",
                    'symbol': symbol,
                    'company_name': company_name,
                    'value': symbol,  # What gets selected
                    'subtitle': subtitle,
                    'series': series,
                    'search_relevance': result.get('priority', 4),  # From multi-tier search
                    'isin': result.get('isin_number', '')
                }
                
                formatted_suggestions.append(suggestion)
                
            except Exception as e:
                logger.error(f"Error formatting suggestion: {e}")
                continue
        
        return formatted_suggestions
    
    def _highlight_search_term(self, text: str, search_term: str) -> str:
        """
        Add highlighting markers for search term in text
        In real UI, this would be replaced with HTML/CSS highlighting
        """
        if not search_term or not text:
            return text
        
        try:
            # Simple case-insensitive highlighting
            highlighted = re.sub(
                f'({re.escape(search_term)})', 
                r'**\1**',  # Markdown-style highlighting
                text, 
                flags=re.IGNORECASE
            )
            return highlighted
        except Exception:
            return text
    
    # 3. User Selection Processing
    
    def handle_user_selection(self, selected_symbol: str) -> Dict[str, Any]:
        """
        Main function to process user's company selection
        Returns complete ticker data ready for news scraper
        """
        try:
            if not selected_symbol:
                return self.handle_errors("invalid_selection", {"symbol": selected_symbol})
            
            # Clean the selected symbol
            clean_symbol = self.process_raw_input(selected_symbol)
            
            logger.info(f"Processing user selection: {clean_symbol}")
            
            # Validate ticker exists (direct call to db_manager)
            if not self.db_manager.validate_ticker(clean_symbol):
                return self.handle_errors("ticker_not_found", {"symbol": clean_symbol})
            
            # Get complete company data (direct call to db_manager)
            company_data = self.db_manager.get_company_details(clean_symbol)
            
            if not company_data:
                return self.handle_errors("data_fetch_failed", {"symbol": clean_symbol})
            
            # Cache this selection
            self.cache_recent_searches(company_data)
            
            # Prepare final output for news scraper
            final_output = self.prepare_for_news_scraper(company_data)
            
            logger.info(f"Successfully processed selection: {clean_symbol}")
            return final_output
            
        except Exception as e:
            logger.error(f"Error handling user selection: {e}")
            return self.handle_errors("selection_processing_error", {
                "symbol": selected_symbol, 
                "error": str(e)
            })
    
    def prepare_for_news_scraper(self, company_data: Dict) -> Dict[str, Any]:
        """
        Format company data for news scraper integration
        Creates standardized output with all necessary fields
        """
        try:
            if not company_data:
                return {}
            
            # Generate optimized search terms
            search_terms = self._generate_search_terms(company_data)
            
            # Create final ticker object for news scraper
            ticker_object = {
                # Core identification
                'symbol': company_data['symbol'],
                'nse_symbol': company_data['symbol'],
                'company_name': company_data['company_name'],
                
                # Search optimization for news scraper
                'search_terms': search_terms,
                'primary_search_term': company_data['symbol'],
                'secondary_search_term': company_data['company_name'],
                
                # Additional metadata
                'series': company_data.get('series', ''),
                'isin': company_data.get('isin_number', ''),
                'listing_date': company_data.get('listing_date'),
                'market_lot': company_data.get('market_lot'),
                'face_value': company_data.get('face_value'),
                
                # Processing metadata
                'validated': True,
                'source': 'nse_database',
                'processed_timestamp': datetime.now(timezone.utc).isoformat(),
                'ready_for_news_scraper': True,
                
                # Configuration for news scraper
                'news_config': {
                    'date_range_days': 30,
                    'max_articles': 50,
                    'sources': ['economic_times', 'business_standard', 'moneycontrol'],
                    'relevance_threshold': 0.7
                }
            }
            
            logger.debug(f"Prepared ticker object for: {company_data['symbol']}")
            return ticker_object
            
        except Exception as e:
            logger.error(f"Error preparing data for news scraper: {e}")
            return self.handle_errors("preparation_error", {"company_data": company_data})
    
    def _generate_search_terms(self, company_data: Dict) -> List[str]:
        """
        Generate optimized search terms for news scraping
        Company-specific focus (not group-level as discussed)
        """
        search_terms = []
        
        symbol = company_data.get('symbol', '')
        company_name = company_data.get('company_name', '')
        
        if symbol:
            search_terms.append(symbol)
        
        if company_name:
            search_terms.append(company_name)
            
            # Add clean company name without suffixes
            clean_name = company_name
            suffixes_to_remove = [
                ' Ltd', ' Limited', ' Pvt', ' Private', ' Company', 
                ' Corp', ' Corporation', ' Inc', ' Incorporated'
            ]
            
            for suffix in suffixes_to_remove:
                clean_name = clean_name.replace(suffix, '')
            
            clean_name = clean_name.strip()
            if clean_name != company_name and clean_name:
                search_terms.append(clean_name)
            
            # Add first two words for broader matching
            words = clean_name.split()
            if len(words) >= 2:
                first_two_words = ' '.join(words[:2])
                if first_two_words not in search_terms:
                    search_terms.append(first_two_words)
        
        # Remove duplicates while preserving order
        unique_terms = []
        for term in search_terms:
            if term and term not in unique_terms:
                unique_terms.append(term)
        
        return unique_terms
    
    # 4. Utility Functions
    
    def cache_recent_searches(self, selection: Dict) -> None:
        """
        Store user's recent selections for quick access
        """
        try:
            # Create cache entry
            cache_entry = {
                'symbol': selection.get('symbol', ''),
                'company_name': selection.get('company_name', ''),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'series': selection.get('series', '')
            }
            
            # Remove if already exists (move to front)
            self.recent_searches = [
                item for item in self.recent_searches 
                if item.get('symbol') != cache_entry['symbol']
            ]
            
            # Add to front
            self.recent_searches.insert(0, cache_entry)
            
            # Limit cache size
            if len(self.recent_searches) > self.max_recent_searches:
                self.recent_searches = self.recent_searches[:self.max_recent_searches]
            
            logger.debug(f"Cached recent search: {selection.get('symbol', '')}")
            
        except Exception as e:
            logger.error(f"Error caching recent search: {e}")
    
    def get_recent_searches(self, limit: int = 5) -> List[Dict]:
        """
        Get user's recent searches for quick selection
        """
        try:
            return self.recent_searches[:limit]
        except Exception as e:
            logger.error(f"Error getting recent searches: {e}")
            return []
    
    def handle_no_results(self, query: str) -> List[Dict]:
        """
        Handle when database returns no results
        Provide helpful suggestions or alternatives
        """
        try:
            # Try broader search with first few characters
            if len(query) > 3:
                broader_query = query[:3]
                broader_results = self.db_manager.search_companies(broader_query, 5)
                
                if broader_results:
                    return [{
                        'type': 'suggestion',
                        'message': f'No exact matches for "{query}". Did you mean:',
                        'suggestions': self.format_suggestions_for_ui(broader_results, query)
                    }]
            
            # Suggest popular stocks or recent searches
            recent = self.get_recent_searches(3)
            if recent:
                return [{
                    'type': 'recent',
                    'message': 'No results found. Your recent searches:',
                    'suggestions': recent
                }]
            
            # Default fallback
            return [{
                'type': 'no_results',
                'message': f'No companies found matching "{query}". Please try a different search term.',
                'suggestions': []
            }]
            
        except Exception as e:
            logger.error(f"Error handling no results: {e}")
            return []
    
    def handle_errors(self, error_type: str, details: Dict) -> Dict[str, Any]:
        """
        Centralized error handling with consistent response format
        """
        error_messages = {
            'invalid_selection': 'Please select a valid company',
            'ticker_not_found': 'Company not found in database',
            'data_fetch_failed': 'Unable to fetch company information',
            'selection_processing_error': 'Error processing your selection',
            'autocomplete_error': 'Error fetching suggestions',
            'preparation_error': 'Error preparing data for processing'
        }
        
        error_response = {
            'success': False,
            'error': True,
            'error_type': error_type,
            'error_message': error_messages.get(error_type, 'An unexpected error occurred'),
            'details': details,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.error(f"Error handled: {error_type} - {details}")
        return error_response
    
    # Additional utility methods
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the processor and database connection
        """
        try:
            # Test database connection
            db_test = self.db_manager.test_connection()
            
            # Test basic functionality
            test_queries = ['TCS', 'ADANI', 'RELIANCE']
            test_results = {}
            
            for query in test_queries:
                suggestions = self.get_autocomplete_suggestions(query, 3)
                test_results[query] = len(suggestions)
            
            return {
                'status': 'success',
                'database_status': db_test.get('status', 'unknown'),
                'test_queries': test_results,
                'recent_searches_count': len(self.recent_searches),
                'configuration': {
                    'min_chars': self.min_chars,
                    'debounce_ms': int(self.debounce_delay * 1000)
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics and usage info
        """
        try:
            db_stats = self.db_manager.database_stats()
            
            return {
                'database_stats': db_stats,
                'recent_searches_count': len(self.recent_searches),
                'configuration': {
                    'min_chars': self.min_chars,
                    'debounce_delay_ms': int(self.debounce_delay * 1000),
                    'max_recent_searches': self.max_recent_searches
                },
                'recent_searches': self.get_recent_searches(5)
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}

# Convenience factory function
def create_user_input_processor(db_path="data/nse_stocks.db", 
                               min_chars=2, 
                               debounce_ms=300) -> UserInputProcessor:
    """Factory function to create UserInputProcessor instance"""
    return UserInputProcessor(db_path, min_chars, debounce_ms)

# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize processor
        processor = UserInputProcessor()
        
        print("=== User Input Processor Test ===")
        
        # Test connection
        connection_test = processor.test_connection()
        print("\n1. Connection Test:")
        print(json.dumps(connection_test, indent=2))
        
        # Test autocomplete
        print("\n2. Autocomplete Test:")
        
        test_queries = ["ADANI", "TCS", "REL"]
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            suggestions = processor.get_autocomplete_suggestions(query, 3)
            
            if suggestions and not suggestions[0].get('error'):
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"   {i}. {suggestion.get('display_text', 'N/A')}")
            else:
                print(f"   No results or error for '{query}'")
        
        # Test selection handling
        print("\n3. Selection Test:")
        selection_result = processor.handle_user_selection("TCS")
        
        if not selection_result.get('error'):
            print(f"   Selected: {selection_result.get('company_name', 'N/A')}")
            print(f"   Symbol: {selection_result.get('symbol', 'N/A')}")
            print(f"   Search Terms: {selection_result.get('search_terms', [])}")
            print(f"   Ready for News Scraper: {selection_result.get('ready_for_news_scraper', False)}")
        else:
            print(f"   Selection Error: {selection_result.get('error_message', 'Unknown error')}")
        
        # Test recent searches
        print("\n4. Recent Searches:")
        recent = processor.get_recent_searches(3)
        for search in recent:
            print(f"   - {search.get('symbol', 'N/A')} ({search.get('company_name', 'N/A')})")
        
        # Get processor stats
        print("\n5. Processor Statistics:")
        stats = processor.get_stats()
        if 'error' not in stats:
            print(f"   Database companies: {stats.get('database_stats', {}).get('Total Companies', 'N/A')}")
            print(f"   Recent searches: {stats.get('recent_searches_count', 0)}")
            print(f"   Min characters: {stats.get('configuration', {}).get('min_chars', 'N/A')}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure db_setup.py has been run successfully.")