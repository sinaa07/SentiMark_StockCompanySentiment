import sqlite3
import os
import logging
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NSEDatabaseManager:
    """Manager class for NSE stock database operations"""
    
    def __init__(self, db_path="data/nse_stocks.db"):
        self.db_path = db_path
        self.validate_database()
    
    def validate_database(self):
        """Ensure database file exists and is accessible"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}. Please run db_setup.py first.")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute query and return results as list of dictionaries"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Query execution error: {e}")
            return []
    
    # Core Search & Autocomplete Functions
    
    def search_companies(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Main autocomplete search function
        Returns ranked list of matching companies
        """
        if not query or len(query.strip()) == 0:
            return []
        
        query = query.strip().upper()
        
        # Multi-tier search for better ranking
        search_queries = [
            # Tier 1: Exact symbol match
            ("""
                SELECT symbol, company_name, series, isin_number, 1 as priority
                FROM nse_stocks 
                WHERE symbol = ?
                LIMIT ?
            """, (query, limit)),
            
            # Tier 2: Symbol starts with query
            ("""
                SELECT symbol, company_name, series, isin_number, 2 as priority
                FROM nse_stocks 
                WHERE symbol LIKE ? AND symbol != ?
                ORDER BY symbol
                LIMIT ?
            """, (f"{query}%", query, limit)),
            
            # Tier 3: Company name starts with query
            ("""
                SELECT symbol, company_name, series, isin_number, 3 as priority
                FROM nse_stocks 
                WHERE company_name LIKE ? 
                AND symbol NOT LIKE ?
                ORDER BY company_name
                LIMIT ?
            """, (f"{query}%", f"{query}%", limit)),
            
            # Tier 4: Company name contains query
            ("""
                SELECT symbol, company_name, series, isin_number, 4 as priority
                FROM nse_stocks 
                WHERE company_name LIKE ? 
                AND company_name NOT LIKE ?
                ORDER BY company_name
                LIMIT ?
            """, (f"%{query}%", f"{query}%", limit))
        ]
        
        all_results = []
        remaining_limit = limit
        
        for sql_query, params in search_queries:
            if remaining_limit <= 0:
                break
                
            # Update limit in params
            params = params[:-1] + (remaining_limit,)
            results = self.execute_query(sql_query, params)
            all_results.extend(results)
            remaining_limit = limit - len(all_results)
        
        # Remove duplicates while preserving order
        seen_symbols = set()
        unique_results = []
        for result in all_results:
            if result['symbol'] not in seen_symbols:
                seen_symbols.add(result['symbol'])
                unique_results.append(result)
        
        return unique_results[:limit]
    
    def get_company_suggestions(self, partial_text: str, limit: int = 10) -> List[Dict]:
        """
        Formatted suggestions for UI autocomplete
        Returns user-friendly format
        """
        results = self.search_companies(partial_text, limit)
        
        suggestions = []
        for result in results:
            suggestion = {
                'symbol': result['symbol'],
                'company_name': result['company_name'],
                'display_text': f"{result['symbol']} - {result['company_name']}",
                'series': result.get('series', ''),
                'value': result['symbol']  # For form inputs
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    # Ticker Validation & Resolution Functions
    
    def validate_ticker(self, symbol: str) -> bool:
        """Check if ticker symbol exists in database"""
        if not symbol:
            return False
        
        symbol = symbol.strip().upper()
        query = "SELECT COUNT(*) as count FROM nse_stocks WHERE symbol = ?"
        results = self.execute_query(query, (symbol,))
        
        return results[0]['count'] > 0 if results else False
    
    def get_company_details(self, symbol: str) -> Optional[Dict]:
        """
        Get complete company information for a symbol
        Returns detailed ticker object for news scraper
        """
        if not symbol:
            return None
        
        symbol = symbol.strip().upper()
        query = """
            SELECT symbol, company_name, series, listing_date, 
                   paid_up_value, market_lot, isin_number, face_value
            FROM nse_stocks 
            WHERE symbol = ?
        """
        
        results = self.execute_query(query, (symbol,))
        
        if not results:
            return None
        
        company = results[0]
        
        # Format for news scraper integration
        return {
            'symbol': company['symbol'],
            'nse_symbol': company['symbol'],
            'company_name': company['company_name'],
            'series': company['series'],
            'listing_date': company['listing_date'],
            'isin_number': company['isin_number'],
            'market_lot': company.get('market_lot'),
            'face_value': company.get('face_value'),
            'paid_up_value': company.get('paid_up_value'),
            'status': 'active',
            'source': 'nse_database'
        }
    
    # Advanced Search Features
    
    def search_by_sector(self, sector_name: str, limit: int = 50) -> List[Dict]:
        """Search companies by sector (if sector data available)"""
        # Note: Current NSE CSV doesn't have sector info
        # This is a placeholder for future enhancement
        logger.warning("Sector search not implemented - NSE CSV doesn't contain sector data")
        return []
    
    def search_multi_field(self, query: str, limit: int = 15) -> List[Dict]:
        """Enhanced search across multiple fields"""
        if not query:
            return []
        
        query = query.strip().upper()
        
        sql_query = """
            SELECT symbol, company_name, series, isin_number
            FROM nse_stocks 
            WHERE symbol LIKE ? OR company_name LIKE ? OR isin_number LIKE ?
            ORDER BY 
                CASE 
                    WHEN symbol = ? THEN 1
                    WHEN symbol LIKE ? THEN 2
                    WHEN company_name LIKE ? THEN 3
                    ELSE 4
                END,
                symbol, company_name
            LIMIT ?
        """
        
        params = (
            f"%{query}%", f"%{query}%", f"%{query}%",  # WHERE clause
            query, f"{query}%", f"{query}%",           # ORDER BY clause
            limit
        )
        
        return self.execute_query(sql_query, params)
    
    def get_similar_companies(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Find companies with similar names"""
        company = self.get_company_details(symbol)
        if not company:
            return []
        
        # Extract first word of company name for similarity
        company_words = company['company_name'].split()
        if not company_words:
            return []
        
        first_word = company_words[0]
        
        query = """
            SELECT symbol, company_name, series
            FROM nse_stocks 
            WHERE company_name LIKE ? AND symbol != ?
            ORDER BY company_name
            LIMIT ?
        """
        
        return self.execute_query(query, (f"{first_word}%", symbol, limit))
    
    # Data Retrieval Helpers
    
    def get_all_series(self) -> List[str]:
        """Get all unique series (EQ, BE, etc.)"""
        query = "SELECT DISTINCT series FROM nse_stocks WHERE series IS NOT NULL ORDER BY series"
        results = self.execute_query(query)
        return [row['series'] for row in results]
    
    def get_company_by_isin(self, isin: str) -> Optional[Dict]:
        """Alternative lookup by ISIN number"""
        if not isin:
            return None
        
        query = "SELECT * FROM nse_stocks WHERE isin_number = ?"
        results = self.execute_query(query, (isin.strip(),))
        
        return results[0] if results else None
    
    def get_recently_listed(self, days: int = 30, limit: int = 20) -> List[Dict]:
        """Get recently listed companies"""
        query = """
            SELECT symbol, company_name, listing_date, series
            FROM nse_stocks 
            WHERE listing_date IS NOT NULL 
                AND listing_date >= date('now', '-{} days')
            ORDER BY listing_date DESC
            LIMIT ?
        """.format(days)
        
        return self.execute_query(query, (limit,))
    
    def get_companies_by_series(self, series: str, limit: int = 100) -> List[Dict]:
        """Get companies by series (EQ, BE, etc.)"""
        query = """
            SELECT symbol, company_name, series
            FROM nse_stocks 
            WHERE series = ?
            ORDER BY company_name
            LIMIT ?
        """
        
        return self.execute_query(query, (series.upper(), limit))
    
    # Integration Functions
    
    def format_for_news_scraper(self, company_data: Dict) -> Dict:
        """
        Convert company data to format expected by news scraper
        """
        if not company_data:
            return {}
        
        # Generate search terms for news APIs
        search_terms = self.get_search_terms(company_data['symbol'])
        
        return {
            'ticker': company_data['symbol'],
            'nse_symbol': company_data['symbol'],
            'company_name': company_data['company_name'],
            'search_terms': search_terms,
            'series': company_data.get('series', ''),
            'isin': company_data.get('isin_number', ''),
            'validated': True,
            'source': 'nse_database'
        }
    
    def get_search_terms(self, symbol: str) -> List[str]:
        """
        Generate search terms for news scraping
        """
        company = self.get_company_details(symbol)
        if not company:
            return [symbol]
        
        search_terms = [
            symbol,  # TCS
            company['company_name'],  # Tata Consultancy Services
        ]
        
        # Add variations of company name
        company_name = company['company_name']
        
        # Remove common suffixes for better search
        suffixes_to_remove = [' Ltd', ' Limited', ' Pvt', ' Private', ' Company', ' Corp', ' Corporation']
        clean_name = company_name
        for suffix in suffixes_to_remove:
            clean_name = clean_name.replace(suffix, '')
        
        if clean_name != company_name:
            search_terms.append(clean_name.strip())
        
        # Add first few words for partial matching
        words = clean_name.split()
        if len(words) >= 2:
            search_terms.append(' '.join(words[:2]))
        
        return list(set(search_terms))  # Remove duplicates
    
    # Utility Functions
    
    def database_stats(self) -> Dict:
        """Get database statistics and health check"""
        stats_queries = [
            ("Total Companies", "SELECT COUNT(*) as count FROM nse_stocks"),
            ("Active Series", "SELECT COUNT(DISTINCT series) as count FROM nse_stocks WHERE series IS NOT NULL"),
            ("Companies with ISIN", "SELECT COUNT(*) as count FROM nse_stocks WHERE isin_number IS NOT NULL"),
            ("Recent Listings (30 days)", "SELECT COUNT(*) as count FROM nse_stocks WHERE listing_date >= date('now', '-30 days')"),
            ("Database Size (KB)", "SELECT page_count * page_size / 1024 as size FROM pragma_page_count(), pragma_page_size()"),
        ]
        
        stats = {}
        for description, query in stats_queries:
            try:
                results = self.execute_query(query)
                stats[description] = results[0]['count'] if 'count' in results[0] else results[0]['size']
            except Exception as e:
                stats[description] = f"Error: {str(e)}"
        
        return stats
    
    def test_connection(self) -> Dict:
        """Test database connection and basic operations"""
        try:
            # Test connection
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM nse_stocks")
                count = cursor.fetchone()[0]
            
            # Test search functionality
            test_searches = [
                self.search_companies("TCS", 3),
                self.search_companies("ADANI", 3),
                self.get_company_details("TCS") is not None
            ]
            
            return {
                'status': 'success',
                'total_records': count,
                'search_tests_passed': all(test_searches),
                'database_path': self.db_path
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'database_path': self.db_path
            }
    
    def backup_database(self, backup_path: str = None) -> str:
        """Create a backup copy of the database"""
        if not backup_path:
            backup_path = f"{self.db_path}.backup"
        
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise

# Convenience function for quick usage
def get_nse_manager(db_path="data/nse_stocks.db") -> NSEDatabaseManager:
    """Factory function to create NSEDatabaseManager instance"""
    return NSEDatabaseManager(db_path)

# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize manager
        db_manager = NSEDatabaseManager()
        
        # Test connection
        print("=== Database Connection Test ===")
        connection_test = db_manager.test_connection()
        print(json.dumps(connection_test, indent=2))
        
        # Test search functionality
        print("\n=== Search Tests ===")
        
        # Test Adani search
        print("\n1. Searching for 'ADANI':")
        adani_results = db_manager.search_companies("ADANI", 5)
        for result in adani_results:
            print(f"  {result['symbol']} - {result['company_name']}")
        
        # Test TCS search
        print("\n2. Searching for 'TCS':")
        tcs_results = db_manager.search_companies("TCS", 3)
        for result in tcs_results:
            print(f"  {result['symbol']} - {result['company_name']}")
        
        # Test company details
        print("\n3. Getting TCS company details:")
        tcs_details = db_manager.get_company_details("TCS")
        if tcs_details:
            print(f"  Company: {tcs_details['company_name']}")
            print(f"  Symbol: {tcs_details['symbol']}")
            print(f"  Series: {tcs_details['series']}")
        
        # Test suggestions format
        print("\n4. Autocomplete suggestions for 'REL':")
        suggestions = db_manager.get_company_suggestions("REL", 3)
        for suggestion in suggestions:
            print(f"  {suggestion['display_text']}")
        
        # Database stats
        print("\n=== Database Statistics ===")
        stats = db_manager.database_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you've run db_setup.py first to create the database.")