import yfinance as yf
import json
import os
import logging
from typing import Dict, Optional
from datetime import datetime

class BasicStockInfo:
    """Simple stock info fetcher for MVP - just basic company details"""
    
    def __init__(self):
        # Load company mappings
        self.mappings_file = "data/stock_mappings.json"
        self.company_mappings = self._load_mappings()
        
    def _load_mappings(self) -> Dict:
        """Load company name to ticker mappings"""
        try:
            with open(self.mappings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
            
    def get_company_basic_info(self, ticker_or_name: str) -> Dict:
        """
        Get basic company information for display context
        
        Args:
            ticker_or_name: Stock ticker (e.g., 'RELIANCE') or company name
            
        Returns:
            Dictionary with basic company info for display
        """
        try:
            # Find ticker from mappings
            ticker = ticker_or_name.upper().replace('.NS', '').replace('.BO', '')
            
            if ticker in self.company_mappings:
                company_info = self.company_mappings[ticker]
                return {
                    'ticker': f"{ticker}.NS",
                    'company_name': company_info['company_name'],
                    'search_terms': company_info['search_terms'],
                    'sector': company_info.get('sector', 'Unknown'),
                    'status': 'found'
                }
            else:
                # Try to search by company name in mappings
                for mapped_ticker, info in self.company_mappings.items():
                    if (ticker_or_name.lower() in info['company_name'].lower() or 
                        any(term.lower() in ticker_or_name.lower() for term in info['search_terms'])):
                        return {
                            'ticker': f"{mapped_ticker}.NS",
                            'company_name': info['company_name'], 
                            'search_terms': info['search_terms'],
                            'sector': info.get('sector', 'Unknown'),
                            'status': 'found'
                        }
                
                # If not found in mappings, return basic info
                return {
                    'ticker': ticker_or_name,
                    'company_name': ticker_or_name,
                    'search_terms': [ticker_or_name],
                    'sector': 'Unknown',
                    'status': 'not_in_database'
                }
                
        except Exception as e:
            logging.error(f"Error getting company info: {e}")
            return {
                'ticker': ticker_or_name,
                'company_name': ticker_or_name,
                'search_terms': [ticker_or_name],
                'sector': 'Unknown', 
                'status': 'error'
            }
    
    def get_current_price(self, ticker: str) -> Dict:
        """
        Get current stock price (optional for MVP - just for display)
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with current price info or None if not needed
        """
        try:
            # Add .NS if not present
            if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
                ticker = f"{ticker}.NS"
                
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")  # Just get last 2 days
            
            if hist.empty:
                return {'status': 'no_data', 'ticker': ticker}
                
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
            
            return {
                'ticker': ticker,
                'current_price': round(current_price, 2),
                'price_change': round(change, 2),
                'price_change_pct': round(change_pct, 2),
                'status': 'success',
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logging.warning(f"Could not fetch price for {ticker}: {e}")
            return {'status': 'error', 'ticker': ticker}
    
    def validate_ticker(self, ticker_or_name: str) -> bool:
        """
        Simple validation to check if company exists
        
        Args:
            ticker_or_name: Ticker or company name to validate
            
        Returns:
            True if company is found, False otherwise
        """
        info = self.get_company_basic_info(ticker_or_name)
        return info['status'] == 'found'
    
    def get_popular_stocks(self) -> List[Dict]:
        """
        Get list of popular Indian stocks for dropdown/suggestions
        
        Returns:
            List of popular stocks with basic info
        """
        popular_stocks = []
        
        # Get first 10 companies from mappings as popular stocks
        for ticker, info in list(self.company_mappings.items())[:10]:
            popular_stocks.append({
                'ticker': ticker,
                'company_name': info['company_name'],
                'display_name': f"{info['company_name']} ({ticker})"
            })
            
        return popular_stocks

# Enhanced stock mappings with sectors
ENHANCED_STOCK_MAPPINGS = {
    "RELIANCE": {
        "ticker": "RELIANCE.NS",
        "company_name": "Reliance Industries Limited",
        "search_terms": ["Reliance Industries", "RIL", "Reliance"],
        "sector": "Oil & Gas"
    },
    "TCS": {
        "ticker": "TCS.NS", 
        "company_name": "Tata Consultancy Services",
        "search_terms": ["TCS", "Tata Consultancy", "Tata Consultancy Services"],
        "sector": "IT Services"
    },
    "INFY": {
        "ticker": "INFY.NS",
        "company_name": "Infosys Limited", 
        "search_terms": ["Infosys", "INFY"],
        "sector": "IT Services"
    },
    "HDFCBANK": {
        "ticker": "HDFCBANK.NS",
        "company_name": "HDFC Bank Limited",
        "search_terms": ["HDFC Bank", "HDFC"],
        "sector": "Banking"
    },
    "ICICIBANK": {
        "ticker": "ICICIBANK.NS", 
        "company_name": "ICICI Bank Limited",
        "search_terms": ["ICICI Bank", "ICICI"],
        "sector": "Banking"
    },
    "HINDUNILVR": {
        "ticker": "HINDUNILVR.NS",
        "company_name": "Hindustan Unilever Limited", 
        "search_terms": ["Hindustan Unilever", "HUL"],
        "sector": "FMCG"
    },
    "ITC": {
        "ticker": "ITC.NS",
        "company_name": "ITC Limited",
        "search_terms": ["ITC", "ITC Limited"],
        "sector": "FMCG"
    },
    "SBIN": {
        "ticker": "SBIN.NS", 
        "company_name": "State Bank of India",
        "search_terms": ["State Bank of India", "SBI"],
        "sector": "Banking"
    },
    "BHARTIARTL": {
        "ticker": "BHARTIARTL.NS",
        "company_name": "Bharti Airtel Limited",
        "search_terms": ["Bharti Airtel", "Airtel"],
        "sector": "Telecom"
    },
    "KOTAKBANK": {
        "ticker": "KOTAKBANK.NS",
        "company_name": "Kotak Mahindra Bank Limited", 
        "search_terms": ["Kotak Mahindra Bank", "Kotak Bank"],
        "sector": "Banking"
    },
    "LT": {
        "ticker": "LT.NS",
        "company_name": "Larsen & Toubro Limited",
        "search_terms": ["Larsen & Toubro", "L&T"],
        "sector": "Infrastructure"
    },
    "WIPRO": {
        "ticker": "WIPRO.NS",
        "company_name": "Wipro Limited",
        "search_terms": ["Wipro"],
        "sector": "IT Services"
    },
    "MARUTI": {
        "ticker": "MARUTI.NS",
        "company_name": "Maruti Suzuki India Limited",
        "search_terms": ["Maruti Suzuki", "Maruti"],
        "sector": "Automotive"
    },
    "BAJFINANCE": {
        "ticker": "BAJFINANCE.NS",
        "company_name": "Bajaj Finance Limited",
        "search_terms": ["Bajaj Finance"],
        "sector": "NBFC"
    },
    "HCLTECH": {
        "ticker": "HCLTECH.NS",
        "company_name": "HCL Technologies Limited",
        "search_terms": ["HCL Technologies", "HCL Tech"],
        "sector": "IT Services"
    }
}

def create_enhanced_mappings_file():
    """Create enhanced stock mappings file with sectors"""
    os.makedirs("data", exist_ok=True)
    
    with open("data/stock_mappings.json", 'w', encoding='utf-8') as f:
        json.dump(ENHANCED_STOCK_MAPPINGS, f, ensure_ascii=False, indent=2)
    
    print("Enhanced stock mappings file created!")

# Example usage for testing
if __name__ == "__main__":
    # Create enhanced mappings file
    create_enhanced_mappings_file()
    
    # Test the basic stock info
    stock_info = BasicStockInfo()
    
    # Test company lookup
    print("Testing company lookup...")
    companies = ["RELIANCE", "TCS", "Infosys", "HDFC Bank"]
    
    for company in companies:
        info = stock_info.get_company_basic_info(company)
        print(f"\n{company}:")
        print(f"  Company: {info['company_name']}")
        print(f"  Ticker: {info['ticker']}")
        print(f"  Sector: {info['sector']}")
        print(f"  Status: {info['status']}")
        
        # Optional: Get current price (comment out if you don't want price data)
        # price_info = stock_info.get_current_price(info['ticker'])
        # if price_info['status'] == 'success':
        #     print(f"  Current Price: ₹{price_info['current_price']}")
        #     print(f"  Change: {price_info['price_change']} ({price_info['price_change_pct']:.2f}%)")
    
    # Test popular stocks
    print("\nPopular stocks:")
    popular = stock_info.get_popular_stocks()
    for stock in popular[:5]:
        print(f"  {stock['display_name']}")