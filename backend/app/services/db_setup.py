import sqlite3
import pandas as pd
import os
from datetime import datetime
import logging
from app.utils import resolve_path
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NSEDatabaseSetup:
    def __init__(self, db_path="data/nse_stocks.db", csv_path="data/nse_master.csv"):
        self.db_path = resolve_path(db_path)
        self.csv_path = resolve_path(csv_path)
        
    def create_database_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        logger.info(f"Ensured directory exists: {os.path.dirname(self.db_path)}")
    
    def create_connection(self):
        """Create SQLite database connection"""
        try:
            conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to SQLite database: {self.db_path}")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def create_tables(self, conn):
        """Create tables and indexes"""
        try:
            cursor = conn.cursor()
            
            # Create main nse_stocks table
            create_table_sql = '''
            CREATE TABLE IF NOT EXISTS nse_stocks (
                symbol TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                series TEXT,
                listing_date DATE,
                paid_up_value REAL,
                market_lot INTEGER,
                isin_number TEXT,
                face_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            '''
            cursor.execute(create_table_sql)
            logger.info("Created nse_stocks table")
            
            # Create indexes for fast searching
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_company_name ON nse_stocks(company_name);",
                "CREATE INDEX IF NOT EXISTS idx_symbol ON nse_stocks(symbol);",
                "CREATE INDEX IF NOT EXISTS idx_series ON nse_stocks(series);",
                "CREATE INDEX IF NOT EXISTS idx_search_combined ON nse_stocks(symbol, company_name);"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            logger.info("Created all indexes")
            conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def clean_data(self, df):
        """Clean and validate CSV data before insertion"""
        logger.info(f"Cleaning data for {len(df)} records")
        
        # Clean column names (remove spaces, special characters)
        df.columns = df.columns.str.strip()
        
        # Map CSV columns to database columns
        column_mapping = {
            'SYMBOL': 'symbol',
            'NAME OF COMPANY': 'company_name',
            'SERIES': 'series',
            'DATE OF LISTING': 'listing_date',
            'PAID UP VALUE': 'paid_up_value',
            'MARKET LOT': 'market_lot',
            'ISIN NUMBER': 'isin_number',
            'FACE VALUE': 'face_value'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Clean data
        df['symbol'] = df['symbol'].str.strip().str.upper()
        df['company_name'] = df['company_name'].str.strip()
        df['series'] = df['series'].str.strip().str.upper()
        df['isin_number'] = df['isin_number'].str.strip()
        
        # Handle date conversion
        if 'listing_date' in df.columns:
            df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
            df['listing_date'] = df['listing_date'].dt.strftime('%Y-%m-%d')
        
        # Handle numeric columns
        numeric_columns = ['paid_up_value', 'market_lot', 'face_value']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing symbol or company name
        initial_count = len(df)
        df = df.dropna(subset=['symbol', 'company_name'])
        final_count = len(df)
        
        if initial_count != final_count:
            logger.warning(f"Removed {initial_count - final_count} rows due to missing symbol/company name")
        
        # Remove duplicates based on symbol
        df = df.drop_duplicates(subset=['symbol'], keep='first')
        logger.info(f"Final cleaned dataset: {len(df)} records")
        
        return df
    
    def import_csv_data(self, conn):
        """Import NSE CSV data into SQLite database"""
        try:
            # Check if CSV file exists
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
            logger.info(f"Reading CSV file: {self.csv_path}")
            
            # Read CSV file
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            logger.info(f"Read {len(df)} records from CSV")
            
            # Clean the data
            df_clean = self.clean_data(df)
            
            # Insert data into database
            df_clean.to_sql('nse_stocks', conn, if_exists='replace', index=False, method='multi')
            logger.info(f"Successfully imported {len(df_clean)} records into database")
            
            # Get some stats
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nse_stocks")
            total_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT series) FROM nse_stocks")
            unique_series = cursor.fetchone()[0]
            
            logger.info(f"Database stats: {total_records} total records, {unique_series} unique series")
            
        except Exception as e:
            logger.error(f"Error importing CSV data: {e}")
            raise
    
    def create_sample_queries(self, conn):
        """Test database with sample queries"""
        try:
            cursor = conn.cursor()
            
            # Test queries
            test_queries = [
                ("Total records", "SELECT COUNT(*) FROM nse_stocks"),
                ("Sample records", "SELECT symbol, company_name FROM nse_stocks LIMIT 5"),
                ("Adani companies", "SELECT symbol, company_name FROM nse_stocks WHERE company_name LIKE '%Adani%'"),
                ("TCS search", "SELECT symbol, company_name FROM nse_stocks WHERE symbol LIKE '%TCS%' OR company_name LIKE '%TCS%'")
            ]
            
            logger.info("\n=== Database Test Queries ===")
            for description, query in test_queries:
                cursor.execute(query)
                results = cursor.fetchall()
                logger.info(f"{description}: {results}")
            
        except sqlite3.Error as e:
            logger.error(f"Error running test queries: {e}")
    
    def setup_database(self):
        """Main method to set up the entire database"""
        try:
            logger.info("Starting NSE database setup...")
            
            # Create directory
            self.create_database_directory()
            
            # Create connection
            conn = self.create_connection()
            
            # Create tables and indexes
            self.create_tables(conn)
            
            # Import CSV data
            self.import_csv_data(conn)
            
            # Run test queries
            self.create_sample_queries(conn)
            
            # Close connection
            conn.close()
            
            logger.info("Database setup completed successfully!")
            logger.info(f"Database file created at: {os.path.abspath(self.db_path)}")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise

def main():
    """Main function to run database setup"""
    # Initialize with your CSV file path
    csv_file_path = input("Enter path to your NSE CSV file (or press Enter for 'data/nse_master.csv'): ").strip()
    if not csv_file_path:
        csv_file_path = "data/nse_master.csv"
    
    # Setup database
    db_setup = NSEDatabaseSetup(csv_path=csv_file_path)
    db_setup.setup_database()

if __name__ == "__main__":
    main()