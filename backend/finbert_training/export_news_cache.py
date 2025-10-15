# export_news_cache.py
"""
Export real news articles from SQLite news_cache database to CSV.
Prepares data for pseudo-labeling and training.
"""

import sqlite3
import pandas as pd
import argparse
from pathlib import Path
from typing import Optional


def export_news_cache_to_csv(
    db_path: str,
    export_path: str,
    min_relevance: float = 0.0
) -> pd.DataFrame:
    """
    Export news articles from SQLite database to CSV format.
    
    Args:
        db_path: Path to the SQLite database file
        export_path: Path where CSV will be saved
        min_relevance: Minimum relevance score filter (default: 0.0)
        
    Returns:
        DataFrame containing exported news articles
        
    Raises:
        FileNotFoundError: If database file doesn't exist
        sqlite3.Error: If database query fails
    """
    # Check if database exists
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    print(f"Connecting to database: {db_path}")
    
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        
        # SQL query to extract relevant news data
        query = """
        SELECT 
            article_title,
            article_content,
            source_name,
            published_date,
            article_url,
            relevance_score,
            company_symbol
        FROM news_cache
        WHERE relevance_score >= ?
        ORDER BY published_date DESC
        """
        
        print(f"Executing query with min_relevance >= {min_relevance}...")
        
        # Execute query and load into DataFrame
        df = pd.read_sql_query(query, conn, params=(min_relevance,))
        
        conn.close()
        
        print(f"✅ Loaded {len(df)} articles from database")
        
        if df.empty:
            print("⚠️  No articles found matching the criteria")
            return df
        
        # Combine title and content into single text field
        print("Processing text fields...")
        df['text'] = df.apply(
            lambda row: f"{row['article_title']}. {row['article_content']}"
            if pd.notna(row['article_content']) and row['article_content'].strip()
            else row['article_title'],
            axis=1
        )
        
        # Remove rows with empty text
        initial_len = len(df)
        df = df[df['text'].str.strip() != ''].copy()
        
        if len(df) < initial_len:
            print(f"⚠️  Removed {initial_len - len(df)} articles with empty text")
        
        # Rename columns for clarity
        df = df.rename(columns={
            'source_name': 'source',
            'article_url': 'url',
            'company_symbol': 'symbol'
        })
        
        # Select and reorder columns
        df = df[[
            'text',
            'source',
            'published_date',
            'url',
            'relevance_score',
            'symbol'
        ]]
        
        # Create export directory if it doesn't exist
        export_dir = Path(export_path).parent
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(export_path, index=False)
        
        print(f"\n✅ Exported to: {export_path}")
        print(f"   Total articles: {len(df)}")
        print(f"   Date range: {df['published_date'].min()} to {df['published_date'].max()}")
        print(f"   Avg relevance score: {df['relevance_score'].mean():.3f}")
        print(f"\nSource distribution:")
        print(df['source'].value_counts())
        
        return df
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {str(e)}")
        raise
    except Exception as e:
        print(f"❌ Error during export: {str(e)}")
        raise


def main():
    """CLI entrypoint for news cache export."""
    parser = argparse.ArgumentParser(
        description="Export news articles from SQLite cache to CSV"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Path to the SQLite database file"
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default="./data/news_export.csv",
        help="Output CSV file path (default: ./data/news_export.csv)"
    )
    parser.add_argument(
        "--min_relevance",
        type=float,
        default=0.0,
        help="Minimum relevance score filter (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("News Cache Export")
    print("="*70)
    print(f"Database: {args.db_path}")
    print(f"Export path: {args.export_path}")
    print(f"Min relevance: {args.min_relevance}")
    print("="*70 + "\n")
    
    try:
        # Export news cache
        df = export_news_cache_to_csv(
            db_path=args.db_path,
            export_path=args.export_path,
            min_relevance=args.min_relevance
        )
        
        print("\n✅ Export complete!")
        
    except Exception as e:
        print(f"\n❌ Export failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()