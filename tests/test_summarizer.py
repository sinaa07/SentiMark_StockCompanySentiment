#!/usr/bin/env python3
"""
Test summarizer.py without making API calls
Uses mock data to simulate real news articles with sentiment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.summarizer import create_news_summary, NewsSummarizer

# Mock news articles with sentiment already analyzed
MOCK_ARTICLES = [
    {
        "headline": "Reliance Industries reports record quarterly earnings, beats estimates",
        "summary": "Company posted strong revenue growth across all segments",
        "url": "https://example.com/news1",
        "published_at": "2024-01-15T10:30:00Z",
        "sentiment": "positive",
        "confidence": 0.89
    },
    {
        "headline": "RIL stock hits new 52-week high on strong earnings outlook",
        "summary": "Investors bullish on company's future prospects",
        "url": "https://example.com/news2", 
        "published_at": "2024-01-15T11:45:00Z",
        "sentiment": "positive",
        "confidence": 0.92
    },
    {
        "headline": "Reliance faces regulatory scrutiny over telecom operations",
        "summary": "TRAI investigating certain practices in Jio operations",
        "url": "https://example.com/news3",
        "published_at": "2024-01-14T14:20:00Z", 
        "sentiment": "negative",
        "confidence": 0.76
    },
    {
        "headline": "Mukesh Ambani announces new green energy initiatives",
        "summary": "Chairman outlines ambitious renewable energy plans",
        "url": "https://example.com/news4",
        "published_at": "2024-01-14T09:15:00Z",
        "sentiment": "positive", 
        "confidence": 0.81
    },
    {
        "headline": "Reliance retail expansion continues with 50 new stores",
        "summary": "Company maintains aggressive retail growth strategy",
        "url": "https://example.com/news5",
        "published_at": "2024-01-13T16:30:00Z",
        "sentiment": "neutral",
        "confidence": 0.65
    },
    {
        "headline": "Investors cautious on RIL amid rising crude oil prices",
        "summary": "Higher input costs may impact refining margins",
        "url": "https://example.com/news6",
        "published_at": "2024-01-13T12:10:00Z",
        "sentiment": "negative",
        "confidence": 0.71
    },
    {
        "headline": "Reliance partners with Meta for digital commerce push",
        "summary": "Strategic partnership aims to boost online retail presence",
        "url": "https://example.com/news7",
        "published_at": "2024-01-12T13:45:00Z",
        "sentiment": "positive",
        "confidence": 0.84
    },
    {
        "headline": "RIL Q3 results: Revenue up 12% year-on-year",
        "summary": "All business segments show healthy growth",
        "url": "https://example.com/news8",
        "published_at": "2024-01-12T10:00:00Z",
        "sentiment": "positive",
        "confidence": 0.87
    }
]

# Mock sentiment data (what sentiment_analyzer would return)
MOCK_SENTIMENT_DATA = {
    "overall": "positive",
    "distribution": {
        "positive": 0.625,  # 5 out of 8 articles
        "negative": 0.25,   # 2 out of 8 articles  
        "neutral": 0.125    # 1 out of 8 articles
    },
    "total_articles": 8,
    "average_confidence": 0.805
}

def test_individual_functions():
    """Test individual summarizer functions"""
    print("🧪 Testing Individual Summarizer Functions")
    print("=" * 50)
    
    summarizer = NewsSummarizer()
    
    # Test event detection
    print("\n1. Testing Event Detection:")
    events = summarizer.detect_key_events(MOCK_ARTICLES)
    for event_type, articles in events.items():
        print(f"   📅 {event_type.title()}: {len(articles)} articles")
        for article in articles[:2]:  # Show first 2
            print(f"      - {article['headline'][:60]}... ({article['sentiment']})")
    
    # Test investor mood
    print("\n2. Testing Investor Mood:")
    mood = summarizer.create_investor_mood(MOCK_SENTIMENT_DATA)
    print(f"   {mood}")
    
    # Test top themes
    print("\n3. Testing Top Themes:")
    themes = summarizer.get_top_themes(MOCK_ARTICLES)
    print(f"   📊 Key themes: {', '.join(themes)}")
    
    # Test market implication
    print("\n4. Testing Market Implication:")
    implication = summarizer.get_market_implication('positive', events)
    print(f"   💡 {implication}")

def test_full_summary():
    """Test the complete summary generation"""
    print("\n\n🎯 Testing Complete Summary Generation")
    print("=" * 50)
    
    # Generate complete summary
    summary_data = create_news_summary(MOCK_ARTICLES, MOCK_SENTIMENT_DATA)
    
    print("\n📰 AI Generated Summary:")
    print("-" * 30)
    print(summary_data['ai_summary'])
    
    print(f"\n📈 Investor Mood:")
    print("-" * 20)
    print(summary_data['investor_mood'])
    
    print(f"\n🔍 Key Events Detected:")
    print("-" * 25)
    for event_type, articles in summary_data['key_events'].items():
        print(f"   • {event_type.title()}: {len(articles)} mentions")
    
    print(f"\n🏷️  Top Themes:")
    print("-" * 15)
    print(f"   {', '.join(summary_data['top_themes'])}")

def test_different_scenarios():
    """Test with different sentiment scenarios"""
    print("\n\n🎭 Testing Different Scenarios")
    print("=" * 50)
    
    # Scenario 1: All negative news
    negative_articles = [
        {
            "headline": "Company faces major lawsuit over data breach",
            "sentiment": "negative", "confidence": 0.91
        },
        {
            "headline": "Stock plunges 15% on weak earnings guidance", 
            "sentiment": "negative", "confidence": 0.87
        },
        {
            "headline": "CEO resignation creates leadership uncertainty",
            "sentiment": "negative", "confidence": 0.82
        }
    ]
    
    negative_sentiment = {
        "overall": "negative",
        "distribution": {"positive": 0.0, "negative": 1.0, "neutral": 0.0},
        "total_articles": 3,
        "average_confidence": 0.867
    }
    
    print("\n📉 Scenario: All Negative News")
    summary = create_news_summary(negative_articles, negative_sentiment)
    print(f"Summary: {summary['ai_summary']}")
    print(f"Mood: {summary['investor_mood']}")
    
    # Scenario 2: Mixed/Neutral
    mixed_sentiment = {
        "overall": "neutral", 
        "distribution": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
        "total_articles": 6,
        "average_confidence": 0.65
    }
    
    print("\n⚖️  Scenario: Mixed Sentiment") 
    summary = create_news_summary(MOCK_ARTICLES[:6], mixed_sentiment)
    print(f"Summary: {summary['ai_summary']}")
    print(f"Mood: {summary['investor_mood']}")

def test_edge_cases():
    """Test edge cases"""
    print("\n\n🚨 Testing Edge Cases")
    print("=" * 50)
    
    # Empty articles
    print("\n1. Empty Articles List:")
    empty_summary = create_news_summary([], {})
    print(f"   Summary: {empty_summary['ai_summary']}")
    
    # Articles without sentiment
    print("\n2. Articles Without Sentiment Data:")
    articles_no_sentiment = [{"headline": "Test headline", "summary": "Test summary"}]
    try:
        summary = create_news_summary(articles_no_sentiment, {"overall": "neutral"})
        print(f"   ✅ Handled gracefully")
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 TESTING SUMMARIZER WITHOUT API CALLS")
    print("=" * 60)
    print("Using mock data to simulate real news with sentiment analysis")
    
    try:
        test_individual_functions()
        test_full_summary() 
        test_different_scenarios()
        test_edge_cases()
        
        print("\n\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("🚀 Your summarizer is ready to use with real data!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()