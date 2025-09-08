import requests
import os
from typing import List, Dict
from dotenv import load_dotenv
import re
from collections import Counter

load_dotenv()

class NewsSummarizer:
    def __init__(self):
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        
        # Event keywords for detection
        self.event_keywords = {
            'earnings': ['earnings', 'quarterly', 'revenue', 'profit', 'loss', 'results'],
            'leadership': ['ceo', 'chief executive', 'chairman', 'director', 'appointed', 'resigned'],
            'merger': ['merger', 'acquisition', 'bought', 'deal', 'takeover'],
            'regulatory': ['regulatory', 'compliance', 'investigation', 'fine', 'penalty'],
            'partnership': ['partnership', 'collaboration', 'joint venture', 'alliance'],
            'product': ['launch', 'product', 'service', 'innovation', 'patent']
        }
    
    def generate_summary(self, articles: List[Dict], sentiment_data: Dict) -> str:
        """
        Generate AI-powered summary of news sentiment and key events
        """
        if not articles:
            return "No recent news available for analysis."
        
        # Detect key events
        events = self.detect_key_events(articles)
        
        # Get sentiment breakdown
        overall_sentiment = sentiment_data.get('overall', 'neutral')
        distribution = sentiment_data.get('distribution', {})
        total_articles = sentiment_data.get('total_articles', 0)
        
        # Calculate percentages
        positive_pct = int(distribution.get('positive', 0) * 100)
        negative_pct = int(distribution.get('negative', 0) * 100)
        neutral_pct = int(distribution.get('neutral', 0) * 100)
        
        # Generate summary based on sentiment and events
        summary_parts = []
        
        # Sentiment overview
        if overall_sentiment == 'positive':
            summary_parts.append(f"Market sentiment is currently **positive** with {positive_pct}% of recent coverage showing optimistic outlook.")
        elif overall_sentiment == 'negative':
            summary_parts.append(f"Market sentiment is **negative** with {negative_pct}% of coverage expressing concerns.")
        else:
            summary_parts.append(f"Market sentiment remains **neutral** with mixed coverage ({positive_pct}% positive, {negative_pct}% negative).")
        
        # Key events
        if events:
            event_text = self.format_events(events)
            summary_parts.append(event_text)
        
        # Market implications
        market_implication = self.get_market_implication(overall_sentiment, events)
        if market_implication:
            summary_parts.append(market_implication)
        
        return " ".join(summary_parts)
    
    def detect_key_events(self, articles: List[Dict]) -> Dict:
        """
        Detect significant events from news headlines
        """
        events = {category: [] for category in self.event_keywords.keys()}
        
        for article in articles:
            headline = article.get('headline', '').lower()
            summary = article.get('summary', '').lower()
            text = f"{headline} {summary}"
            
            for category, keywords in self.event_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        events[category].append({
                            'headline': article.get('headline', ''),
                            'sentiment': article.get('sentiment', 'neutral'),
                            'confidence': article.get('confidence', 0.0)
                        })
                        break
        
        # Filter out empty categories
        return {k: v for k, v in events.items() if v}
    
    def format_events(self, events: Dict) -> str:
        """
        Format detected events into readable text
        """
        event_descriptions = []
        
        priority_events = ['earnings', 'leadership', 'merger', 'regulatory']
        
        for event_type in priority_events:
            if event_type in events and events[event_type]:
                count = len(events[event_type])
                sentiments = [e['sentiment'] for e in events[event_type]]
                dominant_sentiment = Counter(sentiments).most_common(1)[0][0]
                
                if event_type == 'earnings':
                    if dominant_sentiment == 'positive':
                        event_descriptions.append("Strong quarterly results are driving positive investor sentiment")
                    else:
                        event_descriptions.append("Recent earnings reports show mixed to concerning results")
                
                elif event_type == 'leadership':
                    event_descriptions.append("Leadership changes are creating market uncertainty")
                
                elif event_type == 'merger':
                    event_descriptions.append("M&A activity is generating significant market interest")
                
                elif event_type == 'regulatory':
                    event_descriptions.append("Regulatory developments are impacting market perception")
        
        return ". ".join(event_descriptions) + "." if event_descriptions else ""
    
    def get_market_implication(self, sentiment: str, events: Dict) -> str:
        """
        Generate market implication based on sentiment and events
        """
        implications = {
            'positive': [
                "Investors appear optimistic about near-term prospects",
                "Market conditions favor continued positive momentum",
                "Strong fundamentals support current market sentiment"
            ],
            'negative': [
                "Caution advised as negative sentiment may impact stock performance",
                "Market headwinds suggest potential volatility ahead", 
                "Investors should monitor developments closely"
            ],
            'neutral': [
                "Market sentiment suggests a wait-and-see approach among investors",
                "Mixed signals indicate potential for movement in either direction",
                "Current conditions support a balanced investment perspective"
            ]
        }
        
        # Modify based on events
        if 'earnings' in events:
            if sentiment == 'positive':
                return "Strong earnings momentum likely to continue driving positive sentiment"
            else:
                return "Earnings concerns may weigh on stock performance in the near term"
        
        return implications.get(sentiment, [""])[0] if sentiment in implications else ""
    
    def create_investor_mood(self, sentiment_data: Dict) -> str:
        """
        Create a simple investor mood description
        """
        overall = sentiment_data.get('overall', 'neutral')
        confidence = sentiment_data.get('average_confidence', 0.0)
        
        mood_map = {
            'positive': {
                'high': "🟢 **Bullish** - Strong positive sentiment with high confidence",
                'medium': "🟢 **Optimistic** - Positive sentiment with moderate confidence", 
                'low': "🟡 **Cautiously Positive** - Positive but with lower confidence"
            },
            'negative': {
                'high': "🔴 **Bearish** - Strong negative sentiment with high confidence",
                'medium': "🔴 **Pessimistic** - Negative sentiment with moderate confidence",
                'low': "🟡 **Cautiously Negative** - Negative but with lower confidence"
            },
            'neutral': {
                'high': "⚪ **Neutral** - Balanced sentiment with clear conviction",
                'medium': "⚪ **Mixed** - Neutral sentiment with moderate confidence",
                'low': "⚪ **Uncertain** - Neutral sentiment with low confidence"
            }
        }
        
        # Determine confidence level
        if confidence > 0.7:
            conf_level = 'high'
        elif confidence > 0.5:
            conf_level = 'medium'
        else:
            conf_level = 'low'
        
        return mood_map.get(overall, {}).get(conf_level, "⚪ **Neutral** - Unable to determine clear sentiment")
    
    def get_top_themes(self, articles: List[Dict], limit: int = 3) -> List[str]:
        """
        Extract top themes/topics from headlines
        """
        # Simple keyword extraction
        all_text = ' '.join([article.get('headline', '') for article in articles]).lower()
        
        # Remove common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'stock', 'shares', 'company'}
        
        # Extract meaningful words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)
        words = [word for word in words if word not in common_words]
        
        # Get most common themes
        word_counts = Counter(words)
        top_themes = [word.title() for word, count in word_counts.most_common(limit) if count > 1]
        
        return top_themes

# Utility functions for easy integration
def create_news_summary(articles: List[Dict], sentiment_data: Dict) -> Dict:
    """
    Create comprehensive news summary with all components
    """
    summarizer = NewsSummarizer()
    
    return {
        'ai_summary': summarizer.generate_summary(articles, sentiment_data),
        'investor_mood': summarizer.create_investor_mood(sentiment_data), 
        'key_events': summarizer.detect_key_events(articles),
        'top_themes': summarizer.get_top_themes(articles)
    }