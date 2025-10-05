# finbert_preprocessor.py - FinBERT Ready Output (Fixed)
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinBERTPreprocessor:
    """
    Preprocessor for preparing filtered news articles for FinBERT sentiment analysis.
    Handles context-aware chunking and API-ready batch formatting.
    """
    
    def __init__(self, max_tokens_per_chunk: int = 400, chunk_overlap: int = 50, batch_size: int = 10):
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
    def prepare_for_finbert(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point for preparing articles for FinBERT analysis.
        
        Args:
            articles: List of filtered, deduplicated, and sorted articles from content_filter.py
        
        Returns:
            Flat list of preprocessed articles ready for FinBERT client
        """
        if not articles:
            logger.warning("No articles provided for FinBERT preprocessing")
            return []
        
        logger.info(f"Starting FinBERT preprocessing for {len(articles)} articles")
        
        try:
            # Select top articles
            top_articles = self._select_top_articles(articles)
            logger.info(f"Selected {len(top_articles)} articles for processing")
            
            # Clean and prepare article text
            cleaned_articles = self._clean_articles(top_articles)
            logger.info(f"Cleaning complete: {len(cleaned_articles)} articles retained, {len(top_articles) - len(cleaned_articles)} skipped for empty text")
            
            # Chunk long articles with context awareness
            chunked_articles = self.chunk_long_articles(cleaned_articles)
            logger.info(f"Created {len(chunked_articles)} chunks from {len(cleaned_articles)} cleaned articles")
            
            # Convert to flat list format for FinBERT client
            finbert_ready = self._convert_to_finbert_format(chunked_articles)
            
            logger.info(f"Prepared {len(finbert_ready)} text entries for FinBERT")
            return finbert_ready
            
        except Exception as e:
            logger.error(f"Error in prepare_for_finbert: {str(e)}")
            return []
    
    def _select_top_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select top 15 articles prioritizing relevance and recency."""
        try:
            # Articles are already sorted by recency from content_filter
            # Take top 15 for optimal API usage and processing time
            return articles[:15]
        except Exception as e:
            logger.error(f"Error selecting top articles: {str(e)}")
            return []
    
    def _clean_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean article text for better FinBERT processing."""
        cleaned = []
        
        for article in articles:
            try:
                cleaned_article = article.copy()
                
                # Clean title - handle None/empty values
                title = article.get('title', '') or ''
                cleaned_article['title'] = self._clean_text(str(title))
                
                # Clean content - handle None/empty values
                content = article.get('description', '') or ''
                cleaned_article['content'] = self._clean_text(str(content))

                # Normalize/propagate metadata expected downstream
                # Map 'published' -> 'published_date' for consistent usage
                if 'published_date' not in cleaned_article:
                    cleaned_article['published_date'] = article.get('published', '') or article.get('published_date', '')
                # Ensure source present
                cleaned_article['source'] = article.get('source', cleaned_article.get('source', 'unknown')) or 'unknown'
                # Keep link if available for unique identification
                cleaned_article['link'] = article.get('link', cleaned_article.get('link', ''))
                
                # Combine title and content for processing
                combined_text = ""
                if cleaned_article['title']:
                    combined_text += cleaned_article['title'] + ". "
                if cleaned_article['content']:
                    combined_text += cleaned_article['content']
                
                cleaned_article['full_text'] = combined_text.strip()
                
                # Only add if there's actual content
                if cleaned_article['full_text']:
                    cleaned.append(cleaned_article)
                else:
                    logger.warning(f"Skipping article with no content: {article.get('title', 'Unknown')}")
                    
            except Exception as e:
                logger.error(f"Error cleaning article: {str(e)}")
                continue
        
        return cleaned
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for FinBERT processing."""
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n+', ' ', text)
            
            # Normalize financial symbols and currency
            text = re.sub(r'₹\s*(\d+)', r'INR \1', text)  # Indian Rupee
            text = re.sub(r'\$\s*(\d+)', r'USD \1', text)  # US Dollar
            
            # Clean up quotes and special characters
            text = re.sub(r'["""]', '"', text)
            text = re.sub(r"['']", "'", text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text  # Return original text if cleaning fails
    
    def chunk_long_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Context-aware chunking of articles for FinBERT token limits.
        Preserves sentence boundaries and maintains financial context.
        """
        chunks = []
        
        for article in articles:
            try:
                full_text = article.get('full_text', '')
                if not full_text:
                    logger.debug("Skipping chunking for article with empty full_text")
                    continue
                
                # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
                estimated_tokens = len(full_text) / 4
                
                if estimated_tokens <= self.max_tokens_per_chunk:
                    # Article is short enough, use as single chunk
                    chunk = {
                        'text': full_text,
                        'source': str(article.get('source', 'unknown')),
                        'published_date': article.get('published_date'),
                        'original_title': str(article.get('title', '')),
                        'link': article.get('link', ''),
                        'chunk_id': 0,
                        'total_chunks': 1
                    }
                    chunks.append(chunk)
                else:
                    # Article needs chunking
                    article_chunks = self._create_context_aware_chunks(full_text, article)
                    chunks.extend(article_chunks)
                    
            except Exception as e:
                logger.error(f"Error chunking article: {str(e)}")
                continue
        
        return chunks
    
    def _create_context_aware_chunks(self, text: str, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create overlapping chunks while preserving sentence boundaries."""
        try:
            sentences = self._split_into_sentences(text)
            if not sentences:
                return []
            
            chunks = []
            chunk_id = 0
            
            current_chunk = ""
            current_chunk_tokens = 0
            
            i = 0
            while i < len(sentences):
                sentence = sentences[i]
                sentence_tokens = len(sentence) / 4  # Rough token estimate
                
                # Check if adding this sentence would exceed token limit
                if current_chunk_tokens + sentence_tokens > self.max_tokens_per_chunk and current_chunk:
                    # Create chunk from current content
                    chunk = {
                        'text': current_chunk.strip(),
                        'source': str(article.get('source', 'unknown')),
                        'published_date': article.get('published_date'),
                        'original_title': str(article.get('title', '')),
                        'link': article.get('link', ''),
                        'chunk_id': chunk_id,
                        'total_chunks': 0  # Will be updated after all chunks are created
                    }
                    chunks.append(chunk)
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                    current_chunk_tokens = len(current_chunk) / 4
                else:
                    # Add sentence to current chunk
                    current_chunk += (" " + sentence) if current_chunk else sentence
                    current_chunk_tokens += sentence_tokens
                
                i += 1
            
            # Add final chunk if it has content
            if current_chunk.strip():
                chunk = {
                    'text': current_chunk.strip(),
                    'source': str(article.get('source', 'unknown')),
                    'published_date': article.get('published_date'),
                    'original_title': str(article.get('title', '')),
                    'link': article.get('link', ''),
                    'chunk_id': chunk_id,
                    'total_chunks': 0
                }
                chunks.append(chunk)
            
            # Update total_chunks count for all chunks of this article
            total_chunks = len(chunks)
            article_title = str(article.get('title', ''))
            for chunk in chunks:
                if chunk['original_title'] == article_title:
                    chunk['total_chunks'] = total_chunks
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating context-aware chunks: {str(e)}")
            return []
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling financial context."""
        try:
            if not text:
                return []
            
            # Basic sentence splitting that preserves financial abbreviations
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            
            # Clean up sentences
            cleaned_sentences = []
            for sentence in sentences:
                if sentence:  # Check if sentence exists
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 10:  # Filter out very short fragments
                        cleaned_sentences.append(sentence)
            
            return cleaned_sentences
            
        except Exception as e:
            logger.error(f"Error splitting sentences: {str(e)}")
            return [text]  # Return original text as single sentence if splitting fails
    
    def _get_overlap_text(self, chunk_text: str) -> str:
        """Get overlap text from the end of current chunk."""
        try:
            if not chunk_text:
                return ""
            
            words = chunk_text.split()
            overlap_words = max(1, int(self.chunk_overlap / 4))  # Ensure at least 1 word
            
            if len(words) > overlap_words:
                return " ".join(words[-overlap_words:])
            return ""
            
        except Exception as e:
            logger.error(f"Error getting overlap text: {str(e)}")
            return ""
    
    def _convert_to_finbert_format(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert chunked articles to flat list format expected by FinBERT client.
        
        Args:
            chunks: List of chunked articles
        
        Returns:
            Flat list with format: [{'id': '...', 'text': '...', 'metadata': {...}}, ...]
        """
        finbert_ready = []
        
        for chunk in chunks:
            try:
                # Safely construct chunk_id
                source = str(chunk.get('source', 'unknown'))
                title = str(chunk.get('original_title', 'untitled'))
                chunk_id = chunk.get('chunk_id', 0)
                
                # Clean strings for use in ID
                source_clean = re.sub(r'[^\w\-_]', '_', source)
                title_clean = re.sub(r'[^\w\-_]', '_', title)[:50]
                
                item = {
                    'id': f"{source_clean}_{title_clean}_{chunk_id}",
                    'text': str(chunk.get('text', '')),
                    'metadata': {
                        'source': source,
                        'published_date': chunk.get('published_date'),
                        'title': title,
                        'chunk_position': f"{chunk_id + 1}/{chunk.get('total_chunks', 1)}",
                        'url': chunk.get('link', '')
                    }
                }
                
                finbert_ready.append(item)
                
            except Exception as e:
                logger.error(f"Error converting chunk to FinBERT format: {str(e)}")
                continue
        
        return finbert_ready

# Usage example and integration point
if __name__ == "__main__":
    try:
        # Example usage with mock data
        preprocessor = FinBERTPreprocessor()
        
        # Mock articles from content_filter.py output
        sample_articles = [
            {
                'title': 'Reliance Industries Reports Strong Q3 Results',
                'description': 'Reliance Industries Limited announced robust quarterly results with revenue growth of 15% year-over-year. The company reported consolidated revenue of INR 2.3 lakh crore for the quarter ending December 2024.',
                'source': 'Economic Times',
                'published': '2024-01-15',
                'relevance_score': 0.95
            },
            {
                'title': None,  # Test None handling
                'description': '',  # Test empty content
                'source': 'Test Source',
                'published': '2024-01-16',
                'relevance_score': 0.8
            }
        ]
        
        # Process articles
        result = preprocessor.prepare_for_finbert(sample_articles)
        
        print(f"\nProcessing complete:")
        print(f"- Total entries prepared: {len(result)}")
        
        # Display first entry as example
        if result:
            print(f"\nExample entry:")
            print(f"- ID: {result[0]['id']}")
            print(f"- Text length: {len(result[0]['text'])} chars")
            print(f"- Metadata: {result[0]['metadata']}")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        logger.error(f"Main execution error: {str(e)}")
        import traceback
        traceback.print_exc()