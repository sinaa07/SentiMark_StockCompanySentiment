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
        
    def prepare_for_finbert(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main entry point for preparing articles for FinBERT analysis.
        
        Args:
            articles: List of filtered, deduplicated, and sorted articles from content_filter.py
                     Each article should have: title, content, source, published_date, relevance_score
        
        Returns:
            Dict containing batched input ready for FinBERT API calls
        """
        if not articles:
            logger.warning("No articles provided for FinBERT preprocessing")
            return {"batches": [], "total_chunks": 0, "articles_processed": 0}
        
        logger.info(f"Starting FinBERT preprocessing for {len(articles)} articles")
        
        try:
            # Select top articles (already sorted by recency from content_filter)
            top_articles = self._select_top_articles(articles)
            logger.info(f"Selected {len(top_articles)} articles for processing")
            
            # Clean and prepare article text
            cleaned_articles = self._clean_articles(top_articles)
            
            # Chunk long articles with context awareness
            chunked_articles = self.chunk_long_articles(cleaned_articles)
            logger.info(f"Created {len(chunked_articles)} chunks from {len(top_articles)} articles")
            
            # Create batched input for API calls
            batched_input = self.create_finbert_input_batch(chunked_articles)
            
            logger.info(f"Prepared {len(batched_input['batches'])} batches for FinBERT API")
            return batched_input
            
        except Exception as e:
            logger.error(f"Error in prepare_for_finbert: {str(e)}")
            return {"batches": [], "total_chunks": 0, "articles_processed": 0, "error": str(e)}
    
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
                content = article.get('content', '') or ''
                cleaned_article['content'] = self._clean_text(str(content))
                
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
    
    def create_finbert_input_batch(self, processed_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create batched input format for FinBERT API calls.
        Groups chunks into optimal batch sizes for API efficiency.
        """
        if not processed_articles:
            return {"batches": [], "total_chunks": 0, "articles_processed": 0}
        
        try:
            batches = []
            current_batch = []
            
            for chunk in processed_articles:
                try:
                    # Safely construct chunk_id
                    source = str(chunk.get('source', 'unknown'))
                    title = str(chunk.get('original_title', 'untitled'))
                    chunk_id = chunk.get('chunk_id', 0)
                    
                    # Clean strings for use in ID (remove special characters)
                    source_clean = re.sub(r'[^\w\-_]', '_', source)
                    title_clean = re.sub(r'[^\w\-_]', '_', title)[:50]  # Limit length
                    
                    chunk_item = {
                        'text': str(chunk.get('text', '')),
                        'chunk_id': f"{source_clean}_{title_clean}_{chunk_id}",
                        'metadata': {
                            'source': source,
                            'published_date': chunk.get('published_date'),
                            'original_title': title,
                            'chunk_position': f"{chunk_id + 1}/{chunk.get('total_chunks', 1)}"
                        }
                    }
                    
                    current_batch.append(chunk_item)
                    
                    # Create batch when batch_size is reached
                    if len(current_batch) >= self.batch_size:
                        batches.append({
                            'batch_id': len(batches),
                            'texts': [item['text'] for item in current_batch],
                            'chunk_metadata': current_batch.copy()
                        })
                        current_batch = []
                        
                except Exception as e:
                    logger.error(f"Error processing chunk for batch: {str(e)}")
                    continue
            
            # Add remaining chunks as final batch
            if current_batch:
                batches.append({
                    'batch_id': len(batches),
                    'texts': [item['text'] for item in current_batch],
                    'chunk_metadata': current_batch.copy()
                })
            
            # Count unique articles safely
            unique_titles = set()
            for chunk in processed_articles:
                title = chunk.get('original_title')
                if title:
                    unique_titles.add(str(title))
            
            result = {
                'batches': batches,
                'total_chunks': len(processed_articles),
                'articles_processed': len(unique_titles),
                'processing_timestamp': datetime.now().isoformat(),
                'batch_config': {
                    'max_tokens_per_chunk': self.max_tokens_per_chunk,
                    'chunk_overlap': self.chunk_overlap,
                    'batch_size': self.batch_size
                }
            }
            
            logger.info(f"Created {len(batches)} batches with {result['total_chunks']} total chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error creating batched input: {str(e)}")
            return {"batches": [], "total_chunks": 0, "articles_processed": 0, "error": str(e)}


# Usage example and integration point
if __name__ == "__main__":
    try:
        # Example usage with mock data
        preprocessor = FinBERTPreprocessor()
        
        # Mock articles from content_filter.py output
        sample_articles = [
            {
                'title': 'Reliance Industries Reports Strong Q3 Results',
                'content': 'Reliance Industries Limited announced robust quarterly results with revenue growth of 15% year-over-year. The company reported consolidated revenue of INR 2.3 lakh crore for the quarter ending December 2024.',
                'source': 'Economic Times',
                'published_date': '2024-01-15',
                'relevance_score': 0.95
            },
            {
                'title': None,  # Test None handling
                'content': '',  # Test empty content
                'source': 'Test Source',
                'published_date': '2024-01-16',
                'relevance_score': 0.8
            }
        ]
        
        # Process articles
        result = preprocessor.prepare_for_finbert(sample_articles)
        
        print(f"Processing complete:")
        print(f"- Articles processed: {result['articles_processed']}")
        print(f"- Total chunks created: {result['total_chunks']}")
        print(f"- Batches for API: {len(result['batches'])}")
        
        if 'error' in result:
            print(f"- Error encountered: {result['error']}")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        logger.error(f"Main execution error: {str(e)}")