import argparse
import asyncio
import json
import logging
import os
from dotenv import load_dotenv

from user_input_processor import UserInputProcessor
from news_collector import NewsCollector
from content_filter import ContentFilter
from finbert_preprocessor import FinBERTPreprocessor
from finbert_client import FinBERTClient


class SentimentPipeline:
    """End-to-end pipeline: user input -> news -> sentiment"""

    def __init__(self, db_path="data/nse_stocks.db", hf_api_token: str = ""):
        self.user_input = UserInputProcessor(db_path=db_path)
        self.news_collector = NewsCollector()
        self.filter = ContentFilter()
        self.preprocessor = FinBERTPreprocessor()
        self.client = FinBERTClient(api_token=hf_api_token)

    async def run(self, user_query: str):
        try:
            company_data = self.user_input.handle_user_selection(user_query)
            if company_data.get("error"):
                return {"stage": "user_input", "result": company_data}

            raw_articles = self.news_collector.collect_company_news(company_data)
            if not raw_articles:
                return {"stage": "news_collection", "error": "No articles found"}

            filtered = self.filter.filter_company_articles(raw_articles, company_data)
            if not filtered:
                return {"stage": "content_filter", "error": "No relevant articles after filtering"}

            preprocessed = self.preprocessor.prepare_for_finbert(filtered)
            sentiment_result = await self.client.analyze_company_sentiment(preprocessed, company_data)

            return {
                "stage": "complete",
                "company": company_data,
                "sentiment": sentiment_result,
            }

        except Exception as e:
            logging.exception("Pipeline execution failed")
            return {"stage": "pipeline", "error": str(e)}


class PipelineCLI:
    """CLI runner for the SentimentPipeline"""

    def __init__(self):
        load_dotenv()
        self.hf_api_token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.hf_api_token:
            raise ValueError("❌ HUGGINGFACE_TOKEN not found in .env file")

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Run sentiment pipeline for one or more company symbols")
        parser.add_argument("symbols", nargs="+", help="Company symbols (e.g., RELIANCE, TCS, INFY)")
        parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
        return parser.parse_args()

    def run(self):
        args = self.parse_args()

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        results = {}
        pipeline = SentimentPipeline(hf_api_token=self.hf_api_token)

        for symbol in args.symbols:
            logging.info(f"Running pipeline for {symbol}...")
            result = asyncio.run(pipeline.run(symbol))
            results[symbol] = result

        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    PipelineCLI().run()
