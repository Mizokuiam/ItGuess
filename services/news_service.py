import os
import finnhub
from newsapi import NewsApiClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NewsService:
    def __init__(self):
        # Get API keys from environment variables
        news_api_key = os.getenv('NEWS_API_KEY')
        finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        
        # Validate API keys
        if not news_api_key or not finnhub_api_key:
            raise ValueError("Missing required API keys. Please set NEWS_API_KEY and FINNHUB_API_KEY in your environment variables.")
        
        # Initialize clients
        self.newsapi = NewsApiClient(api_key=news_api_key)
        self.finnhub_client = finnhub.Client(api_key=finnhub_api_key)

    def get_company_news(self, symbol):
        try:
            # Get news from both sources
            news_api_articles = self.get_news_api_articles(symbol)
            finnhub_articles = self.get_finnhub_articles(symbol)
            
            # Combine and return results
            return news_api_articles + finnhub_articles
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return []

    def get_news_api_articles(self, symbol):
        try:
            response = self.newsapi.get_everything(
                q=symbol,
                language='en',
                sort_by='publishedAt'
            )
            return response.get('articles', [])
        except Exception as e:
            print(f"Error fetching NewsAPI articles: {str(e)}")
            return []

    def get_finnhub_articles(self, symbol):
        try:
            response = self.finnhub_client.company_news(symbol)
            return response if response else []
        except Exception as e:
            print(f"Error fetching Finnhub articles: {str(e)}")
            return []
