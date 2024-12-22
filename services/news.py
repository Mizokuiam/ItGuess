import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class NewsService:
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)  # Cache news for 15 minutes
        
    def get_news(self, symbol):
        """Get news for a given stock symbol"""
        try:
            # Check cache first
            if symbol in self.cache:
                last_update, news_df = self.cache[symbol]
                if datetime.now() - last_update < self.cache_duration:
                    return news_df
            
            # Fetch fresh news
            stock = yf.Ticker(symbol)
            news = stock.news
            
            if not news:
                return pd.DataFrame()
                
            # Convert news to DataFrame
            news_data = []
            for item in news[:10]:  # Limit to 10 most recent news
                # Basic sentiment - can be enhanced with NLP
                title = item.get('title', '')
                
                # Simple sentiment analysis based on keywords
                positive_words = ['up', 'rise', 'gain', 'positive', 'growth', 'surge', 'jump', 'boost']
                negative_words = ['down', 'fall', 'drop', 'negative', 'decline', 'plunge', 'cut', 'loss']
                
                title_lower = title.lower()
                positive_count = sum(1 for word in positive_words if word in title_lower)
                negative_count = sum(1 for word in negative_words if word in title_lower)
                
                if positive_count > negative_count:
                    sentiment = "Positive"
                    sentiment_color = "#28a745"  # Green
                elif negative_count > positive_count:
                    sentiment = "Negative"
                    sentiment_color = "#dc3545"  # Red
                else:
                    sentiment = "Neutral"
                    sentiment_color = "#6c757d"  # Gray
                
                # Format date
                pub_date = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                formatted_date = pub_date.strftime('%Y-%m-%d %H:%M')
                
                news_data.append({
                    'title': title,
                    'summary': item.get('summary', '')[:200] + '...',  # Truncate long summaries
                    'date': formatted_date,
                    'sentiment': sentiment,
                    'sentiment_color': sentiment_color,
                    'url': item.get('link', '')
                })
            
            # Create DataFrame
            news_df = pd.DataFrame(news_data)
            
            # Update cache
            self.cache[symbol] = (datetime.now(), news_df)
            
            return news_df
            
        except Exception as e:
            print(f"Error fetching news for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()  # Return empty DataFrame on error
            
    def clear_cache(self):
        """Clear the news cache"""
        self.cache = {}
