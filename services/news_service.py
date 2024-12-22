import os
from newsapi import NewsApiClient
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timedelta
import finnhub

class NewsService:
    def __init__(self):
        self.newsapi = NewsApiClient(api_key='168ee0f418f14fcf8f88cdd1b6bf5963')
        self.finnhub_client = finnhub.Client(api_key='ctjutipr01quipmv6v3gctjutipr01quipmv6v40')

    def get_company_news(self, symbol, company_name):
        """Get news articles for a company from multiple sources"""
        try:
            all_articles = []
            
            # Get NewsAPI articles
            newsapi_articles = self._get_newsapi_articles(company_name)
            for article in newsapi_articles:
                try:
                    all_articles.append({
                        'title': article.get('title', ''),
                        'summary': article.get('description', ''),
                        'url': article.get('url', ''),
                        'date': article.get('publishedAt', datetime.now().isoformat()),
                        'source': article.get('source', {}).get('name', 'NewsAPI'),
                        'sentiment': None,  # Will be calculated later
                        'sentiment_category': None  # Will be calculated later
                    })
                except Exception as e:
                    print(f"Error processing NewsAPI article: {str(e)}")
                    continue
            
            # Get Finnhub articles
            finnhub_articles = self._get_finnhub_articles(symbol)
            for article in finnhub_articles:
                try:
                    timestamp = article.get('datetime', 0)
                    date = datetime.fromtimestamp(timestamp).isoformat() if timestamp else datetime.now().isoformat()
                    
                    all_articles.append({
                        'title': article.get('headline', ''),
                        'summary': article.get('summary', ''),
                        'url': article.get('url', ''),
                        'date': date,
                        'source': article.get('source', 'Finnhub'),
                        'sentiment': None,  # Will be calculated later
                        'sentiment_category': None  # Will be calculated later
                    })
                except Exception as e:
                    print(f"Error processing Finnhub article: {str(e)}")
                    continue
            
            # Create DataFrame
            df = pd.DataFrame(all_articles)
            
            if not df.empty:
                # Calculate sentiment for all articles
                df['sentiment'] = df.apply(
                    lambda row: self._calculate_sentiment(row['title'], row['summary']), 
                    axis=1
                )
                
                # Add sentiment categories
                df['sentiment_category'] = df['sentiment'].apply(self._get_sentiment_category)
                
                # Convert dates to datetime
                df['date'] = pd.to_datetime(df['date'])
                
                # Sort by date
                df = df.sort_values('date', ascending=False)
            
            return df
            
        except Exception as e:
            print(f"Error in get_company_news: {str(e)}")
            return pd.DataFrame()
    
    def _get_newsapi_articles(self, company_name):
        """Get articles from NewsAPI"""
        try:
            response = self.newsapi.get_everything(
                q=company_name,
                language='en',
                sort_by='publishedAt',
                from_param=(datetime.now() - timedelta(days=30)).date().isoformat(),
                to=datetime.now().date().isoformat()
            )
            return response.get('articles', [])
        except Exception as e:
            print(f"Error fetching NewsAPI articles: {str(e)}")
            return []
    
    def _get_finnhub_articles(self, symbol):
        """Get articles from Finnhub"""
        try:
            end_date = int(datetime.now().timestamp())
            start_date = int((datetime.now() - timedelta(days=30)).timestamp())
            
            news = self.finnhub_client.company_news(symbol, _from=start_date, to=end_date)
            return news if news else []
        except Exception as e:
            print(f"Error fetching Finnhub news: {str(e)}")
            return []
    
    def _calculate_sentiment(self, title, summary):
        """Calculate sentiment score for an article"""
        try:
            text = f"{title} {summary}"
            if not text.strip():
                return 0
            
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception as e:
            print(f"Error calculating sentiment: {str(e)}")
            return 0
    
    def _get_sentiment_category(self, sentiment):
        """Get sentiment category based on polarity score"""
        if sentiment > 0.1:
            return "Positive"
        elif sentiment < -0.1:
            return "Negative"
        return "Neutral"
    
    def get_peer_comparison(self, symbol):
        """Get peer comparison data"""
        try:
            # Get peer symbols
            peers = self.finnhub_client.company_peers(symbol)
            
            if not peers:
                return None
                
            # Limit to top 5 peers
            peers = peers[:5]
            
            # Get basic financials for all peers
            peer_data = []
            for peer in peers:
                metrics = self.finnhub_client.company_basic_financials(peer, 'all')
                if metrics and 'metric' in metrics:
                    peer_data.append({
                        'symbol': peer,
                        'pe_ratio': metrics['metric'].get('peNormalizedAnnual', None),
                        'price_to_sales': metrics['metric'].get('psAnnual', None),
                        'price_to_book': metrics['metric'].get('pbAnnual', None),
                        'debt_to_equity': metrics['metric'].get('totalDebtToEquityQuarterly', None)
                    })
            
            return pd.DataFrame(peer_data)
        except Exception as e:
            print(f"Error in peer comparison: {str(e)}")
            return None
