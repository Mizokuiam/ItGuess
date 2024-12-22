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
            if newsapi_articles and isinstance(newsapi_articles, list):
                for article in newsapi_articles:
                    try:
                        # Ensure we have required fields
                        if not article.get('title') or not article.get('url'):
                            continue
                            
                        # Get the date, defaulting to now if not available
                        date = datetime.now()
                        if article.get('publishedAt'):
                            try:
                                date = pd.to_datetime(article['publishedAt'])
                            except:
                                pass
                        
                        all_articles.append({
                            'title': article['title'],
                            'summary': article.get('description', ''),
                            'url': article['url'],
                            'date': date.isoformat(),
                            'source': article.get('source', {}).get('name', 'NewsAPI'),
                            'sentiment': None,
                            'sentiment_category': None
                        })
                    except Exception as e:
                        print(f"Error processing NewsAPI article: {str(e)}")
                        continue
            
            # Get Finnhub articles
            finnhub_articles = self._get_finnhub_articles(symbol)
            if finnhub_articles and isinstance(finnhub_articles, list):
                for article in finnhub_articles:
                    try:
                        # Ensure we have required fields
                        if not article.get('headline') or not article.get('url'):
                            continue
                            
                        # Get the date, defaulting to now if not available
                        date = datetime.now()
                        if article.get('datetime'):
                            try:
                                date = datetime.fromtimestamp(int(article['datetime']))
                            except:
                                pass
                        
                        all_articles.append({
                            'title': article['headline'],
                            'summary': article.get('summary', ''),
                            'url': article['url'],
                            'date': date.isoformat(),
                            'source': article.get('source', 'Finnhub'),
                            'sentiment': None,
                            'sentiment_category': None
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
                
                # Reset index
                df = df.reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error in get_company_news: {str(e)}")
            return pd.DataFrame()

    def _get_newsapi_articles(self, company_name):
        """Get recent articles from NewsAPI"""
        try:
            # Get news from the last 7 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            response = self.newsapi.get_everything(
                q=company_name,
                language='en',
                sort_by='publishedAt',
                from_param=start_date.date().isoformat(),
                to=end_date.date().isoformat(),
                page_size=20
            )
            
            articles = response.get('articles', [])
            return articles if articles else []
            
        except Exception as e:
            print(f"Error fetching NewsAPI articles: {str(e)}")
            return []
    
    def _get_finnhub_articles(self, symbol):
        """Get recent articles from Finnhub"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            news = self.finnhub_client.company_news(
                symbol, 
                _from=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d')
            )
            
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
