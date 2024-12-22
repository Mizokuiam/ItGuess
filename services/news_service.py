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
        """Get news articles for a company"""
        try:
            # Get news from both APIs
            news_api_articles = self._get_newsapi_articles(company_name)
            finnhub_articles = self._get_finnhub_articles(symbol)
            
            # Combine and process articles
            all_articles = news_api_articles + finnhub_articles
            return self._process_articles(all_articles)
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return None
            
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
        except:
            return []
            
    def _get_finnhub_articles(self, symbol):
        """Get articles from Finnhub"""
        try:
            end_date = int(datetime.now().timestamp())
            start_date = int((datetime.now() - timedelta(days=30)).timestamp())
            
            news = self.finnhub_client.company_news(symbol, _from=start_date, to=end_date)
            
            # Convert Finnhub response to match NewsAPI format
            formatted_news = []
            for article in news:
                formatted_news.append({
                    'title': article.get('headline', ''),
                    'summary': article.get('summary', ''),
                    'url': article.get('url', ''),
                    'publishedAt': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                    'source': {'name': article.get('source', 'Finnhub')}
                })
            return formatted_news
        except Exception as e:
            print(f"Error fetching Finnhub news: {str(e)}")
            return []
            
    def _process_articles(self, articles):
        """Process and analyze sentiment of articles"""
        processed_articles = []
        
        for article in articles:
            try:
                # Extract text for sentiment analysis
                title = article.get('title', '')
                summary = article.get('summary', '') or article.get('description', '')
                
                if not title and not summary:
                    continue
                
                # Combine text for sentiment analysis
                text = f"{title} {summary}"
                
                # Perform sentiment analysis
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity
                
                # Categorize sentiment
                if sentiment > 0.1:
                    sentiment_category = "Positive"
                elif sentiment < -0.1:
                    sentiment_category = "Negative"
                else:
                    sentiment_category = "Neutral"
                
                processed_articles.append({
                    'title': title,
                    'summary': summary,
                    'url': article.get('url', ''),
                    'date': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'sentiment': sentiment,
                    'sentiment_category': sentiment_category
                })
            except Exception as e:
                print(f"Error processing article: {str(e)}")
                continue
        
        # Create DataFrame and sort by date
        df = pd.DataFrame(processed_articles)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
        
        return df
    
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
