import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def fetch_stock_data(self, symbol, period='2y'):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators for the dataset"""
        try:
            indicators = pd.DataFrame(index=df.index)
            
            # Simple Moving Averages
            indicators['SMA20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
            indicators['SMA50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
            
            # RSI
            indicators['RSI'] = RSIIndicator(close=df['Close']).rsi()
            
            # Bollinger Bands
            bollinger = BollingerBands(close=df['Close'])
            indicators['Upper Band'] = bollinger.bollinger_hband()
            indicators['Lower Band'] = bollinger.bollinger_lband()
            
            return indicators.fillna(method='bfill')
        
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return None
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Price features
            features['Open'] = df['Open']
            features['High'] = df['High']
            features['Low'] = df['Low']
            features['Close'] = df['Close']
            features['Volume'] = df['Volume']
            
            # Technical indicators
            indicators = self.calculate_indicators(df)
            features = pd.concat([features, indicators], axis=1)
            
            # Price changes
            features['Price_Change'] = df['Close'].pct_change()
            features['Price_Change_1d'] = features['Price_Change'].shift(1)
            features['Price_Change_5d'] = df['Close'].pct_change(periods=5)
            
            # Volatility
            features['Volatility'] = features['Price_Change'].rolling(window=20).std()
            
            # Volume indicators
            features['Volume_Change'] = df['Volume'].pct_change()
            features['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
            
            # Fill missing values
            features = features.fillna(method='bfill')
            
            # Scale features
            scaled_features = pd.DataFrame(
                self.scaler.fit_transform(features),
                index=features.index,
                columns=features.columns
            )
            
            return scaled_features
        
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None
