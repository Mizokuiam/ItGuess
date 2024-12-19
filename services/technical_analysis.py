import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import yfinance as yf

class TechnicalAnalysisService:
    def __init__(self, symbol=None, data=None):
        self.symbol = symbol
        self.data = data if data is not None else self._fetch_data()
        
    def _fetch_data(self):
        """Fetch data from Yahoo Finance if symbol is provided"""
        if self.symbol:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(start=start_date, end=end_date)
            return data
        return pd.DataFrame()

    def calculate_indicators(self, data=None):
        """Calculate main technical indicators and return them as a dictionary"""
        if data is not None:
            self.data = data
        
        if self.data.empty:
            return {}

        # Calculate RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Calculate MACD
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        # Calculate Volume SMA
        volume_sma = self.data['Volume'].rolling(window=20).mean()

        # Get the latest values
        latest_rsi = rsi.iloc[-1]
        latest_macd = macd.iloc[-1]
        latest_signal = signal.iloc[-1]
        latest_volume = self.data['Volume'].iloc[-1]
        latest_volume_sma = volume_sma.iloc[-1]

        # Return indicators dictionary
        return {
            'RSI': latest_rsi,
            'MACD': latest_macd,
            'Signal': latest_signal,
            'Volume': latest_volume,
            'Volume_SMA': latest_volume_sma
        }

    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        if self.data.empty:
            return None
            
        # Basic indicators
        self.calculate_moving_averages()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_stochastic()
        
        # Additional indicators
        self.calculate_support_resistance()
        self.calculate_volume_indicators()
        
        return self.data
    
    def calculate_moving_averages(self):
        """Calculate various moving averages"""
        self.data['SMA20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['EMA12'] = self.data['Close'].ewm(span=12).mean()
        self.data['EMA26'] = self.data['Close'].ewm(span=26).mean()
        
    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index"""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
    def calculate_macd(self):
        """Calculate MACD indicator"""
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        self.data['BB_Middle'] = self.data['Close'].rolling(window=period).mean()
        rolling_std = self.data['Close'].rolling(window=period).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (rolling_std * std_dev)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (rolling_std * std_dev)
        
    def calculate_stochastic(self, period=14):
        """Calculate Stochastic Oscillator"""
        low_min = self.data['Low'].rolling(window=period).min()
        high_max = self.data['High'].rolling(window=period).max()
        self.data['%K'] = ((self.data['Close'] - low_min) / (high_max - low_min)) * 100
        self.data['%D'] = self.data['%K'].rolling(window=3).mean()
        
    def calculate_support_resistance(self, window=20):
        """Calculate Support and Resistance levels"""
        self.data['Support'] = self.data['Low'].rolling(window=window).min()
        self.data['Resistance'] = self.data['High'].rolling(window=window).max()
        
    def calculate_volume_indicators(self):
        """Calculate Volume-based indicators"""
        self.data['OBV'] = (np.sign(self.data['Close'].diff()) * self.data['Volume']).cumsum()
        self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20).mean()
        
    def get_trading_signals(self):
        """Generate trading signals based on technical indicators"""
        signals = {}
        
        # RSI signals
        signals['RSI'] = 'Oversold' if self.data['RSI'].iloc[-1] < 30 else 'Overbought' if self.data['RSI'].iloc[-1] > 70 else 'Neutral'
        
        # MACD signals
        signals['MACD'] = 'Buy' if self.data['MACD'].iloc[-1] > self.data['Signal_Line'].iloc[-1] else 'Sell'
        
        # Bollinger Bands signals
        last_close = self.data['Close'].iloc[-1]
        signals['BB'] = 'Oversold' if last_close < self.data['BB_Lower'].iloc[-1] else 'Overbought' if last_close > self.data['BB_Upper'].iloc[-1] else 'Neutral'
        
        # Moving Average signals
        signals['MA'] = 'Bullish' if self.data['SMA20'].iloc[-1] > self.data['SMA50'].iloc[-1] else 'Bearish'
        
        return signals
        
    def get_summary(self):
        """Get a summary of technical analysis"""
        if self.data.empty:
            return None
            
        last_row = self.data.iloc[-1]
        return {
            'price': last_row['Close'],
            'change': ((last_row['Close'] - self.data.iloc[-2]['Close']) / self.data.iloc[-2]['Close']) * 100,
            'rsi': last_row['RSI'],
            'macd': last_row['MACD'],
            'signal_line': last_row['Signal_Line'],
            'bb_upper': last_row['BB_Upper'],
            'bb_lower': last_row['BB_Lower'],
            'signals': self.get_trading_signals()
        }

class PredictionService:
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler()
        self.model = None
        
    def prepare_data(self, window_size=60):
        """Prepare data for prediction"""
        # Prepare features
        features = ['Close', 'Volume', 'RSI', 'MACD', '%K', '%D']
        X = self.data[features].values
        y = self.data['Close'].values
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - window_size):
            X_seq.append(X_scaled[i:(i + window_size)])
            y_seq.append(y[i + window_size])
            
        return np.array(X_seq), np.array(y_seq)
        
    def train_model(self):
        """Train the prediction model"""
        X, y = self.prepare_data()
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        
        return self.model
        
    def make_prediction(self, period='1d'):
        """Make price prediction"""
        if self.model is None:
            self.train_model()
            
        # Prepare last sequence
        X_last = self.prepare_data()[0][-1:]
        
        # Make prediction
        prediction = self.model.predict(X_last.reshape(1, -1))[0]
        
        return {
            'predicted_price': prediction,
            'confidence': self.model.score(X_last.reshape(1, -1), [self.data['Close'].iloc[-1]]),
            'period': period
        }
        
    def get_prediction_factors(self):
        """Get factors affecting the prediction"""
        if self.model is None:
            return None
            
        features = ['Close', 'Volume', 'RSI', 'MACD', '%K', '%D']
        importance = self.model.feature_importances_
        
        return dict(zip(features, importance))
