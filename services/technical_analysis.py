import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

class TechnicalAnalysis:
    def __init__(self, data):
        self.data = data
        
    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        # Basic indicators
        self.calculate_moving_averages()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_stochastic()
        
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
        """Calculate MACD"""
        self.data['MACD'] = self.data['EMA12'] - self.data['EMA26']
        self.data['Signal'] = self.data['MACD'].ewm(span=9).mean()
        self.data['MACD_Hist'] = self.data['MACD'] - self.data['Signal']
        
    def calculate_bollinger_bands(self, period=20):
        """Calculate Bollinger Bands"""
        rolling_mean = self.data['Close'].rolling(window=period).mean()
        rolling_std = self.data['Close'].rolling(window=period).std()
        self.data['Upper Band'] = rolling_mean + (rolling_std * 2)
        self.data['Lower Band'] = rolling_mean - (rolling_std * 2)
        
    def calculate_stochastic(self, period=14):
        """Calculate Stochastic Oscillator"""
        low_min = self.data['Low'].rolling(window=period).min()
        high_max = self.data['High'].rolling(window=period).max()
        self.data['%K'] = ((self.data['Close'] - low_min) / (high_max - low_min)) * 100
        self.data['%D'] = self.data['%K'].rolling(window=3).mean()
        
    def get_support_resistance(self):
        """Calculate support and resistance levels"""
        pivot = (self.data['High'].iloc[-1] + self.data['Low'].iloc[-1] + self.data['Close'].iloc[-1]) / 3
        
        support1 = 2 * pivot - self.data['High'].iloc[-1]
        support2 = pivot - (self.data['High'].iloc[-1] - self.data['Low'].iloc[-1])
        
        resistance1 = 2 * pivot - self.data['Low'].iloc[-1]
        resistance2 = pivot + (self.data['High'].iloc[-1] - self.data['Low'].iloc[-1])
        
        return {
            'support1': support1,
            'support2': support2,
            'resistance1': resistance1,
            'resistance2': resistance2
        }
        
    def get_signals(self):
        """Generate trading signals based on indicators"""
        signals = []
        
        # RSI signals
        if self.data['RSI'].iloc[-1] < 30:
            signals.append(('RSI', 'Oversold', 'Buy'))
        elif self.data['RSI'].iloc[-1] > 70:
            signals.append(('RSI', 'Overbought', 'Sell'))
            
        # MACD signals
        if (self.data['MACD_Hist'].iloc[-2] < 0 and self.data['MACD_Hist'].iloc[-1] > 0):
            signals.append(('MACD', 'Crossover', 'Buy'))
        elif (self.data['MACD_Hist'].iloc[-2] > 0 and self.data['MACD_Hist'].iloc[-1] < 0):
            signals.append(('MACD', 'Crossover', 'Sell'))
            
        # Moving Average signals
        if (self.data['Close'].iloc[-1] > self.data['SMA20'].iloc[-1] and 
            self.data['Close'].iloc[-2] < self.data['SMA20'].iloc[-2]):
            signals.append(('SMA20', 'Crossover', 'Buy'))
        elif (self.data['Close'].iloc[-1] < self.data['SMA20'].iloc[-1] and 
              self.data['Close'].iloc[-2] > self.data['SMA20'].iloc[-2]):
            signals.append(('SMA20', 'Crossover', 'Sell'))
            
        return signals

class PredictionService:
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, window_size=60):
        """Prepare data for prediction"""
        # Create features
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()
        self.data['MA_Ratio'] = self.data['Close'] / self.data['SMA20']
        
        # Create sequences
        X = []
        y = []
        
        for i in range(window_size, len(self.data)):
            features = self.data[['Returns', 'Volatility', 'MA_Ratio']].iloc[i-window_size:i].values
            X.append(features)
            y.append(self.data['Close'].iloc[i])
            
        return np.array(X), np.array(y)
        
    def train_model(self):
        """Train the prediction model"""
        X, y = self.prepare_data()
        
        if len(X) == 0:
            return None
            
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        
        return model, X_test, y_test
        
    def make_prediction(self, period='1d'):
        """Make price prediction"""
        model, X_test, y_test = self.train_model()
        
        if model is None:
            return None, 0
            
        # Make prediction
        last_sequence = X_test[-1].reshape(1, -1)
        prediction = model.predict(last_sequence)[0]
        
        # Calculate confidence score
        confidence = model.score(X_test.reshape(X_test.shape[0], -1), y_test)
        
        return prediction, confidence
        
    def get_prediction_factors(self):
        """Get factors affecting the prediction"""
        factors = []
        
        # Trend analysis
        current_price = self.data['Close'].iloc[-1]
        sma20 = self.data['SMA20'].iloc[-1]
        rsi = self.data['RSI'].iloc[-1]
        
        if current_price > sma20:
            factors.append("Price above 20-day moving average (Bullish)")
        else:
            factors.append("Price below 20-day moving average (Bearish)")
            
        if rsi > 70:
            factors.append("RSI indicates overbought conditions")
        elif rsi < 30:
            factors.append("RSI indicates oversold conditions")
            
        # Volatility
        volatility = self.data['Returns'].std() * np.sqrt(252)  # Annualized volatility
        factors.append(f"Current volatility: {volatility:.2%}")
        
        return factors
