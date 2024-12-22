import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import yfinance as yf

class TechnicalAnalysisService:
    def __init__(self):
        self.data = None
        
    def analyze(self, symbol):
        """Analyze stock data and return technical indicators"""
        try:
            # Get data
            stock = yf.Ticker(symbol)
            self.data = stock.history(period="1y")
            
            if self.data.empty:
                return None
            
            # Calculate all indicators
            analysis = {
                'ma_cross': self._calculate_ma_cross(),
                'macd': self._calculate_macd(),
                'rsi': self._calculate_rsi(),
                'stochastic': self._calculate_stochastic(),
                'obv': self._calculate_obv(),
                'volume': self._calculate_volume_analysis()
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error in technical analysis: {str(e)}")
            return None
    
    def _calculate_ma_cross(self):
        """Calculate Moving Average Crossover"""
        try:
            ma20 = self.data['Close'].rolling(window=20).mean()
            ma50 = self.data['Close'].rolling(window=50).mean()
            
            # Generate signal
            signal = "Buy" if ma20.iloc[-1] > ma50.iloc[-1] else "Sell"
            
            return {
                'MA20': ma20,
                'MA50': ma50,
                'signal': signal
            }
        except:
            return {'MA20': [], 'MA50': [], 'signal': 'Neutral'}
    
    def _calculate_macd(self):
        """Calculate MACD"""
        try:
            exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            # Generate signal
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            
            buy_signal = current_macd > current_signal
            signal_str = "Buy" if buy_signal else "Sell"
            
            return {
                'MACD': macd,
                'Signal': signal,
                'signal': signal_str
            }
        except:
            return {'MACD': [], 'Signal': [], 'signal': 'Neutral'}
    
    def _calculate_rsi(self, period=14):
        """Calculate RSI"""
        try:
            delta = self.data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            
            # Generate signal
            if current_rsi > 70:
                signal = "Sell"
            elif current_rsi < 30:
                signal = "Buy"
            else:
                signal = "Neutral"
            
            return {
                'RSI': rsi,
                'value': current_rsi,
                'signal': signal
            }
        except:
            return {'RSI': [], 'value': 50, 'signal': 'Neutral'}
    
    def _calculate_stochastic(self, period=14):
        """Calculate Stochastic Oscillator"""
        try:
            low_min = self.data['Low'].rolling(window=period).min()
            high_max = self.data['High'].rolling(window=period).max()
            
            k = 100 * ((self.data['Close'] - low_min) / (high_max - low_min))
            d = k.rolling(window=3).mean()
            
            current_k = k.iloc[-1]
            current_d = d.iloc[-1]
            
            # Generate signal
            if current_k > 80 and current_d > 80:
                signal = "Sell"
            elif current_k < 20 and current_d < 20:
                signal = "Buy"
            else:
                signal = "Neutral"
            
            return {
                'K': k,
                'D': d,
                'signal': signal
            }
        except:
            return {'K': [], 'D': [], 'signal': 'Neutral'}
    
    def _calculate_obv(self):
        """Calculate On-Balance Volume"""
        try:
            obv = (np.sign(self.data['Close'].diff()) * self.data['Volume']).fillna(0).cumsum()
            
            # Calculate OBV trend
            obv_sma = obv.rolling(window=20).mean()
            trend = "up" if obv.iloc[-1] > obv_sma.iloc[-1] else "down"
            
            # Generate signal
            signal = "Buy" if trend == "up" else "Sell"
            
            return {
                'OBV': obv,
                'trend': trend,
                'signal': signal
            }
        except:
            return {'OBV': [], 'trend': 'neutral', 'signal': 'Neutral'}
    
    def _calculate_volume_analysis(self):
        """Analyze volume patterns"""
        try:
            volume = self.data['Volume']
            volume_ma = volume.rolling(window=20).mean()
            
            current_volume = volume.iloc[-1]
            current_ma = volume_ma.iloc[-1]
            
            # Determine volume trend
            trend = "up" if current_volume > current_ma else "down"
            
            # Generate signal
            if trend == "up" and self.data['Close'].iloc[-1] > self.data['Open'].iloc[-1]:
                signal = "Buy"
            elif trend == "down" and self.data['Close'].iloc[-1] < self.data['Open'].iloc[-1]:
                signal = "Sell"
            else:
                signal = "Neutral"
            
            return {
                'Volume': volume,
                'MA20': volume_ma,
                'trend': trend,
                'signal': signal
            }
        except:
            return {'Volume': [], 'MA20': [], 'trend': 'neutral', 'signal': 'Neutral'}

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
