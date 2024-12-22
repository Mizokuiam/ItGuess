import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

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
            if len(self.data) < 50:  # Need at least 50 data points for MA50
                return {'MA20': None, 'MA50': None, 'signal': 'Neutral'}
                
            ma20 = self.data['Close'].rolling(window=20).mean()
            ma50 = self.data['Close'].rolling(window=50).mean()
            
            # Check for valid data
            if pd.isna(ma20.iloc[-1]) or pd.isna(ma50.iloc[-1]):
                return {'MA20': None, 'MA50': None, 'signal': 'Neutral'}
            
            # Generate signal
            signal = "Buy" if ma20.iloc[-1] > ma50.iloc[-1] else "Sell"
            
            return {
                'MA20': float(ma20.iloc[-1]),  # Convert to float to avoid scalar issues
                'MA50': float(ma50.iloc[-1]),
                'signal': signal
            }
        except Exception as e:
            print(f"Error in MA cross calculation: {str(e)}")
            return {'MA20': None, 'MA50': None, 'signal': 'Neutral'}
    
    def _calculate_macd(self):
        """Calculate MACD"""
        try:
            if len(self.data) < 26:  # Need at least 26 data points for MACD
                return {'MACD': None, 'Signal': None, 'signal': 'Neutral'}
                
            exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=9, adjust=False).mean()
            
            # Check for valid data
            if pd.isna(macd.iloc[-1]) or pd.isna(signal_line.iloc[-1]):
                return {'MACD': None, 'Signal': None, 'signal': 'Neutral'}
            
            # Generate signal
            current_macd = float(macd.iloc[-1])  # Convert to float
            current_signal = float(signal_line.iloc[-1])  # Convert to float
            
            buy_signal = current_macd > current_signal
            signal_str = "Buy" if buy_signal else "Sell"
            
            return {
                'MACD': current_macd,
                'Signal': current_signal,
                'signal': signal_str
            }
        except Exception as e:
            print(f"Error in MACD calculation: {str(e)}")
            return {'MACD': None, 'Signal': None, 'signal': 'Neutral'}
    
    def _calculate_rsi(self, period=14):
        """Calculate RSI"""
        try:
            if len(self.data) < period:
                return {'RSI': None, 'signal': 'Neutral'}
                
            delta = self.data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            
            # Avoid division by zero
            rs = pd.Series(0, index=gain.index)
            valid_loss = loss != 0
            rs[valid_loss] = gain[valid_loss] / loss[valid_loss]
            
            rsi = 100 - (100 / (1 + rs))
            
            # Check for valid data
            if pd.isna(rsi.iloc[-1]):
                return {'RSI': None, 'signal': 'Neutral'}
            
            current_rsi = float(rsi.iloc[-1])  # Convert to float
            
            # Generate signal
            if current_rsi > 70:
                signal = "Sell"
            elif current_rsi < 30:
                signal = "Buy"
            else:
                signal = "Neutral"
            
            return {
                'RSI': current_rsi,
                'signal': signal
            }
        except Exception as e:
            print(f"Error in RSI calculation: {str(e)}")
            return {'RSI': None, 'signal': 'Neutral'}
    
    def _calculate_stochastic(self, period=14):
        """Calculate Stochastic Oscillator"""
        try:
            if len(self.data) < period:
                return {'K': None, 'D': None, 'signal': 'Neutral'}
                
            low_min = self.data['Low'].rolling(window=period).min()
            high_max = self.data['High'].rolling(window=period).max()
            
            # Handle division by zero
            k = pd.Series(0, index=self.data.index)
            denominator = high_max - low_min
            valid_denom = denominator != 0
            k[valid_denom] = 100 * ((self.data['Close'][valid_denom] - low_min[valid_denom]) / denominator[valid_denom])
            
            d = k.rolling(window=3).mean()
            
            # Check for valid data
            if pd.isna(k.iloc[-1]) or pd.isna(d.iloc[-1]):
                return {'K': None, 'D': None, 'signal': 'Neutral'}
            
            current_k = float(k.iloc[-1])  # Convert to float
            current_d = float(d.iloc[-1])  # Convert to float
            
            # Generate signal
            if current_k > 80 and current_d > 80:
                signal = "Sell"
            elif current_k < 20 and current_d < 20:
                signal = "Buy"
            else:
                signal = "Neutral"
            
            return {
                'K': current_k,
                'D': current_d,
                'signal': signal
            }
        except Exception as e:
            print(f"Error in stochastic calculation: {str(e)}")
            return {'K': None, 'D': None, 'signal': 'Neutral'}
    
    def _calculate_obv(self):
        """Calculate On-Balance Volume"""
        try:
            if len(self.data) < 20:  # Need at least 20 points for the SMA
                return {'OBV': None, 'trend': 'neutral', 'signal': 'Neutral'}
            
            # Calculate price changes
            price_diff = self.data['Close'].diff()
            
            # Calculate OBV
            obv = pd.Series(0, index=self.data.index)
            obv_values = []
            current_obv = 0
            
            for i in range(len(self.data)):
                if i == 0:
                    obv_values.append(current_obv)
                    continue
                    
                if price_diff.iloc[i] > 0:
                    current_obv += self.data['Volume'].iloc[i]
                elif price_diff.iloc[i] < 0:
                    current_obv -= self.data['Volume'].iloc[i]
                
                obv_values.append(current_obv)
            
            obv = pd.Series(obv_values, index=self.data.index)
            obv_sma = obv.rolling(window=20).mean()
            
            # Check for valid data
            if pd.isna(obv.iloc[-1]) or pd.isna(obv_sma.iloc[-1]):
                return {'OBV': None, 'trend': 'neutral', 'signal': 'Neutral'}
            
            current_obv = float(obv.iloc[-1])  # Convert to float
            current_sma = float(obv_sma.iloc[-1])  # Convert to float
            
            # Generate signal
            trend = "up" if current_obv > current_sma else "down"
            signal = "Buy" if trend == "up" else "Sell"
            
            return {
                'OBV': current_obv,
                'trend': trend,
                'signal': signal
            }
        except Exception as e:
            print(f"Error in OBV calculation: {str(e)}")
            return {'OBV': None, 'trend': 'neutral', 'signal': 'Neutral'}
    
    def _calculate_volume_analysis(self):
        """Analyze volume patterns"""
        try:
            if len(self.data) < 20:  # Need at least 20 points for the MA
                return {'Volume': None, 'MA20': None, 'trend': 'neutral', 'signal': 'Neutral'}
            
            volume = self.data['Volume']
            volume_ma = volume.rolling(window=20).mean()
            
            # Check for valid data
            if pd.isna(volume.iloc[-1]) or pd.isna(volume_ma.iloc[-1]):
                return {'Volume': None, 'MA20': None, 'trend': 'neutral', 'signal': 'Neutral'}
            
            current_volume = float(volume.iloc[-1])  # Convert to float
            current_ma = float(volume_ma.iloc[-1])  # Convert to float
            
            # Determine volume trend
            trend = "up" if current_volume > current_ma else "down"
            
            # Make sure we have valid price data
            if pd.isna(self.data['Close'].iloc[-1]) or pd.isna(self.data['Open'].iloc[-1]):
                return {'Volume': current_volume, 'MA20': current_ma, 'trend': trend, 'signal': 'Neutral'}
            
            # Generate signal
            if trend == "up" and self.data['Close'].iloc[-1] > self.data['Open'].iloc[-1]:
                signal = "Buy"
            elif trend == "down" and self.data['Close'].iloc[-1] < self.data['Open'].iloc[-1]:
                signal = "Sell"
            else:
                signal = "Neutral"
            
            return {
                'Volume': current_volume,
                'MA20': current_ma,
                'trend': trend,
                'signal': signal
            }
        except Exception as e:
            print(f"Error in volume analysis: {str(e)}")
            return {'Volume': None, 'MA20': None, 'trend': 'neutral', 'signal': 'Neutral'}

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
