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

    def calculate_indicators(self):
        """Calculate main technical indicators and return them as a dictionary"""
        try:
            if self.data is None or self.data.empty:
                return None

            # Ensure we have enough data
            if len(self.data) < 60:  # Need at least 60 days for calculations
                return None

            # Create a copy of the data to avoid modifying the original
            df = self.data.copy()

            # Calculate RSI with current rsi_period
            rsi = self.calculate_rsi(period=14)
            
            # Calculate MACD
            macd = ta.macd(df['Close'])
            
            # Calculate EMAs
            ema20 = ta.ema(df['Close'], length=20)
            ema50 = ta.ema(df['Close'], length=50)
            
            # Calculate Bollinger Bands
            bb = ta.bbands(df['Close'], length=20)

            # Get the latest values and handle NaN values
            result = {}
            
            # Helper function to safely get the latest value
            def get_latest_value(series):
                if series is None or isinstance(series, float):
                    return series
                if isinstance(series, pd.Series) and not series.empty:
                    latest = series.iloc[-1]
                    return None if pd.isna(latest) else latest
                if isinstance(series, np.ndarray):
                    return series[-1] if len(series) > 0 else None
                return None

            # Add indicators to result dict
            result['RSI'] = get_latest_value(rsi)
            result['MACD'] = get_latest_value(macd['MACD_12_26_9'])
            result['Signal'] = get_latest_value(macd['MACDs_12_26_9'])
            result['EMA20'] = get_latest_value(ema20)
            result['EMA50'] = get_latest_value(ema50)
            result['BB_Upper'] = get_latest_value(bb['BBU_20_2.0'])
            result['BB_Middle'] = get_latest_value(bb['BBM_20_2.0'])
            result['BB_Lower'] = get_latest_value(bb['BBL_20_2.0'])
            result['Volume'] = get_latest_value(df['Volume'])

            # Round numeric values
            for key in result:
                if result[key] is not None and not pd.isna(result[key]):
                    if key == 'Volume':
                        result[key] = int(result[key])
                    else:
                        result[key] = round(float(result[key]), 2)
                else:
                    result[key] = 'N/A'

            return result
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return None

    def calculate_rsi(self, data, period=14):
        """Calculate RSI for given price data
        
        Args:
            data: numpy array or pandas Series of prices
            period: RSI period/window (default: 14)
        """
        try:
            # Validate period
            if not isinstance(period, (int, float)) or period <= 0:
                raise ValueError("RSI period must be a positive number")
            period = int(period)  # Convert to integer
            
            # Calculate RSI using pandas_ta
            if data is not None and len(data) > 0:
                rsi = pd.Series(ta.rsi(data, length=period))
                return rsi
            return None
            
        except Exception as e:
            print(f"Error calculating RSI: {str(e)}")
            return None

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence/Divergence)"""
        try:
            if len(prices) < slow + signal:
                return None, None
            
            # Calculate EMAs
            fast_ema = prices.ewm(span=fast, min_periods=1, adjust=False).mean()
            slow_ema = prices.ewm(span=slow, min_periods=1, adjust=False).mean()
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal, min_periods=1, adjust=False).mean()
            
            return macd_line, signal_line
            
        except Exception as e:
            print(f"Error calculating MACD: {str(e)}")
            return None, None

    def calculate_ema(self, data, window=20):
        """Calculate EMA for given data"""
        try:
            if not isinstance(window, (int, float)) or window <= 0:
                raise ValueError("EMA window must be a positive number")
            window = int(window)
            return pd.Series(data).ewm(span=window, adjust=False).mean()
        except Exception as e:
            print(f"Error calculating EMA: {str(e)}")
            return None

    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """Calculate Bollinger Bands for given data"""
        try:
            if not isinstance(window, (int, float)) or window <= 0:
                raise ValueError("Bollinger Bands window must be a positive number")
            window = int(window)
            
            sma = pd.Series(data).rolling(window=window).mean()
            std = pd.Series(data).rolling(window=window).std()
            upper_band = sma + (std * num_std)
            lower_band = sma - (std * num_std)
            return upper_band, sma, lower_band
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {str(e)}")
            return None, None, None

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
        self.data['SMA20'] = self.data['Close'].rolling(window=20, min_periods=1).mean()
        self.data['SMA50'] = self.data['Close'].rolling(window=50, min_periods=1).mean()
        self.data['EMA12'] = self.data['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
        self.data['EMA26'] = self.data['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
        
    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index
        
        Args:
            period: RSI period/window (default: 14)
        """
        try:
            # Validate period
            if not isinstance(period, (int, float)) or period <= 0:
                raise ValueError("RSI period must be a positive number")
            period = int(period)  # Convert to integer
            
            # Calculate RSI using pandas_ta
            if self.data is not None and not self.data.empty:
                rsi = pd.Series(ta.rsi(self.data['Close'], length=period))
                return rsi
            return None
            
        except Exception as e:
            print(f"Error calculating RSI: {str(e)}")
            return None

    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        try:
            exp1 = self.data['Close'].ewm(span=fast, adjust=False, min_periods=1).mean()
            exp2 = self.data['Close'].ewm(span=slow, adjust=False, min_periods=1).mean()
            self.data['MACD'] = exp1 - exp2
            self.data['Signal_Line'] = self.data['MACD'].ewm(span=signal, adjust=False, min_periods=1).mean()
        except Exception as e:
            print(f"Error calculating MACD: {str(e)}")
            self.data['MACD'] = float('nan')
            self.data['Signal_Line'] = float('nan')
        
    def calculate_bollinger_bands(self, period=20, num_std=2):
        """Calculate Bollinger Bands"""
        if period <= 0:
            raise ValueError("Bollinger Bands period must be greater than 0")
        try:
            self.data['BB_Middle'] = self.data['Close'].rolling(window=period, min_periods=1).mean()
            rolling_std = self.data['Close'].rolling(window=period, min_periods=1).std()
            self.data['BB_Upper'] = self.data['BB_Middle'] + (rolling_std * num_std)
            self.data['BB_Lower'] = self.data['BB_Middle'] - (rolling_std * num_std)
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {str(e)}")
            self.data['BB_Middle'] = float('nan')
            self.data['BB_Upper'] = float('nan')
            self.data['BB_Lower'] = float('nan')
        
    def calculate_stochastic(self, period=14):
        """Calculate Stochastic Oscillator"""
        try:
            low_min = self.data['Low'].rolling(window=period, min_periods=1).min()
            high_max = self.data['High'].rolling(window=period, min_periods=1).max()
            self.data['%K'] = ((self.data['Close'] - low_min) / (high_max - low_min)) * 100
            self.data['%D'] = self.data['%K'].rolling(window=3, min_periods=1).mean()
        except Exception as e:
            print(f"Error calculating Stochastic Oscillator: {str(e)}")
            self.data['%K'] = float('nan')
            self.data['%D'] = float('nan')
        
    def calculate_support_resistance(self, window=20):
        """Calculate Support and Resistance levels"""
        if window <= 0:
            raise ValueError("Support and Resistance window must be greater than 0")
        try:
            self.data['Support'] = self.data['Low'].rolling(window=window, min_periods=1).min()
            self.data['Resistance'] = self.data['High'].rolling(window=window, min_periods=1).max()
        except Exception as e:
            print(f"Error calculating Support and Resistance levels: {str(e)}")
            self.data['Support'] = float('nan')
            self.data['Resistance'] = float('nan')
        
    def calculate_volume_indicators(self):
        """Calculate Volume-based indicators"""
        try:
            self.data['OBV'] = (np.sign(self.data['Close'].diff()) * self.data['Volume']).cumsum()
            self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20, min_periods=1).mean()
        except Exception as e:
            print(f"Error calculating Volume-based indicators: {str(e)}")
            self.data['OBV'] = float('nan')
            self.data['Volume_SMA'] = float('nan')
        
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
