import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd

class PredictionService:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lstm_model = self._build_lstm_model()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.lr_model = LinearRegression()
        self.metrics = {}
        self.data = None
        
    def _build_lstm_model(self):
        """Build and return the LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def prepare_data(self, data, lookback=60):
        """Prepare data for models"""
        try:
            if len(data) < lookback:
                return None, None
                
            scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
            x_train = []
            y_train = []
            
            for i in range(lookback, len(scaled_data)):
                x_train.append(scaled_data[i-lookback:i, 0])
                y_train.append(scaled_data[i, 0])
                
            return np.array(x_train), np.array(y_train)
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            return None, None

    def train_models(self, symbol):
        """Train all models with the given data"""
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            data = yf.download(symbol, start=start_date, end=end_date)['Close'].values
            
            if len(data) < 60:
                return False
            
            # Prepare data
            x_train, y_train = self.prepare_data(data)
            if x_train is None or y_train is None:
                return False
            
            # Store data for later use
            self.data = data
            
            # Train LSTM
            x_train_lstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            self.lstm_model.fit(x_train_lstm, y_train, epochs=10, batch_size=32, verbose=0)
            lstm_pred = self.lstm_model.predict(x_train_lstm, verbose=0)
            self.metrics['lstm_accuracy'] = float(r2_score(y_train, lstm_pred))
            
            # Train Random Forest
            x_train_rf = x_train.reshape(x_train.shape[0], -1)
            self.rf_model.fit(x_train_rf, y_train)
            rf_pred = self.rf_model.predict(x_train_rf)
            self.metrics['rf_accuracy'] = float(r2_score(y_train, rf_pred))
            
            # Train Linear Regression
            self.lr_model.fit(x_train_rf, y_train)
            lr_pred = self.lr_model.predict(x_train_rf)
            self.metrics['lr_accuracy'] = float(r2_score(y_train, lr_pred))
            
            return True
        except Exception as e:
            print(f"Error training models: {str(e)}")
            self.metrics = {
                'lstm_accuracy': 0.0,
                'rf_accuracy': 0.0,
                'lr_accuracy': 0.0
            }
            return False
            
    def get_prediction(self, symbol, period='1d'):
        """Get price predictions from all models"""
        try:
            # Get historical data
            stock = yf.Ticker(symbol)
            hist = stock.history(period='3mo')
            
            if hist.empty:
                raise ValueError(f"No historical data available for {symbol}")
            
            # Train models if needed
            self.train_models(symbol)
            
            # Prepare latest data for prediction
            latest_data = hist['Close'].values[-60:]
            scaled_data = self.scaler.transform(latest_data.reshape(-1, 1))
            
            # Prepare data for different models
            x_lstm = np.array([scaled_data])
            x_lstm = np.reshape(x_lstm, (x_lstm.shape[0], x_lstm.shape[1], 1))
            x_flat = scaled_data.reshape(1, -1)
            
            # Get predictions from all models
            lstm_pred = self.scaler.inverse_transform(self.lstm_model.predict(x_lstm, verbose=0))[0][0]
            rf_pred = self.scaler.inverse_transform([[self.rf_model.predict(x_flat)[0]]])[0][0]
            lr_pred = self.scaler.inverse_transform([[self.lr_model.predict(x_flat)[0]]])[0][0]
            
            # Calculate ensemble prediction (weighted average)
            weights = {'lstm': 0.5, 'rf': 0.3, 'lr': 0.2}
            ensemble_pred = (
                lstm_pred * weights['lstm'] +
                rf_pred * weights['rf'] +
                lr_pred * weights['lr']
            )
            
            # Adjust prediction based on period
            ensemble_pred = self._adjust_prediction(ensemble_pred, period)
            
            # Calculate confidence based on model performance
            confidence = self._calculate_confidence()
            
            return {
                'ensemble': round(ensemble_pred, 2),
                'lstm': round(lstm_pred, 2),
                'rf': round(rf_pred, 2),
                'lr': round(lr_pred, 2),
                'confidence': confidence,
                'metrics': self.metrics,
                'date': (datetime.now() + self._get_period_delta(period)).strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return self._get_fallback_prediction(symbol, period)
    
    def _adjust_prediction(self, prediction, period):
        """Adjust prediction based on the time period"""
        adjustments = {
            '1d': 1.0,
            '1w': 1.02,
            '1m': 1.05,
            '3m': 1.10,
            '6m': 1.15,
            '1y': 1.20
        }
        return prediction * adjustments.get(period, 1.0)
    
    def _calculate_confidence(self):
        """Calculate confidence score based on model performance"""
        if not self.metrics:
            return 85  # Default confidence
            
        # Use R² scores to calculate confidence
        r2_scores = [
            self.metrics['lstm_accuracy'],
            self.metrics['rf_accuracy'],
            self.metrics['lr_accuracy']
        ]
        avg_r2 = np.mean(r2_scores)
        confidence = min(95, max(60, avg_r2 * 100))  # Scale between 60-95%
        return round(confidence, 1)
    
    def _get_period_delta(self, period):
        """Convert period string to timedelta"""
        period_map = {
            '1d': timedelta(days=1),
            '1w': timedelta(weeks=1),
            '1m': timedelta(days=30),
            '3m': timedelta(days=90),
            '6m': timedelta(days=180),
            '1y': timedelta(days=365)
        }
        return period_map.get(period, timedelta(days=1))
    
    def _get_fallback_prediction(self, symbol, period):
        """Provide a simple fallback prediction when models fail"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period='5d')
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                avg_change = hist['Close'].pct_change().mean()
                prediction = current_price * (1 + avg_change)
                return {
                    'ensemble': round(prediction, 2),
                    'lstm': round(prediction, 2),
                    'rf': round(prediction, 2),
                    'lr': round(prediction, 2),
                    'confidence': 60,
                    'metrics': {},
                    'date': (datetime.now() + self._get_period_delta(period)).strftime('%Y-%m-%d')
                }
        except:
            pass
            
        return {
            'ensemble': 0,
            'lstm': 0,
            'rf': 0,
            'lr': 0,
            'confidence': 0,
            'metrics': {},
            'date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def predict(self, period='1d'):
        """Make predictions using all models"""
        if not hasattr(self, 'data') or self.data is None:
            return None
            
        predictions = {}
        try:
            # Prepare latest data for prediction
            latest_data = self.data[-60:]  # Get last 60 days
            latest_data = self.scaler.transform(latest_data.reshape(-1, 1))
            latest_data = latest_data.reshape(1, 60, 1)  # Reshape for LSTM
            
            # LSTM prediction
            lstm_pred = self.lstm_model.predict(latest_data)
            predictions['LSTM'] = float(self.scaler.inverse_transform(lstm_pred)[0, 0])
            
            # Random Forest prediction
            rf_input = latest_data.reshape(1, -1)
            rf_pred = self.rf_model.predict(rf_input)
            predictions['Random Forest'] = float(self.scaler.inverse_transform([[rf_pred[0]]])[0, 0])
            
            # Linear Regression prediction
            lr_pred = self.lr_model.predict(rf_input)
            predictions['Linear Regression'] = float(self.scaler.inverse_transform([[lr_pred[0]]])[0, 0])
            
            # Ensemble prediction (average of all models)
            predictions['Ensemble'] = sum(predictions.values()) / len(predictions)
            
            return predictions
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return None
