import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from scipy import stats
import time

class PredictionService:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.history = {}
        self.confidence_intervals = {}
        self.feature_importance = {}
        
    def _prepare_data(self, symbol):
        """Prepare data for prediction"""
        try:
            # Get historical data with retry mechanism
            max_retries = 3
            retry_count = 0
            df = None
            
            while retry_count < max_retries:
                try:
                    stock = yf.Ticker(symbol)
                    df = stock.history(period="2y")
                    if not df.empty:
                        break
                except Exception as e:
                    print(f"Attempt {retry_count + 1} failed: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(1)
            
            if df is None or df.empty:
                print(f"Could not fetch data for {symbol} after {max_retries} attempts")
                return None, None
            
            # Calculate technical indicators
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['RSI'] = self._calculate_rsi(df['Close'])
            df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # Drop any rows with NaN values
            df = df.dropna()
            
            # Create target variable (next day's closing price)
            df['Target'] = df['Close'].shift(-1)
            df = df.dropna()  # Drop the last row which will have NaN target
            
            # Check if we have enough data
            if len(df) < 100:
                print(f"Insufficient data points for {symbol}: {len(df)} < 100")
                return None, None
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'MA5', 'MA20', 'RSI', 'Volume_MA', 'Volatility']
            
            # Validate features
            if not all(col in df.columns for col in features):
                print(f"Missing required features for {symbol}")
                return None, None
            
            X = df[features].values
            y = df['Target'].values
            
            # Create and fit the scaler
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Store the scaler
            self.scalers[symbol] = scaler
            
            return X_scaled, y
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            return None, None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def train_models(self, symbol):
        """Train prediction models"""
        try:
            X, y = self._prepare_data(symbol)
            
            if X is None or y is None:
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Train Neural Network
            nn_model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1)
            ])
            
            nn_model.compile(optimizer='adam', loss='mse')
            nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            
            # Store models
            self.models[symbol] = {
                'rf': rf_model,
                'nn': nn_model
            }
            
            # Calculate metrics
            self._calculate_metrics(symbol, X_test, y_test)
            
            # Store feature importance
            self._calculate_feature_importance(symbol)
            
            return True
            
        except Exception as e:
            print(f"Error training models: {str(e)}")
            return False
            
    def _calculate_metrics(self, symbol, X_test, y_test):
        """Calculate model performance metrics"""
        metrics = {}
        history = {'actual': [], 'rf_pred': [], 'nn_pred': []}
        
        for model_name, model in self.models[symbol].items():
            y_pred = model.predict(X_test)
            if model_name == 'nn':
                y_pred = y_pred.flatten()
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics[model_name] = {
                'mse': mse,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
            
            # Store predictions for history
            history[f'{model_name}_pred'] = y_pred.tolist()
        
        history['actual'] = y_test.tolist()
        
        self.metrics[symbol] = metrics
        self.history[symbol] = history
    
    def _calculate_feature_importance(self, symbol):
        """Calculate and store feature importance"""
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'MA5', 'MA20', 'RSI', 'Volume_MA', 'Volatility']
        
        rf_model = self.models[symbol]['rf']
        importance = rf_model.feature_importances_
        
        self.feature_importance[symbol] = dict(zip(features, importance))
    
    def predict(self, symbol, period='1d'):
        """Make predictions with confidence intervals"""
        try:
            if symbol not in self.models:
                print(f"No trained models found for {symbol}")
                return None
            
            # Get latest data
            stock = yf.Ticker(symbol)
            df = stock.history(period="60d")
            
            if df.empty:
                print(f"No historical data available for {symbol}")
                return None
            
            # Calculate technical indicators
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['RSI'] = self._calculate_rsi(df['Close'])
            df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # Get the last complete row
            df = df.dropna()
            if df.empty:
                print(f"No complete data available for {symbol}")
                return None
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'MA5', 'MA20', 'RSI', 'Volume_MA', 'Volatility']
            
            last_row = df.iloc[-1:][features]
            
            # Scale the features
            scaler = self.scalers.get(symbol)
            if scaler is None:
                print(f"No scaler found for {symbol}")
                return None
            
            X = scaler.transform(last_row)
            
            predictions = {}
            confidence_intervals = {}
            
            # Make predictions
            for model_name, model in self.models[symbol].items():
                try:
                    if model_name == 'rf':
                        # Get predictions from all trees
                        tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
                        pred = np.mean(tree_preds)
                        std_err = stats.sem(tree_preds)
                        ci = stats.t.interval(0.95, len(tree_preds)-1, loc=pred, scale=std_err)
                    else:  # Neural Network
                        # Get multiple predictions
                        preds = np.array([model.predict(X, verbose=0)[0][0] for _ in range(100)])
                        pred = np.mean(preds)
                        std_err = stats.sem(preds)
                        ci = stats.t.interval(0.95, len(preds)-1, loc=pred, scale=std_err)
                    
                    predictions[model_name] = float(pred)
                    confidence_intervals[model_name] = (float(ci[0]), float(ci[1]))
                    
                except Exception as e:
                    print(f"Error making prediction with {model_name}: {str(e)}")
                    continue
            
            if not predictions:
                print("No successful predictions made")
                return None
            
            self.confidence_intervals[symbol] = confidence_intervals
            return predictions
            
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return None
    
    def get_prediction_history(self, symbol):
        """Get prediction history"""
        return self.history.get(symbol, None)
    
    def get_confidence_intervals(self, symbol):
        """Get confidence intervals for predictions"""
        return self.confidence_intervals.get(symbol, None)
    
    def get_feature_importance(self, symbol):
        """Get feature importance scores"""
        return self.feature_importance.get(symbol, None)
