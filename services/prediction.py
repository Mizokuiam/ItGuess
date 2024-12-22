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
            
            # Create features and target
            df = df.dropna()
            
            # Check if we have enough data
            if len(df) < 100:  # Need at least 100 data points for meaningful prediction
                print(f"Insufficient data points for {symbol}: {len(df)} < 100")
                return None, None
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'MA5', 'MA20', 'RSI', 'Volume_MA', 'Volatility']
            
            # Validate features
            if not all(col in df.columns for col in features):
                print(f"Missing required features for {symbol}")
                return None, None
            
            X = df[features].values
            y = df['Close'].values
            
            # Scale the data
            self.scalers[symbol] = MinMaxScaler()
            X_scaled = self.scalers[symbol].fit_transform(X)
            
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
            
            # Train Random Forest with error handling
            try:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                rf_score = r2_score(y_test, rf_pred)
                print(f"Random Forest R2 Score: {rf_score:.4f}")
            except Exception as e:
                print(f"Error training Random Forest: {str(e)}")
                return False
            
            # Train Neural Network with error handling
            try:
                nn_model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.1),
                    tf.keras.layers.Dense(1)
                ])
                
                nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                # Add early stopping
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                
                # Train with validation split
                history = nn_model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                nn_pred = nn_model.predict(X_test).flatten()
                nn_score = r2_score(y_test, nn_pred)
                print(f"Neural Network R2 Score: {nn_score:.4f}")
            except Exception as e:
                print(f"Error training Neural Network: {str(e)}")
                return False
            
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
                return None
            
            # Get latest data
            stock = yf.Ticker(symbol)
            df = stock.history(period="60d")
            
            if df.empty:
                return None
            
            # Prepare features
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['RSI'] = self._calculate_rsi(df['Close'])
            df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'MA5', 'MA20', 'RSI', 'Volume_MA', 'Volatility']
            
            X = df[features].iloc[-1:].values
            X_scaled = self.scalers[symbol].transform(X)
            
            predictions = {}
            confidence_intervals = {}
            
            # Make predictions with each model
            for model_name, model in self.models[symbol].items():
                if model_name == 'rf':
                    # Random Forest prediction with confidence interval
                    predictions_array = np.array([tree.predict(X_scaled) for tree in model.estimators_])
                    pred = predictions_array.mean()
                    conf_interval = stats.t.interval(0.95, len(predictions_array)-1,
                                                   loc=pred,
                                                   scale=stats.sem(predictions_array))
                elif model_name == 'nn':
                    # Neural Network prediction
                    pred = model.predict(X_scaled).flatten()[0]
                    # Estimate confidence interval using prediction std
                    std = np.std([model.predict(X_scaled) for _ in range(100)])
                    conf_interval = (pred - 1.96*std, pred + 1.96*std)
                
                predictions[model_name] = pred
                confidence_intervals[model_name] = conf_interval
            
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
