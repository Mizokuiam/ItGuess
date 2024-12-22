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
                print(f"No trained models found for {symbol}")
                return None
            
            print(f"Starting prediction for {symbol}")
            
            # Get latest data
            stock = yf.Ticker(symbol)
            df = stock.history(period="60d")
            print(f"Fetched {len(df)} days of data")
            
            if df.empty:
                print(f"No historical data available for {symbol}")
                return None
            
            # Prepare features
            try:
                print("Calculating technical indicators...")
                df['MA5'] = df['Close'].rolling(window=5).mean()
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['RSI'] = self._calculate_rsi(df['Close'])
                df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
                df['Returns'] = df['Close'].pct_change()
                df['Volatility'] = df['Returns'].rolling(window=20).std()
                
                features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                           'MA5', 'MA20', 'RSI', 'Volume_MA', 'Volatility']
                
                print("Checking features...")
                # Check if all features are available
                missing_features = [f for f in features if f not in df.columns]
                if missing_features:
                    print(f"Missing features: {missing_features}")
                    return None
                
                # Get the last complete row
                print("Getting last complete row...")
                last_complete_row = df.dropna().iloc[-1:]
                if last_complete_row.empty:
                    print(f"No complete data available for {symbol}")
                    return None
                
                print("Feature values before scaling:")
                for feature in features:
                    print(f"{feature}: {last_complete_row[feature].values[0]}")
                
                X = last_complete_row[features].values
                
                # Scale the features using the same scale as training
                scaler = self.scalers.get(symbol)
                if scaler is None:
                    print(f"No scaler found for {symbol}")
                    return None
                
                # Get the scaling parameters
                print("Scaling parameters:")
                scale_params = {}
                for i, feature in enumerate(features):
                    scale_params[feature] = {
                        'min': float(scaler.data_min_[i]),
                        'max': float(scaler.data_max_[i])
                    }
                    print(f"{feature} - min: {scale_params[feature]['min']}, max: {scale_params[feature]['max']}")
                
                # Apply scaling manually to ensure consistency
                X_scaled = np.zeros_like(X, dtype=np.float64)
                for i, feature in enumerate(features):
                    min_val = scale_params[feature]['min']
                    max_val = scale_params[feature]['max']
                    value = float(X[0, i])
                    
                    print(f"Scaling {feature}:")
                    print(f"  Original value: {value}")
                    print(f"  Min value: {min_val}")
                    print(f"  Max value: {max_val}")
                    
                    if np.isnan(value):
                        print(f"  Warning: NaN value detected for {feature}")
                        X_scaled[0, i] = 0
                    elif max_val - min_val > 1e-10:  # Use small epsilon instead of exact 0
                        X_scaled[0, i] = (value - min_val) / (max_val - min_val)
                        print(f"  Scaled value: {X_scaled[0, i]}")
                    else:
                        print(f"  Warning: No variation in {feature}, using 0")
                        X_scaled[0, i] = 0
                
                print("Final scaled features:")
                for i, feature in enumerate(features):
                    print(f"{feature}: {X_scaled[0, i]}")
                
                predictions = {}
                confidence_intervals = {}
                
                # Make predictions with each model
                for model_name, model in self.models[symbol].items():
                    try:
                        print(f"\nMaking prediction with {model_name}...")
                        if model_name == 'rf':
                            # Random Forest prediction with confidence interval
                            predictions_array = np.array([tree.predict(X_scaled) for tree in model.estimators_])
                            pred = float(predictions_array.mean())
                            print(f"RF raw prediction: {pred}")
                            
                            # Calculate confidence interval
                            std_err = stats.sem(predictions_array)
                            conf_interval = stats.t.interval(0.95, len(predictions_array)-1,
                                                           loc=pred,
                                                           scale=std_err)
                            print(f"RF confidence interval: {conf_interval}")
                            
                        elif model_name == 'nn':
                            # Neural Network prediction
                            pred = float(model.predict(X_scaled).flatten()[0])
                            print(f"NN raw prediction: {pred}")
                            
                            # Estimate confidence interval using multiple predictions
                            predictions_array = np.array([float(model.predict(X_scaled).flatten()[0]) 
                                                        for _ in range(100)])
                            mean_pred = np.mean(predictions_array)
                            std_err = stats.sem(predictions_array)
                            conf_interval = stats.t.interval(0.95, len(predictions_array)-1,
                                                           loc=mean_pred,
                                                           scale=std_err)
                            print(f"NN confidence interval: {conf_interval}")
                            
                        predictions[model_name] = pred
                        confidence_intervals[model_name] = conf_interval
                        print(f"Successfully added {model_name} prediction")
                        
                    except Exception as e:
                        print(f"Error making prediction with {model_name}: {str(e)}")
                        continue
                
                if not predictions:
                    print("No successful predictions made")
                    return None
                
                print("Final predictions:", predictions)
                print("Final confidence intervals:", confidence_intervals)
                
                self.confidence_intervals[symbol] = confidence_intervals
                return predictions
                
            except Exception as e:
                print(f"Error preparing features: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
            
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            import traceback
            traceback.print_exc()
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
