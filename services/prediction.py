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
        self.features = []
        
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_momentum(self, prices, period=10):
        """Calculate price momentum"""
        return prices.diff(period)

    def _calculate_bollinger_bands(self, prices, window=20):
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        return upper_band - lower_band  # Return bandwidth as a feature

    def _calculate_macd(self, prices):
        """Calculate MACD"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal  # Return MACD histogram

    def _prepare_data(self, symbol):
        """Prepare data for prediction"""
        try:
            # Get historical data
            stock = yf.Ticker(symbol)
            df = stock.history(period="2y")
            
            if df.empty:
                print(f"No historical data available for {symbol}")
                return None, None

            # Basic price and volume features
            df['Returns'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
            
            # Technical indicators
            df['RSI'] = self._calculate_rsi(df['Close'])
            df['Momentum'] = self._calculate_momentum(df['Close'])
            df['BB_Width'] = self._calculate_bollinger_bands(df['Close'])
            df['MACD_Hist'] = self._calculate_macd(df['Close'])
            
            # Moving averages and trends
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['Trend_5_20'] = df['MA5'] - df['MA20']
            df['Trend_20_50'] = df['MA20'] - df['MA50']
            
            # Volume indicators
            df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Trend'] = df['Volume_MA5'] - df['Volume_MA20']
            
            # Volatility
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            df['Volatility_Change'] = df['Volatility'].pct_change()
            
            # Price patterns
            df['Higher_Highs'] = (df['High'] > df['High'].shift(1)).astype(int)
            df['Lower_Lows'] = (df['Low'] < df['Low'].shift(1)).astype(int)
            df['Price_Trend'] = df['Higher_Highs'] - df['Lower_Lows']
            
            # Target variable (next day's return)
            df['Target'] = df['Close'].shift(-1) / df['Close'] - 1
            
            # Drop missing values
            df = df.dropna()
            
            if len(df) < 100:
                print(f"Insufficient data points for {symbol}")
                return None, None
            
            # Features for prediction
            features = [
                'Returns', 'Volume_Change', 'Price_Range',
                'RSI', 'Momentum', 'BB_Width', 'MACD_Hist',
                'Trend_5_20', 'Trend_20_50',
                'Volume_Trend', 'Volatility', 'Volatility_Change',
                'Price_Trend'
            ]
            
            X = df[features].values
            y = df['Target'].values
            
            # Scale the features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Store the scaler and feature list
            self.scalers[symbol] = scaler
            self.features = features
            
            return X_scaled, y
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            return None, None

    def train_models(self, symbol):
        """Train prediction models"""
        try:
            print(f"\nTraining models for {symbol}...")
            X, y = self._prepare_data(symbol)
            
            if X is None or y is None:
                print("Data preparation failed")
                return False
                
            print(f"Training data shape: X={X.shape}, y={y.shape}")
            print("Feature list:", self.features)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest
            print("\nTraining Random Forest...")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_score = r2_score(y_test, rf_pred)
            print(f"Random Forest R2 Score: {rf_score:.4f}")
            
            # Train Neural Network
            print("\nTraining Neural Network...")
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
            nn_pred = nn_model.predict(X_test, verbose=0).flatten()
            nn_score = r2_score(y_test, nn_pred)
            print(f"Neural Network R2 Score: {nn_score:.4f}")
            
            # Store models
            self.models[symbol] = {
                'rf': rf_model,
                'nn': nn_model
            }
            
            print("\nModels trained successfully")
            return True
            
        except Exception as e:
            print(f"Error training models: {str(e)}")
            import traceback
            traceback.print_exc()
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
        features = self.features
        
        rf_model = self.models[symbol]['rf']
        importance = rf_model.feature_importances_
        
        self.feature_importance[symbol] = dict(zip(features, importance))
    
    def predict(self, symbol, period='1d'):
        """Make predictions with confidence intervals"""
        try:
            print(f"\nStarting prediction for {symbol}...")
            
            if symbol not in self.models:
                print(f"No trained models found for {symbol}")
                return None
            
            print("Models available:", list(self.models[symbol].keys()))
            
            # Get latest data
            print("\nFetching latest data...")
            stock = yf.Ticker(symbol)
            df = stock.history(period="60d")
            
            if df.empty:
                print(f"No historical data available for {symbol}")
                return None
                
            print(f"Fetched {len(df)} days of data")
            
            # Calculate all features
            print("\nCalculating features...")
            df['Returns'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
            df['RSI'] = self._calculate_rsi(df['Close'])
            df['Momentum'] = self._calculate_momentum(df['Close'])
            df['BB_Width'] = self._calculate_bollinger_bands(df['Close'])
            df['MACD_Hist'] = self._calculate_macd(df['Close'])
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['Trend_5_20'] = df['MA5'] - df['MA20']
            df['Trend_20_50'] = df['MA20'] - df['MA50']
            df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Trend'] = df['Volume_MA5'] - df['Volume_MA20']
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            df['Volatility_Change'] = df['Volatility'].pct_change()
            df['Higher_Highs'] = (df['High'] > df['High'].shift(1)).astype(int)
            df['Lower_Lows'] = (df['Low'] < df['Low'].shift(1)).astype(int)
            df['Price_Trend'] = df['Higher_Highs'] - df['Lower_Lows']
            
            # Drop rows with missing values
            print("\nChecking for missing values...")
            before_drop = len(df)
            df = df.dropna()
            after_drop = len(df)
            print(f"Rows before dropna: {before_drop}, after: {after_drop}")
            
            if df.empty:
                print(f"No complete data available for {symbol}")
                return None
            
            # Get the last row
            last_row = df.iloc[-1:]
            
            # Verify features
            print("\nVerifying features...")
            missing_features = [f for f in self.features if f not in last_row.columns]
            if missing_features:
                print(f"Missing features: {missing_features}")
                return None
                
            print("Features available:", list(last_row.columns))
            
            # Prepare features in the same order as training
            print("\nPreparing features for prediction...")
            X = last_row[self.features].values
            print("Input shape:", X.shape)
            print("Feature values:", X[0])
            
            # Scale features
            print("\nScaling features...")
            scaler = self.scalers.get(symbol)
            if scaler is None:
                print(f"No scaler found for {symbol}")
                return None
                
            X_scaled = scaler.transform(X)
            print("Scaled shape:", X_scaled.shape)
            print("Scaled values:", X_scaled[0])
            
            predictions = {}
            confidence_intervals = {}
            
            current_price = float(last_row['Close'].values[0])
            print(f"\nCurrent price: {current_price}")
            
            # Make predictions with each model
            print("\nMaking predictions...")
            for model_name, model in self.models[symbol].items():
                try:
                    print(f"\nPredicting with {model_name}...")
                    if model_name == 'rf':
                        # Get predictions from all trees
                        tree_preds = np.array([tree.predict(X_scaled)[0] for tree in model.estimators_])
                        return_pred = np.mean(tree_preds)
                        print(f"RF return prediction: {return_pred:.4f}")
                        
                        if len(tree_preds) > 1:
                            std_err = stats.sem(tree_preds)
                            return_ci = stats.t.interval(0.95, len(tree_preds)-1, loc=return_pred, scale=std_err)
                        else:
                            return_ci = (return_pred, return_pred)
                        
                        # Convert return prediction to price
                        pred_price = current_price * (1 + return_pred)
                        ci_prices = (current_price * (1 + return_ci[0]), current_price * (1 + return_ci[1]))
                        print(f"RF price prediction: {pred_price:.2f}")
                        
                    else:  # Neural Network
                        # Get multiple predictions
                        return_preds = np.array([model.predict(X_scaled, verbose=0)[0][0] for _ in range(100)])
                        return_pred = np.mean(return_preds)
                        print(f"NN return prediction: {return_pred:.4f}")
                        
                        if len(return_preds) > 1:
                            std_err = stats.sem(return_preds)
                            return_ci = stats.t.interval(0.95, len(return_preds)-1, loc=return_pred, scale=std_err)
                        else:
                            return_ci = (return_pred, return_pred)
                        
                        # Convert return prediction to price
                        pred_price = current_price * (1 + return_pred)
                        ci_prices = (current_price * (1 + return_ci[0]), current_price * (1 + return_ci[1]))
                        print(f"NN price prediction: {pred_price:.2f}")
                    
                    predictions[model_name] = float(pred_price)
                    confidence_intervals[model_name] = (float(ci_prices[0]), float(ci_prices[1]))
                    print(f"Prediction stored for {model_name}")
                    
                except Exception as e:
                    print(f"Error making prediction with {model_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not predictions:
                print("No successful predictions made")
                return None
            
            print("\nPredictions completed successfully")
            print("Final predictions:", predictions)
            print("Confidence intervals:", confidence_intervals)
            
            self.confidence_intervals[symbol] = confidence_intervals
            return predictions
            
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
