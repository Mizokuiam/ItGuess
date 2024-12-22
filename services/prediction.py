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
            print(f"\nPreparing data for {symbol}...")
            
            # Get historical data
            stock = yf.Ticker(symbol)
            df = stock.history(period="2y")
            print(f"Fetched {len(df)} days of historical data")
            
            if df.empty:
                print(f"No historical data available for {symbol}")
                return None, None

            # Print initial data sample
            print("\nInitial data sample:")
            print(df.head())
            
            # Calculate features with validation
            print("\nCalculating features...")
            feature_data = {}
            
            # Basic price and volume features
            try:
                feature_data['Returns'] = df['Close'].pct_change()
                feature_data['Volume_Change'] = df['Volume'].pct_change()
                feature_data['Price_Range'] = (df['High'] - df['Low']) / df['Close']
            except Exception as e:
                print(f"Error calculating basic features: {e}")
                return None, None
            
            # Technical indicators
            try:
                feature_data['RSI'] = self._calculate_rsi(df['Close'])
                feature_data['Momentum'] = self._calculate_momentum(df['Close'])
                feature_data['BB_Width'] = self._calculate_bollinger_bands(df['Close'])
                feature_data['MACD_Hist'] = self._calculate_macd(df['Close'])
            except Exception as e:
                print(f"Error calculating technical indicators: {e}")
                return None, None
            
            # Moving averages and trends
            try:
                feature_data['MA5'] = df['Close'].rolling(window=5).mean()
                feature_data['MA20'] = df['Close'].rolling(window=20).mean()
                feature_data['MA50'] = df['Close'].rolling(window=50).mean()
                feature_data['Trend_5_20'] = feature_data['MA5'] - feature_data['MA20']
                feature_data['Trend_20_50'] = feature_data['MA20'] - feature_data['MA50']
            except Exception as e:
                print(f"Error calculating moving averages: {e}")
                return None, None
            
            # Volume indicators
            try:
                feature_data['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
                feature_data['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
                feature_data['Volume_Trend'] = feature_data['Volume_MA5'] - feature_data['Volume_MA20']
            except Exception as e:
                print(f"Error calculating volume indicators: {e}")
                return None, None
            
            # Volatility and patterns
            try:
                feature_data['Volatility'] = feature_data['Returns'].rolling(window=20).std()
                feature_data['Volatility_Change'] = feature_data['Volatility'].pct_change()
                feature_data['Higher_Highs'] = (df['High'] > df['High'].shift(1)).astype(int)
                feature_data['Lower_Lows'] = (df['Low'] < df['Low'].shift(1)).astype(int)
                feature_data['Price_Trend'] = feature_data['Higher_Highs'] - feature_data['Lower_Lows']
            except Exception as e:
                print(f"Error calculating volatility and patterns: {e}")
                return None, None
            
            # Add features to dataframe
            for name, data in feature_data.items():
                df[name] = data
            
            # Target variable (next day's return)
            df['Target'] = df['Close'].shift(-1) / df['Close'] - 1
            
            # Drop missing values
            print("\nChecking for missing values...")
            before_drop = len(df)
            df = df.dropna()
            after_drop = len(df)
            print(f"Rows dropped due to NaN: {before_drop - after_drop}")
            
            if len(df) < 100:
                print(f"Insufficient data points: {len(df)} < 100")
                return None, None
            
            # Features for prediction
            self.features = [
                'Returns', 'Volume_Change', 'Price_Range',
                'RSI', 'Momentum', 'BB_Width', 'MACD_Hist',
                'Trend_5_20', 'Trend_20_50',
                'Volume_Trend', 'Volatility', 'Volatility_Change',
                'Price_Trend'
            ]
            
            print("\nFeature statistics:")
            X = df[self.features].values
            y = df['Target'].values
            
            for i, feature in enumerate(self.features):
                print(f"{feature}:")
                print(f"  Mean: {np.mean(X[:, i]):.4f}")
                print(f"  Std: {np.std(X[:, i]):.4f}")
                print(f"  Min: {np.min(X[:, i]):.4f}")
                print(f"  Max: {np.max(X[:, i]):.4f}")
            
            # Scale the features
            print("\nScaling features...")
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Verify scaling
            print("\nVerifying scaled data:")
            print("Shape:", X_scaled.shape)
            print("Mean:", np.mean(X_scaled))
            print("Std:", np.std(X_scaled))
            print("Min:", np.min(X_scaled))
            print("Max:", np.max(X_scaled))
            
            # Store the scaler
            self.scalers[symbol] = scaler
            
            return X_scaled, y
            
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            import traceback
            traceback.print_exc()
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
            
            if symbol not in self.scalers:
                print(f"No scaler found for {symbol}")
                return None
            
            if not hasattr(self, 'features') or not self.features:
                print("Feature list not initialized")
                return None
            
            print(f"Models available: {list(self.models[symbol].keys())}")
            print(f"Features required: {self.features}")
            
            # Get latest data
            print("\nFetching latest data...")
            stock = yf.Ticker(symbol)
            df = stock.history(period="60d")
            print(f"Fetched {len(df)} days of data")
            
            if df.empty:
                print(f"No historical data available for {symbol}")
                return None
            
            # Calculate features with validation
            print("\nCalculating prediction features...")
            try:
                # Basic features
                df['Returns'] = df['Close'].pct_change()
                df['Volume_Change'] = df['Volume'].pct_change()
                df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
                
                # Technical indicators
                df['RSI'] = self._calculate_rsi(df['Close'])
                df['Momentum'] = self._calculate_momentum(df['Close'])
                df['BB_Width'] = self._calculate_bollinger_bands(df['Close'])
                df['MACD_Hist'] = self._calculate_macd(df['Close'])
                
                # Moving averages
                df['MA5'] = df['Close'].rolling(window=5).mean()
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA50'] = df['Close'].rolling(window=50).mean()
                df['Trend_5_20'] = df['MA5'] - df['MA20']
                df['Trend_20_50'] = df['MA20'] - df['MA50']
                
                # Volume
                df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
                df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
                df['Volume_Trend'] = df['Volume_MA5'] - df['Volume_MA20']
                
                # Volatility
                df['Volatility'] = df['Returns'].rolling(window=20).std()
                df['Volatility_Change'] = df['Volatility'].pct_change()
                
                # Patterns
                df['Higher_Highs'] = (df['High'] > df['High'].shift(1)).astype(int)
                df['Lower_Lows'] = (df['Low'] < df['Low'].shift(1)).astype(int)
                df['Price_Trend'] = df['Higher_Highs'] - df['Lower_Lows']
                
            except Exception as e:
                print(f"Error calculating features: {e}")
                return None
            
            # Drop rows with missing values
            print("\nHandling missing values...")
            before_drop = len(df)
            df = df.dropna()
            after_drop = len(df)
            print(f"Rows dropped: {before_drop - after_drop}")
            
            if df.empty:
                print("No complete data available after feature calculation")
                return None
            
            # Get the last row
            last_row = df.iloc[-1:]
            print("\nFeature values for prediction:")
            for feature in self.features:
                print(f"{feature}: {last_row[feature].values[0]:.4f}")
            
            # Prepare features
            try:
                X = last_row[self.features].values
                print("\nInput shape:", X.shape)
                
                # Scale features
                scaler = self.scalers[symbol]
                X_scaled = scaler.transform(X)
                print("Scaled shape:", X_scaled.shape)
                print("Scaled values:", X_scaled[0])
                
            except Exception as e:
                print(f"Error preparing features: {e}")
                return None
            
            predictions = {}
            confidence_intervals = {}
            
            current_price = float(last_row['Close'].values[0])
            print(f"\nCurrent price: {current_price:.2f}")
            
            # Make predictions
            for model_name, model in self.models[symbol].items():
                try:
                    print(f"\nPredicting with {model_name}...")
                    
                    if model_name == 'rf':
                        # Random Forest prediction
                        tree_preds = []
                        for tree in model.estimators_:
                            try:
                                pred = tree.predict(X_scaled)[0]
                                if not np.isnan(pred):
                                    tree_preds.append(pred)
                            except Exception as e:
                                print(f"Tree prediction failed: {e}")
                                continue
                        
                        if not tree_preds:
                            print("No valid tree predictions")
                            continue
                            
                        return_pred = np.mean(tree_preds)
                        print(f"Return prediction: {return_pred:.4f}")
                        
                        if len(tree_preds) > 1:
                            std_err = stats.sem(tree_preds)
                            return_ci = stats.t.interval(0.95, len(tree_preds)-1, loc=return_pred, scale=std_err)
                        else:
                            return_ci = (return_pred, return_pred)
                        
                    else:  # Neural Network
                        # Multiple predictions for uncertainty
                        nn_preds = []
                        for _ in range(100):
                            try:
                                pred = model.predict(X_scaled, verbose=0)[0][0]
                                if not np.isnan(pred):
                                    nn_preds.append(pred)
                            except Exception as e:
                                print(f"NN prediction failed: {e}")
                                continue
                        
                        if not nn_preds:
                            print("No valid NN predictions")
                            continue
                            
                        return_pred = np.mean(nn_preds)
                        print(f"Return prediction: {return_pred:.4f}")
                        
                        if len(nn_preds) > 1:
                            std_err = stats.sem(nn_preds)
                            return_ci = stats.t.interval(0.95, len(nn_preds)-1, loc=return_pred, scale=std_err)
                        else:
                            return_ci = (return_pred, return_pred)
                    
                    # Convert return prediction to price
                    pred_price = current_price * (1 + return_pred)
                    ci_prices = (current_price * (1 + return_ci[0]), current_price * (1 + return_ci[1]))
                    
                    print(f"Price prediction: {pred_price:.2f}")
                    print(f"Confidence interval: ({ci_prices[0]:.2f}, {ci_prices[1]:.2f})")
                    
                    predictions[model_name] = float(pred_price)
                    confidence_intervals[model_name] = (float(ci_prices[0]), float(ci_prices[1]))
                    
                except Exception as e:
                    print(f"Error in {model_name} prediction: {e}")
                    continue
            
            if not predictions:
                print("No successful predictions made")
                return None
            
            print("\nFinal predictions:", predictions)
            print("Confidence intervals:", confidence_intervals)
            
            self.confidence_intervals[symbol] = confidence_intervals
            return predictions
            
        except Exception as e:
            print(f"Error in prediction process: {str(e)}")
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
