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
        """Initialize prediction service"""
        self.models = {}
        self.scalers = {}
        self.history = {}
        self.confidence_intervals = {}
        self.feature_importance = {}
        self.features = []
        
        # Print dependency versions
        try:
            import sklearn
            import yfinance
            import tensorflow as tf
            print("\nDependency Versions:")
            print(f"scikit-learn: {sklearn.__version__}")
            print(f"yfinance: {yfinance.__version__}")
            print(f"tensorflow: {tf.__version__}")
        except Exception as e:
            print(f"Error checking versions: {e}")
    
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

    def _validate_model(self, symbol):
        """Validate model and scaler for a symbol"""
        if symbol not in self.models:
            print(f"No trained models found for {symbol}")
            return False
            
        if symbol not in self.scalers:
            print(f"No scaler found for {symbol}")
            return False
            
        if not hasattr(self, 'features') or not self.features:
            print("Feature list not initialized")
            return False
            
        print("\nModel Validation:")
        print(f"Models available: {list(self.models[symbol].keys())}")
        print(f"Features required: {self.features}")
        print(f"Scaler available: {type(self.scalers[symbol]).__name__}")
        
        return True
    
    def _validate_data(self, df, symbol, purpose=""):
        """Validate dataframe for training or prediction"""
        print(f"\nValidating data for {purpose}...")
        
        if df is None or df.empty:
            print("No data available")
            return False
            
        print(f"Data shape: {df.shape}")
        print("\nNull values:")
        null_counts = df.isnull().sum()
        print(null_counts[null_counts > 0])
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
            
        print("\nData sample:")
        print(df.head())
        
        return True
    
    def _validate_features(self, features_df, expected_features):
        """Validate features for prediction"""
        print("\nValidating features...")
        
        missing_features = [f for f in expected_features if f not in features_df.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            return False
            
        print("\nFeature statistics:")
        stats = features_df[expected_features].describe()
        print(stats)
        
        # Check for infinite values
        inf_mask = np.isinf(features_df[expected_features])
        inf_counts = inf_mask.sum()
        if inf_counts.any():
            print("\nInfinite values found:")
            print(inf_counts[inf_counts > 0])
            return False
            
        return True
    
    def _validate_predictions(self, predictions, current_price):
        """Validate prediction outputs"""
        print("\nValidating predictions...")
        
        if not predictions:
            print("No predictions generated")
            return False
            
        for model_name, pred in predictions.items():
            print(f"\n{model_name} prediction:")
            print(f"Current price: {current_price:.2f}")
            print(f"Predicted price: {pred:.2f}")
            print(f"Predicted change: {((pred/current_price - 1) * 100):.2f}%")
            
            # Check for unreasonable predictions
            if pred <= 0 or pred > current_price * 2:
                print(f"Warning: Prediction seems unreasonable")
                return False
                
            if not np.isfinite(pred):
                print(f"Warning: Invalid prediction value")
                return False
                
        return True

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
    
    def predict(self, symbol):
        """Generate predictions for a given stock symbol"""
        try:
            print(f"\nGenerating predictions for {symbol}")
            technical_prediction = self.predict_technical(symbol)
            
            if technical_prediction is None:
                print("Technical prediction failed")
                return None
                
            return {
                'technical': technical_prediction
            }
            
        except Exception as e:
            print(f"Error in predict(): {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def predict_technical(self, symbol, period='1d'):
        """Make predictions based on technical analysis"""
        try:
            print(f"\nAnalyzing {symbol} using technical indicators...")
            
            # Fetch historical data
            stock = yf.Ticker(symbol)
            df = stock.history(period="3mo")  # Fetch 3 months of data
            
            print(f"Initial data fetch: {len(df)} days")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            # Filter to last 60 days
            df = df.last('60D')
            print(f"After filtering: {len(df)} days")
            
            if df.empty:
                print("No data available")
                return None
                
            if len(df) < 20:
                print(f"Insufficient data: only {len(df)} days available")
                return None
                
            print(f"Using {len(df)} days of data")
            print("Data columns:", df.columns.tolist())
            
            current_price = float(df['Close'].iloc[-1])
            print(f"Current price: ${current_price:.2f}")
            
            # Calculate technical indicators with error checking
            try:
                # RSI
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = float(rsi.iloc[-1])
                print(f"RSI: {current_rsi:.2f}")
                
                # MACD
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                current_macd = float(macd.iloc[-1])
                current_signal = float(signal.iloc[-1])
                print(f"MACD: {current_macd:.2f}, Signal: {current_signal:.2f}")
                
                # Moving Averages
                ma5 = df['Close'].rolling(window=5).mean()
                ma20 = df['Close'].rolling(window=20).mean()
                current_ma5 = float(ma5.iloc[-1])
                current_ma20 = float(ma20.iloc[-1])
                print(f"MA5: {current_ma5:.2f}, MA20: {current_ma20:.2f}")
                
                # Price Momentum
                momentum = df['Close'].diff(5)
                current_momentum = float(momentum.iloc[-1])
                print(f"Momentum: {current_momentum:.2f}")
                
                # Volume Analysis
                volume_ma = df['Volume'].rolling(window=20).mean()
                current_volume = float(df['Volume'].iloc[-1])
                volume_ratio = current_volume / float(volume_ma.iloc[-1])
                print(f"Volume ratio: {volume_ratio:.2f}")
                
                # Bollinger Bands
                ma20 = df['Close'].rolling(window=20).mean()
                std20 = df['Close'].rolling(window=20).std()
                upper_band = ma20 + (std20 * 2)
                lower_band = ma20 - (std20 * 2)
                current_upper = float(upper_band.iloc[-1])
                current_lower = float(lower_band.iloc[-1])
                print(f"Bollinger Bands - Upper: {current_upper:.2f}, Lower: {current_lower:.2f}")
                
                # Generate prediction based on technical indicators
                prediction = current_price  # Start with current price
                
                # RSI signals
                if current_rsi > 70:  # Overbought
                    prediction *= 0.99
                elif current_rsi < 30:  # Oversold
                    prediction *= 1.01
                
                # MACD signals
                if current_macd > current_signal:  # Bullish
                    prediction *= 1.01
                else:  # Bearish
                    prediction *= 0.99
                
                # Moving Average signals
                if current_ma5 > current_ma20:  # Bullish
                    prediction *= 1.01
                else:  # Bearish
                    prediction *= 0.99
                
                # Momentum signals
                if current_momentum > 0:  # Positive momentum
                    prediction *= 1.01
                else:  # Negative momentum
                    prediction *= 0.99
                
                # Volume signals
                if volume_ratio > 1.5:  # High volume
                    if current_price > current_ma5:  # High volume + price above MA5
                        prediction *= 1.01
                    else:
                        prediction *= 0.99
                
                # Bollinger Band signals
                bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
                if bb_position > 0.8:  # Near upper band
                    prediction *= 0.99
                elif bb_position < 0.2:  # Near lower band
                    prediction *= 1.01
                
                print(f"Final prediction: ${prediction:.2f}")
                return prediction
                
            except Exception as e:
                print(f"Error calculating technical indicators: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
            
        except Exception as e:
            print(f"Error in predict_technical(): {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def get_confidence_intervals(self, symbol):
        """Calculate confidence intervals for predictions"""
        try:
            print(f"\nCalculating confidence intervals for {symbol}")
            
            # Get predictions
            predictions = self.predict(symbol)
            if predictions is None:
                print("No predictions available for confidence intervals")
                return None
            
            technical_pred = predictions['technical']
            
            # Calculate intervals (using a simple percentage-based approach)
            lower_bound = technical_pred * 0.98  # 2% below prediction
            upper_bound = technical_pred * 1.02  # 2% above prediction
            
            return {
                symbol: {
                    'technical': (lower_bound, upper_bound)
                }
            }
            
        except Exception as e:
            print(f"Error calculating confidence intervals: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def get_prediction_history(self, symbol):
        """Get prediction history"""
        return self.history.get(symbol, None)
    
    def get_feature_importance(self, symbol):
        """Get feature importance scores"""
        return self.feature_importance.get(symbol, None)
