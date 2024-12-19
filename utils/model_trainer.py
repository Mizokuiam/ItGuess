import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def train_random_forest(self, X, y, symbol):
        """Train a Random Forest model"""
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Create and train the model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate the model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Save the model
            model_path = os.path.join(self.model_dir, f'{symbol}_rf_model.joblib')
            joblib.dump(model, model_path)
            
            metrics = {
                'train_score': train_score,
                'test_score': test_score,
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
            
            return model, metrics
        
        except Exception as e:
            logger.error(f"Error training Random Forest model: {str(e)}")
            return None, None
    
    def train_lstm(self, X, y, symbol, sequence_length=60):
        """Train an LSTM model"""
        try:
            # Prepare sequences for LSTM
            X_seq = []
            y_seq = []
            
            for i in range(len(X) - sequence_length):
                X_seq.append(X[i:(i + sequence_length)])
                y_seq.append(y[i + sequence_length])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq, test_size=0.2, shuffle=False
            )
            
            # Create LSTM model
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(sequence_length, X.shape[1]), return_sequences=True),
                Dropout(0.2),
                LSTM(50, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.1,
                verbose=0
            )
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Save the model
            model_path = os.path.join(self.model_dir, f'{symbol}_lstm_model')
            model.save(model_path)
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'history': history.history
            }
            
            return model, metrics
        
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            return None, None
    
    def load_model(self, symbol, model_type='rf'):
        """Load a trained model"""
        try:
            if model_type == 'rf':
                model_path = os.path.join(self.model_dir, f'{symbol}_rf_model.joblib')
                return joblib.load(model_path)
            elif model_type == 'lstm':
                model_path = os.path.join(self.model_dir, f'{symbol}_lstm_model')
                return tf.keras.models.load_model(model_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
