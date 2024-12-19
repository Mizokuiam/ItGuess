from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class PredictionService:
    def __init__(self, data=None):
        self.data = data
        self.scaler = MinMaxScaler()
        self.model = None
        
    def set_data(self, data):
        """Set data for prediction"""
        self.data = data
        
    def prepare_data(self, window_size=60):
        """Prepare data for prediction"""
        if self.data is None:
            raise ValueError("Data must be set before preparing")
            
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
        if self.data is None:
            raise ValueError("Data must be set before training")
            
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
        if self.data is None:
            raise ValueError("Data must be set before making predictions")
            
        if self.model is None:
            self.train_model()
            
        # Prepare last sequence
        X_last = self.prepare_data()[0][-1:]
        
        # Make prediction
        prediction = self.model.predict(X_last.reshape(1, -1))
        
        return float(prediction[0])
