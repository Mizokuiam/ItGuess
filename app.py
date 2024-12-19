from flask import Flask, render_template, jsonify
from utils.data_processor import DataProcessor
from utils.model_trainer import ModelTrainer
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize our utility classes
data_processor = DataProcessor()
model_trainer = ModelTrainer()

# List of supported stock symbols
SUPPORTED_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA']

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', symbols=SUPPORTED_SYMBOLS)

@app.route('/api/history/<symbol>')
def get_history(symbol):
    """Get historical data and technical indicators for a symbol"""
    try:
        # Validate symbol
        symbol = symbol.upper()
        if symbol not in SUPPORTED_SYMBOLS:
            return jsonify({'error': 'Invalid symbol'}), 400
        
        # Fetch historical data
        stock_data = data_processor.fetch_stock_data(symbol)
        if stock_data is None:
            return jsonify({'error': 'Failed to fetch stock data'}), 500
        
        # Calculate technical indicators
        indicators = data_processor.calculate_indicators(stock_data)
        
        # Prepare response data
        response = {
            'dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
            'prices': stock_data['Close'].tolist(),
            'technical_indicators': {
                'sma20': indicators['SMA20'].tolist(),
                'sma50': indicators['SMA50'].tolist(),
                'rsi': indicators['RSI'].tolist(),
                'upper_band': indicators['Upper Band'].tolist(),
                'lower_band': indicators['Lower Band'].tolist()
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error getting history for {symbol}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/predict/<symbol>')
def predict_price(symbol):
    """Get price prediction for a symbol"""
    try:
        # Validate symbol
        symbol = symbol.upper()
        if symbol not in SUPPORTED_SYMBOLS:
            return jsonify({'error': 'Invalid symbol'}), 400
        
        # Fetch latest data
        stock_data = data_processor.fetch_stock_data(symbol)
        if stock_data is None:
            return jsonify({'error': 'Failed to fetch stock data'}), 500
        
        # Prepare features for prediction
        features = data_processor.prepare_features(stock_data)
        
        # Load or train model if needed
        model = model_trainer.load_model(symbol)
        if model is None:
            model, metrics = model_trainer.train_random_forest(
                features.iloc[:-1], 
                stock_data['Close'].iloc[1:],
                symbol
            )
            if model is None:
                return jsonify({'error': 'Failed to train model'}), 500
        
        # Make prediction
        latest_features = features.iloc[-1:]
        prediction = model.predict(latest_features)[0]
        
        response = {
            'symbol': symbol,
            'current_price': stock_data['Close'].iloc[-1],
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error predicting price for {symbol}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
