from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load supported symbols from a JSON file
SYMBOLS_FILE = 'supported_symbols.json'

def load_symbols():
    try:
        if os.path.exists(SYMBOLS_FILE):
            with open(SYMBOLS_FILE, 'r') as f:
                return json.load(f)
        return ['AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA']
    except Exception as e:
        logger.error(f"Error loading symbols: {str(e)}")
        return ['AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA']

def save_symbols(symbols):
    try:
        with open(SYMBOLS_FILE, 'w') as f:
            json.dump(symbols, f)
    except Exception as e:
        logger.error(f"Error saving symbols: {str(e)}")

# List of supported stock symbols
SUPPORTED_SYMBOLS = load_symbols()

def fetch_stock_data(symbol, period='1y'):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_indicators(df):
    """Calculate technical indicators"""
    try:
        if df is None or len(df) == 0:
            return None
            
        # Calculate SMA
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['Upper Band'] = rolling_mean + (rolling_std * 2)
        df['Lower Band'] = rolling_mean - (rolling_std * 2)
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return None

def simple_moving_average_prediction(prices, window=5):
    """Simple prediction using moving average"""
    if len(prices) < window:
        return prices.iloc[-1]
    return prices.rolling(window=window).mean().iloc[-1]

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', symbols=SUPPORTED_SYMBOLS)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/add_stock', methods=['POST'])
def add_stock():
    """Add a new stock symbol"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        
        if not symbol:
            return jsonify({'success': False, 'error': 'No symbol provided'}), 400
            
        if symbol in SUPPORTED_SYMBOLS:
            return jsonify({'success': False, 'error': 'Symbol already exists'}), 400
        
        # Verify the symbol exists
        stock = yf.Ticker(symbol)
        info = stock.info
        
        if not info or 'regularMarketPrice' not in info:
            return jsonify({'success': False, 'error': 'Invalid symbol'}), 400
        
        # Add to supported symbols
        SUPPORTED_SYMBOLS.append(symbol)
        save_symbols(SUPPORTED_SYMBOLS)
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error adding stock: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/history/<symbol>')
def get_history(symbol):
    """Get historical data and technical indicators for a symbol"""
    try:
        # Validate symbol
        symbol = symbol.upper()
        if symbol not in SUPPORTED_SYMBOLS:
            return jsonify({'error': 'Invalid symbol'}), 400
        
        # Fetch historical data
        stock_data = fetch_stock_data(symbol)
        if stock_data is None:
            return jsonify({'error': 'Failed to fetch stock data'}), 500
        
        # Calculate technical indicators
        stock_data = calculate_indicators(stock_data)
        if stock_data is None:
            return jsonify({'error': 'Failed to calculate indicators'}), 500
        
        # Prepare response data
        response = {
            'dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
            'prices': stock_data['Close'].tolist(),
            'technical_indicators': {
                'sma20': stock_data['SMA20'].fillna(0).tolist(),
                'sma50': stock_data['SMA50'].fillna(0).tolist(),
                'rsi': stock_data['RSI'].fillna(0).tolist(),
                'upper_band': stock_data['Upper Band'].fillna(0).tolist(),
                'lower_band': stock_data['Lower Band'].fillna(0).tolist()
            }
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in get_history: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/predict/<symbol>')
def predict_price(symbol):
    """Get simple price prediction for a symbol"""
    try:
        # Validate symbol
        symbol = symbol.upper()
        if symbol not in SUPPORTED_SYMBOLS:
            return jsonify({'error': 'Invalid symbol'}), 400
        
        # Fetch historical data
        stock_data = fetch_stock_data(symbol, period='1mo')
        if stock_data is None:
            return jsonify({'error': 'Failed to fetch stock data'}), 500
        
        # Get the current price
        current_price = stock_data['Close'].iloc[-1]
        
        # Simple prediction using moving average
        prediction = simple_moving_average_prediction(stock_data['Close'])
        
        response = {
            'current_price': float(current_price),
            'prediction': float(prediction),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in predict_price: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
