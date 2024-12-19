from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# List of supported stock symbols
SUPPORTED_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA']

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
        # Calculate SMA
        df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
        
        # Calculate RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Calculate Bollinger Bands
        df['Upper Band'] = ta.volatility.bollinger_hband(df['Close'])
        df['Lower Band'] = ta.volatility.bollinger_lband(df['Close'])
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return None

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
        
        # Simple prediction using last 5 days average change
        last_5_days = stock_data['Close'].tail(5)
        avg_change = (last_5_days.pct_change() + 1).mean()
        prediction = current_price * avg_change
        
        response = {
            'current_price': float(current_price),
            'prediction': float(prediction),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in predict_price: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# For local development
if __name__ == '__main__':
    app.run(debug=True)

# For Vercel
app = app
