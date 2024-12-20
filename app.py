from flask import Flask, render_template, jsonify, request
import logging
import os
import yfinance as yf
from datetime import datetime, timedelta
from services.technical_analysis import TechnicalAnalysisService
from services.prediction import PredictionService

# Initialize Flask app
app = Flask(__name__)

# Set secret key
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
technical_analysis = TechnicalAnalysisService()
prediction_service = PredictionService()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/education')
def education():
    return render_template('education.html')

@app.route('/api/search_stock')
def search_stock():
    query = request.args.get('query', '').upper()
    try:
        # Use yfinance to search for stocks
        matches = []
        if query:
            # This is a simple implementation. You might want to use a proper stock API for production
            common_stocks = {
                'AAPL': 'Apple Inc.',
                'GOOGL': 'Alphabet Inc.',
                'MSFT': 'Microsoft Corporation',
                'AMZN': 'Amazon.com Inc.',
                'META': 'Meta Platforms Inc.',
                'TSLA': 'Tesla Inc.',
                'NVDA': 'NVIDIA Corporation',
                'JPM': 'JPMorgan Chase & Co.',
                'JNJ': 'Johnson & Johnson',
                'V': 'Visa Inc.'
            }
            
            matches = [
                {'symbol': symbol, 'name': name}
                for symbol, name in common_stocks.items()
                if query in symbol or query.lower() in name.lower()
            ]
            
        return jsonify({'results': matches})
    except Exception as e:
        logger.error(f"Error searching stocks: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>')
def get_stock_data(symbol):
    period = request.args.get('period', '1d')
    
    try:
        # Get stock data using yfinance
        stock = yf.Ticker(symbol)
        hist = stock.history(period='1mo')
        
        if hist.empty:
            return jsonify({'error': 'No data available for this stock'}), 404
        
        # Get current stock info
        info = stock.info
        current_price = info.get('regularMarketPrice', hist['Close'].iloc[-1])
        previous_close = info.get('previousClose', hist['Close'].iloc[-2])
        price_change = ((current_price - previous_close) / previous_close) * 100
        
        # Format historical data for chart
        prices = []
        for date, row in hist.iterrows():
            prices.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': round(row['Close'], 2)
            })
        
        # Get technical indicators
        rsi = technical_analysis.calculate_rsi(hist['Close'])
        macd, signal = technical_analysis.calculate_macd(hist['Close'])
        
        # Calculate predictions
        prediction_data = prediction_service.get_prediction(symbol, period)
        
        return jsonify({
            'success': True,
            'prices': prices,
            'current': {
                'price': current_price,
                'change': price_change,
                'volume': info.get('volume', hist['Volume'].iloc[-1])
            },
            'predicted': prediction_data,
            'indicators': {
                'RSI': f"{rsi:.2f}",
                'MACD': f"{macd:.2f}",
                'Signal': f"{signal:.2f}"
            }
        })
    
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
