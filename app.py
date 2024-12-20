from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
from flask_login import LoginManager, login_required, current_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from config import config
from models import db, User
from services.auth import auth_bp
from services.portfolio import PortfolioService
from services.technical_analysis import TechnicalAnalysisService
from services.education import EducationService
from services.export import ExportService
from services.prediction import PredictionService
import logging
import os
import yfinance as yf
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)

# Load configuration
env = os.environ.get('FLASK_ENV', 'production')  # Default to production
app.config.from_object(config[env])

# Set secret key
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Database configuration
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables
with app.app_context():
    try:
        db.create_all()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")

# Register blueprints
app.register_blueprint(auth_bp)

# Initialize services
technical_analysis = TechnicalAnalysisService()
prediction_service = PredictionService()  # Now works without data
portfolio_service = PortfolioService()
education_service = EducationService()
export_service = ExportService()

# API Routes
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

@app.route('/api/add_stock', methods=['POST'])
def add_stock():
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    
    try:
        # Verify stock exists
        stock = yf.Ticker(symbol)
        info = stock.info
        
        if not info or 'regularMarketPrice' not in info:
            return jsonify({
                'success': False,
                'message': 'Invalid stock symbol'
            }), 400
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'name': info.get('shortName', symbol)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/stocks')
def get_stocks():
    # For now, return a list of popular tech stocks
    stocks = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
        {'symbol': 'META', 'name': 'Meta Platforms Inc.'}
    ]
    return jsonify(stocks)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/portfolio')
@login_required
def portfolio():
    return render_template('portfolio.html')

@app.route('/watchlist')
@login_required
def watchlist():
    return render_template('watchlist.html')

@app.route('/education')
def education():
    return render_template('education.html')

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    return render_template('change_password.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/api/portfolio')
@login_required
def get_portfolio():
    try:
        portfolio = portfolio_service.get_portfolio(current_user.id)
        return jsonify(portfolio)
    except Exception as e:
        logger.error(f"Error fetching portfolio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/add', methods=['POST'])
@login_required
def add_to_portfolio():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        shares = data.get('shares', 1)
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
            
        # Verify stock exists
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
        except:
            return jsonify({'error': 'Invalid stock symbol'}), 400
            
        # Add to portfolio
        portfolio_service.add_stock(current_user.id, symbol, shares)
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Error adding to portfolio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/education/indicator/<name>')
def get_indicator_info(name):
    try:
        info = education_service.get_indicator_info(name)
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error fetching indicator info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/portfolio')
@login_required
def export_portfolio():
    try:
        format_type = request.args.get('format', 'csv')
        data = portfolio_service.get_portfolio(current_user.id)
        
        if format_type == 'pdf':
            return export_service.export_pdf(data)
        elif format_type == 'excel':
            return export_service.export_excel(data)
        else:
            return export_service.export_csv(data)
            
    except Exception as e:
        logger.error(f"Error exporting portfolio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Get port from environment variable for Render compatibility
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
