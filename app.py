from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
from flask_login import LoginManager, login_required, current_user
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
env = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[env])

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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register blueprints
app.register_blueprint(auth_bp)

# Initialize services
technical_analysis = TechnicalAnalysisService()
prediction_service = PredictionService()  # Now works without data
portfolio_service = PortfolioService()
education_service = EducationService()
export_service = ExportService()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stock/<symbol>')
def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Get historical data
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return jsonify({'error': 'No data available for this stock'}), 404
            
        # Calculate technical indicators
        indicators = technical_analysis.calculate_indicators(df)
        df = df.join(indicators)
        
        # Format the data for response
        data = {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'prices': df['Close'].tolist(),
            'volumes': df['Volume'].tolist(),
            'indicators': {
                'RSI': df['RSI'].tolist() if 'RSI' in df.columns else [],
                'MACD': df['MACD'].tolist() if 'MACD' in df.columns else [],
                'Signal': df['Signal'].tolist() if 'Signal' in df.columns else [],
                'K': df['%K'].tolist() if '%K' in df.columns else [],
                'D': df['%D'].tolist() if '%D' in df.columns else []
            }
        }
        
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<symbol>')
def predict_stock(symbol):
    try:
        stock = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Get historical data
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return jsonify({'error': 'No data available for this stock'}), 404
            
        # Calculate technical indicators
        indicators = technical_analysis.calculate_indicators(df)
        df = df.join(indicators)
        
        # Set data for prediction
        prediction_service.set_data(df)
        
        # Make prediction
        predicted_price = prediction_service.make_prediction()
        
        return jsonify({
            'predicted_price': predicted_price,
            'current_price': df['Close'].iloc[-1],
            'prediction_date': datetime.now().strftime('%Y-%m-%d')
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/indicators/<symbol>')
def get_indicators(symbol):
    try:
        stock = yf.Ticker(symbol)
        history = stock.history(period='1y')
        indicators = technical_analysis.calculate_indicators(history)
        return jsonify(indicators)
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
