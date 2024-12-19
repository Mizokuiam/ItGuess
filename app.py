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
portfolio_service = PortfolioService()
education_service = EducationService()
export_service = ExportService()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stock/<symbol>')
@login_required
def get_stock_data(symbol):
    try:
        analysis_service = TechnicalAnalysisService(symbol=symbol)
        data = analysis_service.calculate_all_indicators()
        if data is None:
            return jsonify({'error': 'Unable to fetch stock data'}), 400
            
        summary = analysis_service.get_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error getting stock data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<symbol>')
@login_required
def predict_stock(symbol):
    try:
        analysis_service = TechnicalAnalysisService(symbol=symbol)
        data = analysis_service.calculate_all_indicators()
        if data is None:
            return jsonify({'error': 'Unable to fetch stock data'}), 400
            
        prediction_service = PredictionService(data)
        prediction = prediction_service.make_prediction()
        factors = prediction_service.get_prediction_factors()
        
        return jsonify({
            'prediction': prediction,
            'factors': factors
        })
    except Exception as e:
        logger.error(f"Error predicting stock: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio')
@login_required
def get_portfolio():
    try:
        portfolio = portfolio_service.get_portfolio(current_user.id)
        return jsonify(portfolio)
    except Exception as e:
        logger.error(f"Error getting portfolio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/add', methods=['POST'])
@login_required
def add_to_portfolio():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        shares = data.get('shares')
        price = data.get('price')
        
        if not all([symbol, shares, price]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        result = portfolio_service.add_position(
            user_id=current_user.id,
            symbol=symbol,
            shares=shares,
            price=price
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error adding to portfolio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/education/indicator/<name>')
def get_indicator_info(name):
    try:
        info = education_service.get_indicator_info(name)
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting indicator info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/portfolio')
@login_required
def export_portfolio():
    try:
        format_type = request.args.get('format', 'csv')
        if format_type not in ['csv', 'pdf']:
            return jsonify({'error': 'Invalid export format'}), 400
            
        portfolio = portfolio_service.get_portfolio(current_user.id)
        if format_type == 'csv':
            file_path = export_service.export_to_csv(portfolio)
        else:
            file_path = export_service.export_to_pdf(portfolio)
            
        return jsonify({
            'success': True,
            'file_path': file_path
        })
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
