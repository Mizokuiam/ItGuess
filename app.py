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
import logging
import os

# Initialize Flask app
app = Flask(__name__)

# Load configuration
env = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[env])

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

# Register blueprints
app.register_blueprint(auth_bp)

# Initialize services
portfolio_service = PortfolioService()
technical_service = TechnicalAnalysisService()
education_service = EducationService()
export_service = ExportService()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stock/<symbol>')
def get_stock_data(symbol):
    try:
        data = technical_service.get_stock_data(symbol)
        if not data:
            return jsonify({'error': 'Failed to fetch stock data'}), 400
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<symbol>')
def predict_stock(symbol):
    try:
        predictions = technical_service.predict_price(symbol)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio', methods=['GET'])
@login_required
def get_portfolio():
    try:
        portfolio_data = portfolio_service.get_portfolio_value(current_user.id)
        return jsonify(portfolio_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/add', methods=['POST'])
@login_required
def add_to_portfolio():
    try:
        data = request.json
        success, result = portfolio_service.add_position(
            current_user.id,
            data['symbol'],
            data['shares'],
            data['entry_price']
        )
        if success:
            return jsonify({'message': 'Position added successfully'})
        return jsonify({'error': result}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/education/indicator/<name>')
def get_indicator_info(name):
    try:
        info = education_service.get_indicator_info(name)
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/portfolio', methods=['POST'])
@login_required
def export_portfolio():
    try:
        format_type = request.json.get('format', 'pdf')
        portfolio_data = portfolio_service.get_portfolio_value(current_user.id)
        
        if format_type == 'pdf':
            success, buffer = export_service.generate_portfolio_report(portfolio_data, current_user)
        else:
            success, buffer = export_service.export_to_csv(portfolio_data, 'portfolio.csv')
            
        if not success:
            return jsonify({'error': 'Failed to generate report'}), 500
            
        return buffer
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
