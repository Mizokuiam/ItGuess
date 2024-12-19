from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import User, db

# Create Blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

login_manager = LoginManager()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Blueprint routes
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember_me')
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=bool(remember))
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        flash('Invalid username or password', 'error')
    
    return render_template('auth/login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('auth/register.html')
            
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return render_template('auth/register.html')
            
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('auth.login'))
        
    return render_template('auth/register.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@auth_bp.route('/profile')
@login_required
def profile():
    return render_template('auth/profile.html')

@auth_bp.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        
        if not check_password_hash(current_user.password_hash, current_password):
            flash('Current password is incorrect', 'error')
            return render_template('auth/change_password.html')
            
        current_user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        
        flash('Password updated successfully', 'success')
        return redirect(url_for('auth.profile'))
        
    return render_template('auth/change_password.html')

# Service class for additional authentication functionality
class AuthService:
    @staticmethod
    def register_user(username, email, password):
        """Register a new user"""
        try:
            # Check if user already exists
            if User.query.filter_by(username=username).first():
                return False, "Username already exists"
            if User.query.filter_by(email=email).first():
                return False, "Email already exists"
                
            # Create new user
            user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password)
            )
            
            db.session.add(user)
            db.session.commit()
            
            return True, user
        except Exception as e:
            db.session.rollback()
            return False, str(e)
            
    @staticmethod
    def authenticate_user(username, password):
        """Authenticate a user"""
        try:
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password_hash, password):
                return True, user
            return False, "Invalid username or password"
        except Exception as e:
            return False, str(e)
            
    @staticmethod
    def change_user_password(user_id, current_password, new_password):
        """Change user password"""
        try:
            user = User.query.get(user_id)
            if not user:
                return False, "User not found"
                
            if not check_password_hash(user.password_hash, current_password):
                return False, "Current password is incorrect"
                
            user.password_hash = generate_password_hash(new_password)
            db.session.commit()
            
            return True, "Password updated successfully"
        except Exception as e:
            db.session.rollback()
            return False, str(e)
            
    @staticmethod
    def update_user_profile(user_id, data):
        """Update user profile"""
        try:
            user = User.query.get(user_id)
            if not user:
                return False, "User not found"
                
            for key, value in data.items():
                if hasattr(user, key) and key != 'id':
                    setattr(user, key, value)
                    
            db.session.commit()
            return True, user
        except Exception as e:
            db.session.rollback()
            return False, str(e)
