from flask_login import LoginManager
from werkzeug.security import generate_password_hash, check_password_hash
from models import User, db

login_manager = LoginManager()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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
            
            return True, "Registration successful"
        except Exception as e:
            db.session.rollback()
            return False, str(e)
            
    @staticmethod
    def login_user(username, password):
        """Login a user"""
        try:
            user = User.query.filter_by(username=username).first()
            
            if user and check_password_hash(user.password_hash, password):
                return True, user
                
            return False, "Invalid username or password"
        except Exception as e:
            return False, str(e)
            
    @staticmethod
    def change_password(user, old_password, new_password):
        """Change user password"""
        try:
            if not check_password_hash(user.password_hash, old_password):
                return False, "Invalid current password"
                
            user.password_hash = generate_password_hash(new_password)
            db.session.commit()
            
            return True, "Password changed successfully"
        except Exception as e:
            db.session.rollback()
            return False, str(e)
            
    @staticmethod
    def reset_password_request(email):
        """Request password reset"""
        try:
            user = User.query.filter_by(email=email).first()
            if not user:
                return False, "Email not found"
                
            # In a real application, you would:
            # 1. Generate a reset token
            # 2. Send reset email
            # For now, we'll just return success
            return True, "Password reset instructions sent to email"
        except Exception as e:
            return False, str(e)
