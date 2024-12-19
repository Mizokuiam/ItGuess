from models import db
from flask import Flask

def init_db(app):
    """Initialize the database"""
    with app.app_context():
        # Create all tables
        db.create_all()
        
        print("Database initialized successfully")
