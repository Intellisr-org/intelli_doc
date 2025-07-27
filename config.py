"""
Configuration settings for the Flask web application
"""

import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', '1') == '1'
    
    # File upload settings
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'outputs'
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    
    # Model settings
    YOLO_MODEL_PATH = 'weights/best.pt'
    DEVICE = 'mps'  # 'cuda', 'cpu', or 'mps' for Apple Silicon
    
    # Processing settings
    DEFAULT_LANGUAGE = 'si'
    DEFAULT_DPI = 300
    DEFAULT_CONFIDENCE = 75.0
    
    # Performance settings
    RECOGNITION_BATCH_SIZE = int(os.environ.get('RECOGNITION_BATCH_SIZE', '256'))
    DETECTOR_BATCH_SIZE = int(os.environ.get('DETECTOR_BATCH_SIZE', '18'))
    ORDER_BATCH_SIZE = int(os.environ.get('ORDER_BATCH_SIZE', '16'))
    RECOGNITION_STATIC_CACHE = os.environ.get('RECOGNITION_STATIC_CACHE', 'true').lower() == 'true'
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Create necessary directories
        Path(Config.UPLOAD_FOLDER).mkdir(exist_ok=True)
        Path(Config.OUTPUT_FOLDER).mkdir(exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Override with environment variables for production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DEVICE = os.environ.get('DEVICE', 'cpu')

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    WTF_CSRF_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 