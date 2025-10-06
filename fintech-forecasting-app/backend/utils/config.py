"""
Configuration module for the FinTech Forecasting Application.
Contains all application settings and environment variable management.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration class."""
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'mongodb://localhost:27017/fintech_forecasting')
    MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'fintech_forecasting')
    
    # API Configuration
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', 'demo')
    
    # Flask Configuration
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
    
    # ML Model Configuration
    MODEL_UPDATE_INTERVAL = int(os.getenv('MODEL_UPDATE_INTERVAL', 3600))  # seconds
    FORECAST_HORIZON_HOURS = int(os.getenv('FORECAST_HORIZON_HOURS', 72))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    EPOCHS = int(os.getenv('EPOCHS', 50))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')
    
    # Data Collection Configuration
    DATA_COLLECTION_SYMBOLS = os.getenv(
        'DATA_COLLECTION_SYMBOLS', 
        'AAPL,GOOGL,MSFT,TSLA,BTC,ETH'
    ).split(',')
    
    # Model File Paths
    MODELS_DIR = 'models'
    ARIMA_MODEL_PATH = os.path.join(MODELS_DIR, 'arima')
    LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm.h5')
    TRANSFORMER_MODEL_PATH = os.path.join(MODELS_DIR, 'transformer.h5')
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate critical configuration values."""
        required_vars = ['FINNHUB_API_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not getattr(cls, var) or getattr(cls, var) == 'demo':
                missing_vars.append(var)
        
        if missing_vars:
            print(f"WARNING: Missing or default configuration: {missing_vars}")
            return False
        
        return True
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories for the application."""
        directories = [
            cls.MODELS_DIR,
            os.path.dirname(cls.LOG_FILE) if cls.LOG_FILE else 'logs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Create directories on import
Config.create_directories()
