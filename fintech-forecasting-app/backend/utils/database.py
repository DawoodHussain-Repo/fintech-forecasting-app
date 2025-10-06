"""
Database models and MongoDB connection management.
Defines the data schema for financial data storage and forecasting results.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
import logging
from utils.config import Config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """MongoDB database connection and operations manager."""
    
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.connect()
    
    def connect(self):
        """Establish MongoDB connection."""
        try:
            self.client = MongoClient(Config.DATABASE_URL)
            self.db = self.client[Config.MONGODB_DB_NAME]
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("Connected to MongoDB successfully")
            
            # Create indexes
            self.create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def create_indexes(self):
        """Create database indexes for optimal query performance."""
        if self.db is None:
            return
        
        try:
            # Historical prices indexes
            self.db.historical_prices.create_index([
                ("symbol", ASCENDING),
                ("timestamp", DESCENDING)
            ])
            
            # Forecasts indexes
            self.db.forecasts.create_index([
                ("symbol", ASCENDING),
                ("model_type", ASCENDING),
                ("created_at", DESCENDING)
            ])
            
            # Model performance indexes
            self.db.model_performance.create_index([
                ("symbol", ASCENDING),
                ("model_type", ASCENDING),
                ("evaluation_date", DESCENDING)
            ])
            
            # API cache indexes - for Finnhub quotes and yfinance candles
            self.db.api_cache.create_index([
                ("symbol", ASCENDING),
                ("cache_type", ASCENDING),
                ("created_at", DESCENDING)
            ])
            # TTL index to auto-delete old cache entries after 5 minutes
            self.db.api_cache.create_index(
                "created_at",
                expireAfterSeconds=300  # 5 minutes cache
            )
            
            # Trained models indexes
            self.db.trained_models.create_index([
                ("symbol", ASCENDING),
                ("model_type", ASCENDING),
                ("created_at", DESCENDING)
            ])
            
            # Stock data cache indexes (1 year data)
            self.db.stock_data_cache.create_index([
                ("symbol", ASCENDING),
                ("data_type", ASCENDING),
                ("created_at", DESCENDING)
            ])
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
    
    def get_collection(self, name: str) -> Collection:
        """Get a MongoDB collection."""
        if self.db is None:
            raise RuntimeError("Database not connected")
        return self.db[name]
    
    def close(self):
        """Close database connection."""
        if self.client:
            self.client.close()
            logger.info("Database connection closed")

# Global database manager instance
db_manager = DatabaseManager()

class PriceData:
    """Historical price data model."""
    
    def __init__(self, symbol: str, timestamp: datetime, open_price: float,
                 high: float, low: float, close: float, volume: int):
        self.symbol = symbol
        self.timestamp = timestamp
        self.open_price = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "open": self.open_price,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "created_at": datetime.now(timezone.utc)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PriceData':
        """Create instance from dictionary."""
        return cls(
            symbol=data["symbol"],
            timestamp=data["timestamp"],
            open_price=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"]
        )

class ForecastData:
    """Forecast data model."""
    
    def __init__(self, symbol: str, model_type: str, forecast_horizon: int,
                 predictions: List[Dict[str, Any]], metrics: Dict[str, float],
                 confidence_intervals: Optional[List[Dict[str, float]]] = None):
        self.symbol = symbol
        self.model_type = model_type
        self.forecast_horizon = forecast_horizon
        self.predictions = predictions
        self.metrics = metrics
        self.confidence_intervals = confidence_intervals
        self.created_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "symbol": self.symbol,
            "model_type": self.model_type,
            "forecast_horizon": self.forecast_horizon,
            "predictions": self.predictions,
            "metrics": self.metrics,
            "confidence_intervals": self.confidence_intervals,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ForecastData':
        """Create instance from dictionary."""
        instance = cls(
            symbol=data["symbol"],
            model_type=data["model_type"],
            forecast_horizon=data["forecast_horizon"],
            predictions=data["predictions"],
            metrics=data["metrics"],
            confidence_intervals=data.get("confidence_intervals")
        )
        # Ensure created_at is timezone-aware
        created_at = data["created_at"]
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        instance.created_at = created_at
        return instance

class ModelPerformance:
    """Model performance tracking."""
    
    def __init__(self, symbol: str, model_type: str, metrics: Dict[str, float],
                 evaluation_period: str, data_points: int):
        self.symbol = symbol
        self.model_type = model_type
        self.metrics = metrics
        self.evaluation_period = evaluation_period
        self.data_points = data_points
        self.evaluation_date = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "symbol": self.symbol,
            "model_type": self.model_type,
            "metrics": self.metrics,
            "evaluation_period": self.evaluation_period,
            "data_points": self.data_points,
            "evaluation_date": self.evaluation_date
        }

# Database operation functions
def store_price_data(price_data: PriceData) -> bool:
    """Store historical price data in the database."""
    try:
        collection = db_manager.get_collection("historical_prices")
        
        # Check if record already exists
        existing = collection.find_one({
            "symbol": price_data.symbol,
            "timestamp": price_data.timestamp
        })
        
        if existing:
            # Update existing record
            collection.update_one(
                {"_id": existing["_id"]},
                {"$set": price_data.to_dict()}
            )
        else:
            # Insert new record
            collection.insert_one(price_data.to_dict())
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to store price data: {e}")
        return False

def get_price_data(symbol: str, limit: int = 1000) -> List[PriceData]:
    """Retrieve historical price data for a symbol."""
    try:
        collection = db_manager.get_collection("historical_prices")
        
        cursor = collection.find(
            {"symbol": symbol}
        ).sort("timestamp", DESCENDING).limit(limit)
        
        return [PriceData.from_dict(doc) for doc in cursor]
        
    except Exception as e:
        logger.error(f"Failed to retrieve price data: {e}")
        return []

def store_forecast(forecast: ForecastData) -> bool:
    """Store forecast data in the database."""
    try:
        collection = db_manager.get_collection("forecasts")
        collection.insert_one(forecast.to_dict())
        return True
        
    except Exception as e:
        logger.error(f"Failed to store forecast: {e}")
        return False

def get_latest_forecast(symbol: str, model_type: str) -> Optional[ForecastData]:
    """Get the latest forecast for a symbol and model type."""
    try:
        collection = db_manager.get_collection("forecasts")
        
        doc = collection.find_one(
            {"symbol": symbol, "model_type": model_type},
            sort=[("created_at", DESCENDING)]
        )
        
        return ForecastData.from_dict(doc) if doc else None
        
    except Exception as e:
        logger.error(f"Failed to retrieve forecast: {e}")
        return None

def store_model_performance(performance: ModelPerformance) -> bool:
    """Store model performance metrics."""
    try:
        collection = db_manager.get_collection("model_performance")
        collection.insert_one(performance.to_dict())
        return True
        
    except Exception as e:
        logger.error(f"Failed to store model performance: {e}")
        return False

def store_trained_model(symbol: str, model_type: str, model_data: bytes, 
                        metadata: Dict[str, Any] = None) -> bool:
    """Store a trained model in MongoDB using GridFS or binary storage."""
    try:
        import pickle
        from bson import Binary
        
        collection = db_manager.get_collection("trained_models")
        
        # Delete old models for same symbol/model_type (keep only latest)
        collection.delete_many({
            "symbol": symbol,
            "model_type": model_type
        })
        
        # Store new model
        doc = {
            "symbol": symbol,
            "model_type": model_type,
            "model_data": Binary(model_data),
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc)
        }
        
        collection.insert_one(doc)
        logger.info(f"Stored trained {model_type} model for {symbol}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store trained model: {e}")
        return False

def get_trained_model(symbol: str, model_type: str, max_age_hours: int = 24) -> Optional[bytes]:
    """Retrieve a trained model from MongoDB if it exists and is fresh."""
    try:
        collection = db_manager.get_collection("trained_models")
        
        # Find most recent model
        doc = collection.find_one(
            {"symbol": symbol, "model_type": model_type},
            sort=[("created_at", DESCENDING)]
        )
        
        if not doc:
            return None
        
        # Check age - ensure both datetimes are timezone-aware
        created_at = doc["created_at"]
        if created_at.tzinfo is None:
            # Make timezone-aware if needed
            created_at = created_at.replace(tzinfo=timezone.utc)
        
        age = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
        if age > max_age_hours:
            logger.info(f"Trained model for {symbol}/{model_type} is too old ({age:.1f}h)")
            return None
        
        logger.info(f"Retrieved trained {model_type} model for {symbol} (age: {age:.1f}h)")
        return bytes(doc["model_data"])
        
    except Exception as e:
        logger.error(f"Failed to retrieve trained model: {e}")
        return None

def store_stock_data_cache(symbol: str, data: List[Dict[str, Any]], 
                           data_type: str = "1year_hourly") -> bool:
    """Store fetched stock data in MongoDB for subsequent requests."""
    try:
        collection = db_manager.get_collection("stock_data_cache")
        
        # Delete old cache for same symbol/type
        collection.delete_many({
            "symbol": symbol,
            "data_type": data_type
        })
        
        # Store new cache
        doc = {
            "symbol": symbol,
            "data_type": data_type,
            "data": data,
            "created_at": datetime.now(timezone.utc),
            "count": len(data)
        }
        
        collection.insert_one(doc)
        logger.info(f"Cached {len(data)} data points for {symbol} ({data_type})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to cache stock data: {e}")
        return False

def get_stock_data_cache(symbol: str, data_type: str = "1year_hourly", 
                         max_age_hours: int = 24) -> Optional[List[Dict[str, Any]]]:
    """Retrieve cached stock data from MongoDB if fresh."""
    try:
        collection = db_manager.get_collection("stock_data_cache")
        
        doc = collection.find_one(
            {"symbol": symbol, "data_type": data_type},
            sort=[("created_at", DESCENDING)]
        )
        
        if not doc:
            return None
        
        # Check age - ensure both datetimes are timezone-aware
        created_at = doc["created_at"]
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        
        age = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
        if age > max_age_hours:
            logger.info(f"Cached data for {symbol} is too old ({age:.1f}h)")
            return None
        
        logger.info(f"Retrieved {doc['count']} cached data points for {symbol} (age: {age:.1f}h)")
        return doc["data"]
        
    except Exception as e:
        logger.error(f"Failed to retrieve cached stock data: {e}")
        return None

# Track first API call per session
_first_api_calls = set()

def is_first_api_call(symbol: str) -> bool:
    """Check if this is the first API call for a symbol in this session."""
    global _first_api_calls
    if symbol not in _first_api_calls:
        _first_api_calls.add(symbol)
        return True
    return False

def reset_first_api_calls():
    """Reset the first API calls tracker (for testing or restart)."""
    global _first_api_calls
    _first_api_calls.clear()

