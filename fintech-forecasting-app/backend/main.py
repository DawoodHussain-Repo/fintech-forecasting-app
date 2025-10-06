"""
Main Flask application for the FinTech Forecasting API.
Provides REST endpoints for forecasting models and data management.
"""

import logging
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np

from utils.config import Config
from utils.database import (
    PriceData, ForecastData, ModelPerformance,
    store_price_data, get_price_data, store_forecast, 
    get_latest_forecast, store_model_performance
)
from ml.models import create_model, ModelMetrics, create_or_load_model, fit_and_save_model
from utils.data_loader import load_data_async, ensure_symbol_data

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'

# Configure CORS
CORS(app, origins=Config.CORS_ORIGINS)

# Validate configuration on startup
if not Config.validate_config():
    logger.warning("Configuration validation failed - some features may not work")

def generate_mock_price_data(symbol: str, count: int):
    """Generate mock historical price data for testing."""
    mock_data = []
    base_price = 150.0  # Starting price
    
    for i in range(count):
        # Generate realistic price movement
        change = np.random.normal(0, 0.02)  # 2% volatility
        base_price *= (1 + change)
        
        # Ensure price stays positive
        base_price = max(base_price, 10.0)
        
        # Create price data point
        timestamp = datetime.now(timezone.utc) - timedelta(days=count-i)
        
        price_point = PriceData(
            symbol=symbol,
            timestamp=timestamp,
            open_price=base_price * 0.999,
            high=base_price * 1.005,
            low=base_price * 0.995,
            close=base_price,
            volume=1000000
        )
        mock_data.append(price_point)
    
    return mock_data

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    })

def cleanup_old_cache():
    """Clean up expired cache entries from database. Called during server shutdown."""
    try:
        from utils.database import db_manager
        cache_collection = db_manager.get_collection('api_cache')
        
        # Delete all expired cache entries
        result = cache_collection.delete_many({
            'expires_at': {'$lt': datetime.now(timezone.utc)}
        })
        
        logger.info(f"Cleanup: Deleted {result.deleted_count} expired cache entries")
        return result.deleted_count
    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}")
        return 0

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear API cache (admin endpoint)."""
    try:
        from utils.database import db_manager
        cache_collection = db_manager.get_collection('api_cache')
        
        # Optional: clear cache older than specified minutes
        minutes = request.json.get('older_than_minutes', 60) if request.json else 60
        
        result = cache_collection.delete_many({
            'created_at': {'$lt': datetime.now(timezone.utc) - timedelta(minutes=minutes)}
        })
        
        return jsonify({
            'success': True,
            'deleted_count': result.deleted_count,
            'message': f'Cleared {result.deleted_count} cached entries older than {minutes} minutes'
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/candles/<symbol>', methods=['GET'])
def get_candles(symbol: str):
    """Get historical candle data using yfinance with MongoDB caching."""
    try:
        import yfinance as yf
        from utils.database import db_manager
        
        symbol = symbol.upper()
        range_param = request.args.get('range', '1M')
        
        # Check cache first
        cache_collection = db_manager.get_collection('api_cache')
        cache_key = f"candles_{symbol}_{range_param}"
        
        cached_data = cache_collection.find_one({
            'symbol': symbol,
            'cache_type': 'candles',
            'range': range_param,
            'created_at': {'$gte': datetime.now(timezone.utc) - timedelta(minutes=5)}
        })
        
        if cached_data:
            logger.info(f"Cache hit for {symbol} candles")
            return jsonify({'candles': cached_data['data'], 'cached': True})
        
        # Map range to yfinance period
        period_map = {
            '1D': '5d',
            '1W': '1mo',
            '1M': '3mo',
            '6M': '6mo',
            '1Y': '1y',
        }
        period = period_map.get(range_param, '3mo')
        
        # Fetch data from yfinance
        logger.info(f"Fetching fresh candles for {symbol}")
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        # Convert to candle format
        candles = []
        for date, row in hist.iterrows():
            candles.append({
                'timestamp': date.isoformat(),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume']),
            })
        
        # Store in cache
        try:
            cache_collection.insert_one({
                'symbol': symbol,
                'cache_type': 'candles',
                'range': range_param,
                'data': candles,
                'created_at': datetime.now(timezone.utc)
            })
            logger.info(f"Cached candles for {symbol}")
        except Exception as cache_err:
            logger.warning(f"Failed to cache candles: {cache_err}")
        
        return jsonify({'candles': candles, 'cached': False})
    
    except Exception as e:
        logger.error(f"Error fetching candles for {symbol}: {e}")
        return jsonify({'error': str(e), 'candles': []}), 500

@app.route('/api/data/<symbol>', methods=['GET'])
def get_symbol_data(symbol: str):
    """Get historical data for a symbol."""
    try:
        limit = request.args.get('limit', 1000, type=int)
        
        # Get data from database
        price_data = get_price_data(symbol.upper(), limit)
        
        if not price_data:
            # If no data in database, try to fetch from Alpha Vantage
            success = fetch_and_store_data(symbol.upper())
            if success:
                price_data = get_price_data(symbol.upper(), limit)
        
        # Convert to API format
        data = [{
            "timestamp": pd.timestamp.isoformat(),
            "open": pd.open_price,
            "high": pd.high,
            "low": pd.low,
            "close": pd.close,
            "volume": pd.volume
        } for pd in price_data]
        
        return jsonify({
            "symbol": symbol.upper(),
            "data": data,
            "count": len(data)
        })
        
    except Exception as e:
        logger.error(f"Error getting data for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast/<symbol>', methods=['POST'])
def generate_forecast(symbol: str):
    """Generate SHORT-TERM forecast using recent 5-7 days of data with technical indicators."""
    try:
        from utils.database import (
            is_first_api_call, get_stock_data_cache, store_stock_data_cache,
            get_trained_model, store_trained_model
        )
        
        data = request.get_json()
        model_type = data.get('model_type', 'moving_average').lower()
        horizon = data.get('horizon', 24)  # hours
        
        logger.info(f"SHORT-TERM Forecast: {symbol} | Model: {model_type} | Horizon: {horizon}h")
        
        # Validate inputs
        if model_type not in ['moving_average', 'arima', 'lstm', 'gru', 'transformer']:
            return jsonify({"error": "Invalid model type"}), 400
        
        if not 1 <= horizon <= 72:  # Max 72 hours (3 days) for short-term
            return jsonify({"error": "Horizon must be between 1 and 72 hours"}), 400
        
        # Step 1: Fetch RECENT 7 days of HOURLY data (use cache after first call)
        import yfinance as yf
        
        symbol_upper = symbol.upper()
        is_first_call = is_first_api_call(symbol_upper)
        
        # Try to get cached data (skip if first call)
        cached_data = None
        if not is_first_call:
            cached_data = get_stock_data_cache(symbol_upper, "7d_hourly", max_age_hours=6)
        
        if cached_data:
            logger.info(f"Using cached data for {symbol_upper} ({len(cached_data)} points)")
            # Reconstruct data from cache
            close_prices = np.array([d['close'] for d in cached_data])
            high_prices = np.array([d['high'] for d in cached_data])
            low_prices = np.array([d['low'] for d in cached_data])
            volume = np.array([d['volume'] for d in cached_data])
        else:
            # Fetch from API (first call or cache expired)
            logger.info(f"{'[FIRST CALL] ' if is_first_call else ''}Fetching RECENT 7 days of HOURLY data for {symbol_upper}")
            
            try:
                ticker = yf.Ticker(symbol_upper)
                # Get 7 days of HOURLY data (most recent short-term data)
                hist = ticker.history(period="7d", interval="1h")
                
                if hist.empty or len(hist) < 24:
                    logger.warning(f"Insufficient recent data for {symbol_upper}, trying fallback")
                    # Fallback to 5 days of hourly data
                    hist = ticker.history(period="5d", interval="1h")
                
                if hist.empty or len(hist) < 20:
                    logger.error(f"No recent data available for {symbol_upper}")
                    return jsonify({
                        "error": "Insufficient recent data",
                        "details": "Need at least 20 hours of recent price data"
                    }), 400
                
                # Extract OHLCV data
                close_prices = hist['Close'].values
                high_prices = hist['High'].values
                low_prices = hist['Low'].values
                volume = hist['Volume'].values
                
                logger.info(f"Fetched {len(close_prices)} hours of recent data for {symbol_upper}")
                logger.info(f"Data range: {hist.index[0]} to {hist.index[-1]}")
                logger.info(f"Current price: ${close_prices[-1]:.2f}")
                
                # Cache the data for subsequent requests
                cache_data = [
                    {
                        'timestamp': str(hist.index[i]),
                        'close': float(close_prices[i]),
                        'high': float(high_prices[i]),
                        'low': float(low_prices[i]),
                        'volume': float(volume[i])
                    }
                    for i in range(len(close_prices))
                ]
                store_stock_data_cache(symbol_upper, cache_data, "7d_hourly")
                logger.info(f"Cached {len(cache_data)} data points for {symbol_upper}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol_upper}: {e}")
                return jsonify({
                    "error": "Failed to fetch market data",
                    "details": str(e)
                }), 500
        
        # Step 2: Calculate SHORT-TERM Technical Indicators
        logger.info("Calculating short-term technical indicators...")
        
        # 7-period Simple Moving Average
        if len(close_prices) >= 7:
            sma_7 = np.mean(close_prices[-7:])
        else:
            sma_7 = close_prices[-1]
        
        # RSI (14 periods)
        def calculate_rsi(prices, period=14):
            if len(prices) < period + 1:
                return 50  # Neutral
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            if avg_loss == 0:
                return 100
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = calculate_rsi(close_prices)
        
        # Bollinger Bands (20 periods, 2 std)
        if len(close_prices) >= 20:
            bb_period = 20
            bb_std = 2
            sma_20 = np.mean(close_prices[-bb_period:])
            std_20 = np.std(close_prices[-bb_period:])
            bb_upper = sma_20 + (bb_std * std_20)
            bb_lower = sma_20 - (bb_std * std_20)
        else:
            bb_upper = close_prices[-1] * 1.05
            bb_lower = close_prices[-1] * 0.95
        
        # Recent volatility (last 24 hours)
        if len(close_prices) >= 24:
            recent_volatility = np.std(close_prices[-24:]) / np.mean(close_prices[-24:]) * 100
        else:
            recent_volatility = np.std(close_prices) / np.mean(close_prices) * 100
        
        # Momentum (recent trend)
        if len(close_prices) >= 5:
            momentum = (close_prices[-1] - close_prices[-5]) / close_prices[-5] * 100
        else:
            momentum = 0
        
        # Support/Resistance levels (recent highs/lows)
        recent_high = np.max(high_prices[-24:]) if len(high_prices) >= 24 else np.max(high_prices)
        recent_low = np.min(low_prices[-24:]) if len(low_prices) >= 24 else np.min(low_prices)
        
        technical_indicators = {
            "sma_7": float(sma_7),
            "rsi": float(rsi),
            "bb_upper": float(bb_upper),
            "bb_lower": float(bb_lower),
            "volatility_pct": float(recent_volatility),
            "momentum_5d_pct": float(momentum),
            "support_level": float(recent_low),
            "resistance_level": float(recent_high)
        }
        
        logger.info(f"Indicators: RSI={rsi:.1f}, Volatility={recent_volatility:.2f}%, Momentum={momentum:.2f}%")
        
        # Step 3: Load existing model or create and train new one
        logger.info(f"Checking for existing {model_type} model for {symbol_upper}")
        
        # Try to load existing model from backend/models directory
        from ml.models import PersistentModel
        temp_model = create_model(model_type)
        
        if temp_model is None:
            return jsonify({"error": f"Failed to create {model_type} model"}), 500
        
        # Check if we have a fresh model (within 24 hours)
        model = None
        if temp_model.is_model_fresh(symbol_upper, model_type, max_age_hours=24):
            logger.info(f"Fresh model found for {symbol_upper}/{model_type}, loading from disk...")
            model = temp_model.load_model(symbol_upper, model_type, max_age_hours=24)
        
        if model is None:
            # No fresh model found, train a new one
            logger.info(f"Training NEW {model_type} model for {symbol_upper} on {len(close_prices)} hours of RECENT data")
            model = create_model(model_type)
            
            # Train on recent data (fit returns self)
            model = model.fit(close_prices)
            
            # Save the model to backend/models directory for future use
            logger.info(f"Saving trained {model_type} model for {symbol_upper} to backend/models/")
            save_success = model.save_model(symbol_upper, model_type)
            if save_success:
                logger.info(f"✓ Model successfully saved to backend/models/{symbol_upper}_{model_type}_{datetime.now().strftime('%Y%m%d')}.pkl")
            else:
                logger.warning(f"⚠ Failed to save model to disk (will need to retrain next time)")
        else:
            logger.info(f"Using CACHED {model_type} model for {symbol_upper} (trained within last 24 hours)")
        
        # Step 4: Generate predictions
        logger.info(f"Generating {horizon}-hour forecast with confidence intervals")
        predictions = model.predict(close_prices, steps=horizon)
        
        # Ensure predictions are in the right format
        if hasattr(predictions, 'flatten'):
            predictions = predictions.flatten()
        predictions = [float(p) for p in predictions[:horizon]]
        
        # Step 5: Determine price direction and confidence
        current_price = float(close_prices[-1])
        
        def get_direction_and_confidence(pred_price, current, volatility):
            """Determine direction and confidence based on prediction and volatility"""
            change_pct = ((pred_price - current) / current) * 100
            
            # Direction
            if change_pct > 0.5:
                direction = "up"
            elif change_pct < -0.5:
                direction = "down"
            else:
                direction = "sideways"
            
            # Confidence (based on volatility and change magnitude)
            if volatility < 1.0:  # Low volatility
                if abs(change_pct) > 2:
                    confidence = "high"
                elif abs(change_pct) > 0.5:
                    confidence = "medium"
                else:
                    confidence = "low"
            elif volatility < 3.0:  # Medium volatility
                if abs(change_pct) > 3:
                    confidence = "medium"
                else:
                    confidence = "low"
            else:  # High volatility
                confidence = "low"
            
            return direction, confidence
        
        # Predictions for different horizons
        pred_1h = predictions[0] if len(predictions) >= 1 else current_price
        pred_4h = predictions[3] if len(predictions) >= 4 else predictions[-1]
        pred_24h = predictions[23] if len(predictions) >= 24 else predictions[-1]
        
        dir_1h, conf_1h = get_direction_and_confidence(pred_1h, current_price, recent_volatility)
        dir_4h, conf_4h = get_direction_and_confidence(pred_4h, current_price, recent_volatility)
        dir_24h, conf_24h = get_direction_and_confidence(pred_24h, current_price, recent_volatility)
        
        # Step 6: Generate price ranges (using volatility)
        volatility_multiplier = recent_volatility / 100
        
        forecast_points = []
        for i, pred in enumerate(predictions):
            # Dynamic confidence interval based on time horizon and volatility
            time_factor = np.sqrt(i + 1)  # Uncertainty increases with time
            range_width = pred * volatility_multiplier * time_factor
            
            forecast_points.append({
                "timestamp": (datetime.now(timezone.utc) + timedelta(hours=i+1)).isoformat(),
                "predicted_price": float(pred),
                "price_range_low": float(pred - range_width),
                "price_range_high": float(pred + range_width),
                "confidence": "high" if i < 4 else ("medium" if i < 12 else "low")
            })
        
        # Step 7: Prepare response
        response = {
            "symbol": symbol_upper,
            "model_type": model_type,
            "current_price": current_price,
            
            # Short-term forecasts
            "forecast_1h": {
                "direction": dir_1h,
                "confidence": conf_1h,
                "predicted_price": float(pred_1h),
                "price_range": [float(pred_1h * 0.98), float(pred_1h * 1.02)]
            },
            "forecast_4h": {
                "direction": dir_4h,
                "confidence": conf_4h,
                "predicted_price": float(pred_4h),
                "price_range": [float(pred_4h * 0.96), float(pred_4h * 1.04)]
            },
            "forecast_24h": {
                "direction": dir_24h,
                "confidence": conf_24h,
                "predicted_price": float(pred_24h),
                "price_range": [float(pred_24h * 0.93), float(pred_24h * 1.07)]
            },
            
            # Technical indicators
            "technical_indicators": technical_indicators,
            
            # Full forecast timeline
            "forecast": forecast_points,
            
            # Metadata
            "data_points_used": len(close_prices),
            "data_period": "7 days hourly",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cached": False
        }
        
        logger.info(f"✓ Forecast complete: 1h={dir_1h}({conf_1h}), 4h={dir_4h}({conf_4h}), 24h={dir_24h}({conf_24h})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to generate forecast",
            "details": str(e)
        }), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """List available forecasting models."""
    return jsonify({
        "models": [
            {
                "type": "moving_average",
                "name": "Moving Average",
                "description": "Simple moving average forecast",
                "traditional": True
            },
            {
                "type": "arima",
                "name": "ARIMA",
                "description": "AutoRegressive Integrated Moving Average",
                "traditional": True
            },
            {
                "type": "lstm",
                "name": "LSTM",
                "description": "Long Short-Term Memory neural network",
                "traditional": False
            },
            {
                "type": "gru",
                "name": "GRU", 
                "description": "Gated Recurrent Unit neural network",
                "traditional": False
            },
            {
                "type": "transformer",
                "name": "Transformer",
                "description": "Transformer-based neural network",
                "traditional": False
            }
        ]
    })

@app.route('/api/performance/<symbol>', methods=['GET'])
def get_model_performance(symbol: str):
    """Get model performance metrics for a symbol."""
    try:
        from utils.database import db_manager
        
        collection = db_manager.get_collection("model_performance")
        
        # Get recent performance data
        cursor = collection.find(
            {"symbol": symbol.upper()}
        ).sort("evaluation_date", -1).limit(50)
        
        performance_data = []
        for doc in cursor:
            performance_data.append({
                "model_type": doc["model_type"],
                "metrics": doc["metrics"],
                "evaluation_period": doc["evaluation_period"],
                "data_points": doc["data_points"],
                "evaluation_date": doc["evaluation_date"].isoformat()
            })
        
        return jsonify({
            "symbol": symbol.upper(),
            "performance_history": performance_data
        })
        
    except Exception as e:
        logger.error(f"Error getting performance for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/compare-models/<symbol>', methods=['POST'])
def compare_models(symbol: str):
    """Compare multiple models on the same dataset."""
    try:
        data = request.get_json()
        model_types = data.get('models', ['arima', 'lstm', 'gru', 'moving_average', 'transformer'])
        test_size = data.get('test_size', 30)  # Days to use for testing
        
        # Get historical data
        price_data = get_price_data(symbol.upper(), 500)
        if len(price_data) < 50:
            return jsonify({"error": "Insufficient data for comparison"}), 400
        
        price_values = np.array([pd.close for pd in price_data])
        
        # Split data for testing
        split_point = len(price_values) - test_size
        train_data = price_values[:split_point]
        test_data = price_values[split_point:]
        
        comparison_results = []
        
        for model_type in model_types:
            try:
                logger.info(f"Evaluating {model_type} for comparison")
                
                # Create and train model
                model = create_model(model_type)
                model.fit(train_data)
                
                # Generate predictions
                predictions = model.predict(train_data, steps=len(test_data))
                
                # Calculate metrics
                metrics = ModelMetrics.calculate_metrics(test_data, predictions)
                
                comparison_results.append({
                    "name": model_type.upper(),
                    "type": "neural" if model_type in ['lstm', 'gru', 'transformer'] else "traditional",
                    "mse": metrics['mse'],
                    "mae": metrics['mae'], 
                    "rmse": metrics['rmse'],
                    "mape": metrics['mape'],
                    "accuracy": metrics.get('direction_accuracy', 0),
                    "trainingTime": 10.0,  # Mock training time
                    "status": "success",
                    "description": f"{model_type.title()} model evaluation on {test_size} test samples"
                })
                
            except Exception as e:
                logger.error(f"Error evaluating {model_type}: {e}")
                comparison_results.append({
                    "name": model_type.upper(),
                    "type": "neural" if model_type in ['lstm', 'gru', 'transformer'] else "traditional", 
                    "mse": 999.0,
                    "mae": 999.0,
                    "rmse": 999.0,
                    "mape": 999.0,
                    "accuracy": 0,
                    "trainingTime": 0.0,
                    "status": "error",
                    "description": f"Failed to evaluate {model_type}: {str(e)}"
                })
        
        return jsonify({
            "symbol": symbol.upper(),
            "test_period": f"{test_size} days",
            "models": comparison_results
        })
        
    except Exception as e:
        logger.error(f"Error comparing models for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/update-data', methods=['POST'])
def update_data():
    """Manually trigger data update for symbols."""
    try:
        data = request.get_json()
        symbols = data.get('symbols', Config.DATA_COLLECTION_SYMBOLS)
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        results = {}
        for symbol in symbols:
            try:
                success = fetch_and_store_data(symbol.upper())
                results[symbol] = "success" if success else "failed"
            except Exception as e:
                results[symbol] = f"error: {str(e)}"
        
        return jsonify({
            "message": "Data update completed",
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error updating data: {e}")
        return jsonify({"error": str(e)}), 500

def fetch_and_store_data(symbol: str) -> bool:
    """Fetch data from Alpha Vantage and store in database."""
    try:
        # Determine data type based on symbol
        if '/' in symbol:
            # Forex
            from_currency, to_currency = symbol.split('/')
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'FX_DAILY',
                'from_symbol': from_currency,
                'to_symbol': to_currency,
                'apikey': Config.ALPHA_VANTAGE_API_KEY
            }
        elif symbol in ['BTC', 'ETH', 'LTC', 'XRP', 'ADA', 'DOT', 'UNI']:
            # Crypto
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'DIGITAL_CURRENCY_DAILY',
                'symbol': symbol,
                'market': 'USD',
                'apikey': Config.ALPHA_VANTAGE_API_KEY
            }
        else:
            # Stock
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': Config.ALPHA_VANTAGE_API_KEY
            }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle rate limiting
        if 'Note' in data:
            logger.warning(f"Alpha Vantage rate limit hit for {symbol}")
            return False
        
        if 'Error Message' in data:
            logger.error(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
            return False
        
        # Parse data based on type
        time_series_key = None
        for key in data.keys():
            if 'Time Series' in key or 'Daily' in key:
                time_series_key = key
                break
        
        if not time_series_key:
            logger.error(f"No time series data found for {symbol}")
            return False
        
        time_series = data[time_series_key]
        
        # Store data
        stored_count = 0
        for date_str, values in time_series.items():
            try:
                # Parse values based on data format
                if 'USD' in str(values):
                    # Crypto format
                    open_price = float(values.get('1a. open (USD)', values.get('1. open', 0)))
                    high = float(values.get('2a. high (USD)', values.get('2. high', 0)))
                    low = float(values.get('3a. low (USD)', values.get('3. low', 0)))
                    close = float(values.get('4a. close (USD)', values.get('4. close', 0)))
                    volume = float(values.get('5. volume', 0))
                else:
                    # Stock/Forex format
                    open_price = float(values.get('1. open', 0))
                    high = float(values.get('2. high', 0))
                    low = float(values.get('3. low', 0))
                    close = float(values.get('4. close', 0))
                    volume = float(values.get('5. volume', 0))
                
                price_data = PriceData(
                    symbol=symbol,
                    timestamp=datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc),
                    open_price=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=int(volume)
                )
                
                if store_price_data(price_data):
                    stored_count += 1
                    
            except (ValueError, KeyError) as e:
                logger.warning(f"Error parsing data for {symbol} on {date_str}: {e}")
                continue
        
        logger.info(f"Stored {stored_count} data points for {symbol}")
        return stored_count > 0
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return False

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Starting FinTech Forecasting API")
    logger.info(f"Configuration: {Config.FLASK_ENV} mode")
    
    # Start data loading in background thread (optional, will run when needed)
    try:
        logger.info("Starting background data loading...")
        import threading
        data_thread = threading.Thread(target=load_data_async, daemon=True)
        data_thread.start()
    except Exception as e:
        logger.warning(f"Background data loading failed: {e}")
    
    # Start Flask app with Windows-compatible settings
    logger.info("Starting Flask server...")
    app.run(
        host='127.0.0.1',
        port=Config.FLASK_PORT,
        debug=False,  # Disable debug to avoid Windows threading issues
        use_reloader=False,  # Disable reloader to avoid socket issues
        threaded=True
    )
