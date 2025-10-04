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

from config import Config
from database import (
    PriceData, ForecastData, ModelPerformance,
    store_price_data, get_price_data, store_forecast, 
    get_latest_forecast, store_model_performance
)
from forecasting_simple_noml import create_model, ModelMetrics
from data_loader import load_data_async, ensure_symbol_data

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
    """Generate forecast for a symbol using specified model."""
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'lstm').lower()
        horizon = data.get('horizon', 24)  # hours
        retrain = data.get('retrain', False)
        
        # Validate inputs
        if model_type not in ['moving_average', 'arima', 'lstm', 'gru', 'transformer']:
            return jsonify({"error": "Invalid model type"}), 400
        
        if not 1 <= horizon <= 168:  # 1 hour to 1 week
            return jsonify({"error": "Horizon must be between 1 and 168 hours"}), 400
        
        # Get historical data, ensure it exists
        price_data = get_price_data(symbol.upper(), 2000)
        
        if len(price_data) < 10:
            # Try to fetch data from API
            logger.info(f"Insufficient data for {symbol}, attempting to fetch...")
            if ensure_symbol_data(symbol.upper()):
                price_data = get_price_data(symbol.upper(), 2000)
            
            # If still no data, generate mock data
            if len(price_data) < 10:
                logger.info(f"Generating mock data for {symbol}")
                mock_data = generate_mock_price_data(symbol.upper(), 100)
                price_data = mock_data
        
        # Check for existing forecast
        if not retrain:
            existing_forecast = get_latest_forecast(symbol.upper(), model_type)
            if existing_forecast and \
               (datetime.now(timezone.utc) - existing_forecast.created_at).total_seconds() < 3600:
                # Return existing forecast if less than 1 hour old
                return jsonify({
                    "symbol": symbol.upper(),
                    "model_type": model_type,
                    "forecast": existing_forecast.predictions,
                    "metrics": existing_forecast.metrics,
                    "confidence_intervals": existing_forecast.confidence_intervals,
                    "created_at": existing_forecast.created_at.isoformat(),
                    "cached": True
                })
        
        # Create and train model
        logger.info(f"Creating {model_type} model for {symbol}")
        model = create_model(model_type)
        
        # Prepare price data for training
        price_values = np.array([pd.close for pd in price_data])
        
        # Fit model
        logger.info(f"Training model on {len(price_values)} data points")
        model.fit(price_values)
        
        # Generate predictions
        logger.info(f"Generating {horizon} step forecast")
        predictions = model.predict(price_values, steps=horizon)
        
        # Convert to proper format if numpy array
        if hasattr(predictions, 'flatten'):
            predictions = predictions.flatten()
        
        # Ensure predictions is a list of numbers
        predictions = [float(p) for p in predictions]
        
        # Generate confidence intervals (simple estimation)
        confidence_intervals = np.column_stack([
            np.array(predictions) * 0.95,  # Lower bound
            np.array(predictions) * 1.05   # Upper bound
        ])
        
        # Calculate training metrics
        training_metrics = {"model_type": model_type, "training_samples": len(price_values)}
        
        # Create forecast timestamps
        last_timestamp = max(pd.timestamp for pd in price_data)
        forecast_timestamps = [
            (last_timestamp + timedelta(hours=i+1)).isoformat()
            for i in range(horizon)
        ]
        
        # Format predictions
        forecast_points = [
            {
                "timestamp": ts,
                "value": float(pred)
            }
            for ts, pred in zip(forecast_timestamps, predictions)
        ]
        
        # Calculate evaluation metrics on recent data
        evaluation_metrics = {}
        if len(price_data) > horizon * 2:
            # Use portion of data for validation
            split_point = len(price_data) - horizon
            train_data = price_values[:split_point]
            test_data = price_values[split_point:]
            
            # Create validation model
            val_model = create_model(model_type)
            val_model.fit(train_data)
            val_predictions = val_model.predict(train_data, steps=len(test_data))
            
            # Calculate metrics
            evaluation_metrics = ModelMetrics.calculate_metrics(test_data, val_predictions)
        
        # Combine training and evaluation metrics
        all_metrics = {**training_metrics, **evaluation_metrics}
        
        # Store forecast in database
        forecast_data = ForecastData(
            symbol=symbol.upper(),
            model_type=model_type,
            forecast_horizon=horizon,
            predictions=forecast_points,
            metrics=all_metrics,
            confidence_intervals=confidence_intervals.tolist() if confidence_intervals is not None else None
        )
        
        store_forecast(forecast_data)
        
        # Store model performance
        performance = ModelPerformance(
            symbol=symbol.upper(),
            model_type=model_type,
            metrics=evaluation_metrics,
            evaluation_period=f"{horizon}h",
            data_points=len(price_data)
        )
        store_model_performance(performance)
        
        return jsonify({
            "symbol": symbol.upper(),
            "model_type": model_type,
            "forecast": forecast_points,
            "metrics": all_metrics,
            "confidence_intervals": confidence_intervals.tolist() if confidence_intervals is not None else None,
            "created_at": forecast_data.created_at.isoformat(),
            "cached": False
        })
        
    except Exception as e:
        logger.error(f"Error generating forecast for {symbol}: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

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
        from database import db_manager
        
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
