from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from data_fetcher import DataFetcher
from database import Database
from models.traditional import TraditionalForecaster
from models.neural import NeuralForecaster

# Import adaptive learning components
from adaptive_learning import (
    ModelVersionManager,
    PerformanceTracker,
    AdaptiveEnsemble,
    RetrainingScheduler
)

app = Flask(__name__, template_folder='../frontend/templates', 
            static_folder='../frontend/static')
CORS(app)

# Initialize components
data_fetcher = DataFetcher()
db = Database()
traditional_forecaster = TraditionalForecaster()
neural_forecaster = NeuralForecaster(db=db)

# Initialize adaptive learning components
version_manager = ModelVersionManager(db)
performance_tracker = PerformanceTracker(db)
ensemble_rebalancer = AdaptiveEnsemble(db)
scheduler = RetrainingScheduler(db, data_fetcher)

# Start scheduler automatically with default symbols
try:
    scheduler.start(symbols=['AAPL', 'GOOGL', 'BTC-USD'])
    print("✅ Adaptive learning scheduler started")
except Exception as e:
    print(f"⚠️  Scheduler start failed: {e}")

# Popular symbols
POPULAR_SYMBOLS = {
    'stocks': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
    'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD'],
    'forex': ['EURUSD=X', 'GBPUSD=X', 'JPYUSD=X']
}

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', symbols=POPULAR_SYMBOLS)

@app.route('/monitor')
def monitor():
    """Render adaptive learning monitor page"""
    return render_template('monitor.html', symbols=POPULAR_SYMBOLS)

@app.route('/evaluation')
def evaluation():
    """Render model evaluation page"""
    return render_template('evaluation.html')

@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    """Get list of available symbols"""
    return jsonify(POPULAR_SYMBOLS)

@app.route('/api/fetch_data', methods=['POST'])
def fetch_data():
    """Fetch historical data for a symbol"""
    data = request.json
    symbol = data.get('symbol', 'AAPL')
    period = data.get('period', '1y')
    interval = data.get('interval', '1d')
    
    # Fetch data
    df = data_fetcher.fetch_data(symbol, period, interval)
    
    if df is None:
        return jsonify({'error': 'Failed to fetch data'}), 400
    
    # Store in database
    records = df.to_dict('records')
    for record in records:
        if 'date' in record:
            record['date'] = record['date'].isoformat() if hasattr(record['date'], 'isoformat') else str(record['date'])
    
    db.store_historical_data(symbol, records)
    
    # Get symbol info
    info = data_fetcher.get_symbol_info(symbol)
    db.store_metadata(symbol, info)
    
    return jsonify({
        'success': True,
        'symbol': symbol,
        'records': len(records),
        'data': records[-100:]  # Return last 100 records
    })

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """Generate forecast for a symbol with incremental training"""
    data = request.json
    symbol = data.get('symbol', 'AAPL')
    model_type = data.get('model', 'ensemble')  # ensemble, arima, lstm, gru
    horizon = data.get('horizon', '24h')  # 1h, 3h, 24h, 72h
    
    # Parse horizon
    horizon_map = {'1h': 1, '3h': 3, '24h': 24, '72h': 72, '1d': 1, '3d': 3, '7d': 7}
    steps = horizon_map.get(horizon, 24)
    
    print(f"\n{'='*60}")
    print(f"🎯 Forecast Request: {symbol} / {model_type.upper()} / {horizon}")
    print(f"{'='*60}")
    
    # Always fetch fresh data from yfinance for real-time predictions
    print(f"📥 Fetching fresh data for {symbol}...")
    df = data_fetcher.fetch_data(symbol, period='1y', interval='1d')
    
    if df is None or len(df) < 100:
        return jsonify({'error': 'Insufficient data for forecasting'}), 400
    
    print(f"✓ Fetched {len(df)} days of data")
    
    # Store historical data
    records = df.to_dict('records')
    for record in records:
        if 'date' in record:
            record['date'] = record['date'].isoformat() if hasattr(record['date'], 'isoformat') else str(record['date'])
    db.store_historical_data(symbol, records)
    
    # Extract close prices
    close_prices = df['close']
    
    # Generate forecast based on model type
    try:
        # For neural models, do incremental training
        if model_type in ['lstm', 'gru']:
            print(f"🔄 Training {model_type.upper()} model with latest data...")
            
            # Get previous version info
            prev_version = version_manager.get_active_version(symbol, model_type)
            print(f"   Previous version: {prev_version or 'None (first training)'}")
            
            # Train/fine-tune model
            if model_type == 'lstm':
                predictions, metrics = neural_forecaster.lstm_forecast(
                    close_prices, steps=steps, epochs=10, symbol=symbol, use_cache=True
                )
            else:  # gru
                predictions, metrics = neural_forecaster.gru_forecast(
                    close_prices, steps=steps, epochs=10, symbol=symbol, use_cache=True
                )
            
            print(f"✓ Training complete - MAPE: {metrics['mape']:.2f}%")
            
            # Get current version (model was saved during forecast)
            current_version = version_manager.get_active_version(symbol, model_type)
            
            # Log training event
            performance_tracker.log_training_event(
                symbol=symbol,
                model_name=model_type,
                version=current_version or 'v1.0.0',
                trigger='forecast_request',
                data_points=len(close_prices),
                epochs=10,
                final_loss=0.001,
                metrics=metrics,
                status='success'
            )
            
            print(f"📝 Logged training event")
            
            # Log prediction for performance tracking
            # Use last actual price as baseline for comparison
            last_actual_price = float(close_prices.iloc[-1])
            first_predicted_price = float(predictions[0])
            
            performance_tracker.log_prediction(
                symbol=symbol,
                model_name=model_type,
                version=current_version or 'v1.0.0',
                actual_price=last_actual_price,
                predicted_price=first_predicted_price,
                metrics=metrics
            )
            
            print(f"📊 Logged prediction for evaluation")
            
        elif model_type == 'arima':
            predictions, metrics = traditional_forecaster.arima_forecast(close_prices, steps=steps)
            
            # Log prediction for traditional models too
            last_actual_price = float(close_prices.iloc[-1])
            first_predicted_price = float(predictions[0])
            
            performance_tracker.log_prediction(
                symbol=symbol,
                model_name=model_type,
                version='traditional',
                actual_price=last_actual_price,
                predicted_price=first_predicted_price,
                metrics=metrics
            )
            
        elif model_type == 'ma':
            predictions, metrics = traditional_forecaster.moving_average_forecast(close_prices, steps=steps)
            
            # Log prediction
            last_actual_price = float(close_prices.iloc[-1])
            first_predicted_price = float(predictions[0])
            
            performance_tracker.log_prediction(
                symbol=symbol,
                model_name=model_type,
                version='traditional',
                actual_price=last_actual_price,
                predicted_price=first_predicted_price,
                metrics=metrics
            )
            
        elif model_type == 'ensemble':
            predictions, metrics = traditional_forecaster.ensemble_forecast(close_prices, steps=steps)
            
            # Log prediction
            last_actual_price = float(close_prices.iloc[-1])
            first_predicted_price = float(predictions[0])
            
            performance_tracker.log_prediction(
                symbol=symbol,
                model_name=model_type,
                version='traditional',
                actual_price=last_actual_price,
                predicted_price=first_predicted_price,
                metrics=metrics
            )
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
        # Generate future dates
        last_date = df['date'].iloc[-1]
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)
        
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='D')
        
        # Prepare prediction records
        prediction_records = []
        for date, pred in zip(future_dates, predictions):
            prediction_records.append({
                'date': date.isoformat(),
                'predicted_close': float(pred)
            })
        
        # Store predictions in database
        db.store_prediction(symbol, model_type, horizon, prediction_records, metrics)
        
        # Prepare response
        historical_data = df[['date', 'open', 'high', 'low', 'close', 'volume']].tail(100).to_dict('records')
        for record in historical_data:
            if 'date' in record:
                record['date'] = record['date'].isoformat() if hasattr(record['date'], 'isoformat') else str(record['date'])
        
        # Get latest data timestamp
        latest_data_time = df['date'].iloc[-1]
        if hasattr(latest_data_time, 'isoformat'):
            latest_data_time = latest_data_time.isoformat()
        else:
            latest_data_time = str(latest_data_time)
        
        # Get training count
        training_count = performance_tracker.training_logs.count_documents({
            'symbol': symbol,
            'model_name': model_type
        })
        
        print(f"✅ Forecast complete!")
        print(f"   Training count: {training_count}")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'model': model_type,
            'horizon': horizon,
            'historical_data': historical_data,
            'predictions': prediction_records,
            'metrics': metrics,
            'latest_data_time': latest_data_time,
            'prediction_time': datetime.utcnow().isoformat(),
            'training_count': training_count,
            'version': version_manager.get_active_version(symbol, model_type) if model_type in ['lstm', 'gru'] else None
        })
    
    except Exception as e:
        print(f"❌ Forecast error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare_models', methods=['POST'])
def compare_models():
    """Compare multiple forecasting models"""
    data = request.json
    symbol = data.get('symbol', 'AAPL')
    horizon = data.get('horizon', '24h')
    
    horizon_map = {'1h': 1, '3h': 3, '24h': 24, '72h': 72}
    steps = horizon_map.get(horizon, 24)
    
    # Fetch data
    df = data_fetcher.fetch_data(symbol, period='1y', interval='1d')
    
    if df is None or len(df) < 100:
        return jsonify({'error': 'Insufficient data'}), 400
    
    close_prices = df['close']
    
    # Compare models
    results = {}
    
    try:
        # Traditional models
        ma_pred, ma_metrics = traditional_forecaster.moving_average_forecast(close_prices, steps=steps)
        results['moving_average'] = {'predictions': ma_pred.tolist(), 'metrics': ma_metrics}
        
        arima_pred, arima_metrics = traditional_forecaster.arima_forecast(close_prices, steps=steps)
        results['arima'] = {'predictions': arima_pred.tolist(), 'metrics': arima_metrics}
        
        ensemble_pred, ensemble_metrics = traditional_forecaster.ensemble_forecast(close_prices, steps=steps)
        results['ensemble'] = {'predictions': ensemble_pred.tolist(), 'metrics': ensemble_metrics}
        
        # Neural models (with model caching for speed)
        lstm_pred, lstm_metrics = neural_forecaster.lstm_forecast(
            close_prices, steps=steps, epochs=20, symbol=symbol, use_cache=True
        )
        results['lstm'] = {'predictions': lstm_pred.tolist(), 'metrics': lstm_metrics}
        
        gru_pred, gru_metrics = neural_forecaster.gru_forecast(
            close_prices, steps=steps, epochs=20, symbol=symbol, use_cache=True
        )
        results['gru'] = {'predictions': gru_pred.tolist(), 'metrics': gru_metrics}
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'horizon': horizon,
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical/<symbol>', methods=['GET'])
def get_historical(symbol):
    """Get historical data from database"""
    limit = request.args.get('limit', 100, type=int)
    
    records = db.get_historical_data(symbol, limit=limit)
    
    return jsonify({
        'success': True,
        'symbol': symbol,
        'data': records
    })

# ============================================================================
# ADAPTIVE LEARNING API ENDPOINTS
# ============================================================================

@app.route('/api/adaptive/scheduler/start', methods=['POST'])
def start_scheduler():
    """Start the adaptive learning scheduler"""
    data = request.json
    symbols = data.get('symbols', ['AAPL'])
    
    try:
        scheduler.start(symbols=symbols)
        return jsonify({
            'success': True,
            'message': 'Scheduler started',
            'symbols': symbols
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive/scheduler/stop', methods=['POST'])
def stop_scheduler():
    """Stop the adaptive learning scheduler"""
    try:
        scheduler.stop()
        return jsonify({
            'success': True,
            'message': 'Scheduler stopped'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive/scheduler/status', methods=['GET'])
def scheduler_status():
    """Get scheduler status"""
    try:
        status = scheduler.get_status()
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive/retrain', methods=['POST'])
def trigger_retrain():
    """Manually trigger model retraining"""
    data = request.json
    symbol = data.get('symbol', 'AAPL')
    model_name = data.get('model', 'lstm')
    
    try:
        scheduler.trigger_manual_retrain(symbol, model_name)
        return jsonify({
            'success': True,
            'message': f'Retraining triggered for {symbol}/{model_name}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive/performance/<symbol>/<model>', methods=['GET'])
def get_performance(symbol, model):
    """Get performance statistics for a model"""
    try:
        stats = performance_tracker.get_model_statistics(symbol, model)
        return jsonify({
            'success': True,
            'symbol': symbol,
            'model': model,
            'statistics': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive/performance/trend/<symbol>/<model>', methods=['GET'])
def get_performance_trend(symbol, model):
    """Get performance trend over time"""
    days = request.args.get('days', 30, type=int)
    
    try:
        trend = performance_tracker.get_performance_trend(symbol, model, days=days)
        return jsonify({
            'success': True,
            'symbol': symbol,
            'model': model,
            'trend': trend
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive/versions/<symbol>/<model>', methods=['GET'])
def get_version_history(symbol, model):
    """Get version history for a model (or usage history for traditional models)"""
    limit = request.args.get('limit', 20, type=int)
    
    try:
        history = version_manager.get_version_history(symbol, model, limit=limit)
        
        # Convert ObjectId and datetime to strings
        for version in history:
            version['_id'] = str(version['_id'])
            if 'trained_at' in version:
                version['trained_at'] = version['trained_at'].isoformat()
        
        # If no version history (traditional models), create pseudo-history from predictions
        if len(history) == 0:
            # Get prediction history grouped by date
            pipeline = [
                {
                    '$match': {
                        'symbol': symbol,
                        'model_name': model
                    }
                },
                {
                    '$group': {
                        '_id': {
                            '$dateToString': {
                                'format': '%Y-%m-%d',
                                'date': '$timestamp'
                            }
                        },
                        'avg_mape': {'$avg': '$percentage_error'},
                        'count': {'$sum': 1},
                        'timestamp': {'$max': '$timestamp'}
                    }
                },
                {
                    '$sort': {'timestamp': -1}
                },
                {
                    '$limit': limit
                }
            ]
            
            usage_history = list(performance_tracker.performance_collection.aggregate(pipeline))
            
            # Convert to version-like format
            history = []
            for i, usage in enumerate(usage_history):
                history.append({
                    '_id': str(i),
                    'version': f"usage_{i+1}",
                    'trained_at': usage['timestamp'].isoformat(),
                    'performance': {
                        'mape': usage['avg_mape'],
                        'rmse': 0,
                        'mae': 0
                    },
                    'status': 'active' if i == 0 else 'archived',
                    'update_type': 'usage'
                })
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'model': model,
            'history': history
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive/ensemble/weights/<symbol>', methods=['GET'])
def get_ensemble_weights(symbol):
    """Get current ensemble weights"""
    try:
        weights = ensemble_rebalancer.get_current_weights(symbol)
        
        if weights is None:
            return jsonify({
                'success': True,
                'symbol': symbol,
                'weights': None,
                'message': 'No weights found, using equal weights'
            })
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'weights': weights
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive/ensemble/rebalance', methods=['POST'])
def rebalance_ensemble():
    """Manually trigger ensemble rebalancing"""
    data = request.json
    symbol = data.get('symbol', 'AAPL')
    lookback_days = data.get('lookback_days', 7)
    
    try:
        weights = ensemble_rebalancer.rebalance_weights(symbol, lookback_days=lookback_days)
        return jsonify({
            'success': True,
            'symbol': symbol,
            'weights': weights
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive/ensemble/history/<symbol>', methods=['GET'])
def get_weight_history(symbol):
    """Get ensemble weight history"""
    days = request.args.get('days', 30, type=int)
    
    try:
        history = ensemble_rebalancer.get_weight_history(symbol, days=days)
        
        # Convert ObjectId and datetime to strings
        for doc in history:
            doc['_id'] = str(doc['_id'])
            if 'timestamp' in doc:
                doc['timestamp'] = doc['timestamp'].isoformat()
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'history': history
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive/training/logs/<symbol>/<model>', methods=['GET'])
def get_training_logs(symbol, model):
    """Get training logs for a model"""
    limit = request.args.get('limit', 20, type=int)
    
    try:
        logs = performance_tracker.get_training_history(symbol, model, limit=limit)
        
        # Convert ObjectId and datetime to strings
        for log in logs:
            log['_id'] = str(log['_id'])
            if 'training_started' in log:
                log['training_started'] = log['training_started'].isoformat()
            if 'training_completed' in log:
                log['training_completed'] = log['training_completed'].isoformat()
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'model': model,
            'logs': logs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive/simulation/start', methods=['POST'])
def start_simulation():
    """Start a simulation of adaptive learning"""
    data = request.json
    symbol = data.get('symbol', 'AAPL')
    model = data.get('model', 'lstm')
    
    try:
        # This will be used for real-time simulation
        # For now, just return success
        return jsonify({
            'success': True,
            'message': f'Simulation started for {symbol}/{model}',
            'symbol': symbol,
            'model': model
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive/prediction-errors/<symbol>/<model>', methods=['GET'])
def get_prediction_errors(symbol, model):
    """Get prediction errors (actual vs predicted) for error overlay visualization"""
    try:
        days = request.args.get('days', 30, type=int)
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get predictions with actual prices
        predictions = list(performance_tracker.performance_collection.find({
            'symbol': symbol,
            'model_name': model,
            'timestamp': {'$gte': cutoff_date}
        }).sort('timestamp', 1))
        
        error_data = []
        for pred in predictions:
            error_data.append({
                'date': pred['timestamp'].isoformat(),
                'actual': pred['actual_price'],
                'predicted': pred['predicted_price'],
                'error': pred['error'],
                'error_percentage': pred['percentage_error']
            })
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'model': model,
            'errors': error_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive/trained-models', methods=['GET'])
def get_trained_models():
    """Get list of all trained model-symbol combinations (including traditional models)"""
    try:
        # Get unique symbol-model combinations from performance history
        pipeline = [
            {
                '$group': {
                    '_id': {
                        'symbol': '$symbol',
                        'model_name': '$model_name'
                    },
                    'total_predictions': {'$sum': 1},
                    'recent_mape': {'$avg': '$percentage_error'},
                    'last_prediction': {'$max': '$timestamp'}
                }
            },
            {
                '$sort': {'_id.symbol': 1, '_id.model_name': 1}
            }
        ]
        
        results = list(performance_tracker.performance_collection.aggregate(pipeline))
        
        models = []
        for result in results:
            symbol = result['_id']['symbol']
            model_name = result['_id']['model_name']
            
            # Get version count (for neural models)
            version_count = version_manager.versions_collection.count_documents({
                'symbol': symbol,
                'model_name': model_name
            })
            
            # For traditional models, version count is based on usage
            if version_count == 0:
                version_count = 1  # Traditional models don't have versions but are used
            
            # Get days since last training/usage
            last_training = performance_tracker.training_logs.find_one(
                {'symbol': symbol, 'model_name': model_name},
                sort=[('training_started', -1)]
            )
            
            days_since_training = None
            if last_training:
                days_since_training = (datetime.utcnow() - last_training['training_started']).days
            else:
                # For traditional models without training logs, use last prediction time
                if result['last_prediction']:
                    days_since_training = (datetime.utcnow() - result['last_prediction']).days
            
            models.append({
                'symbol': symbol,
                'model_name': model_name,
                'total_predictions': result['total_predictions'],
                'recent_mape': result['recent_mape'],
                'last_prediction': result['last_prediction'].isoformat() if result['last_prediction'] else None,
                'version_count': version_count,
                'days_since_training': days_since_training
            })
        
        return jsonify({
            'success': True,
            'models': models,
            'count': len(models)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
