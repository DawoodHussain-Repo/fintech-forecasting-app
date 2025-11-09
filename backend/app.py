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

app = Flask(__name__, template_folder='../frontend/templates', 
            static_folder='../frontend/static')
CORS(app)

# Initialize components
data_fetcher = DataFetcher()
db = Database()
traditional_forecaster = TraditionalForecaster()
neural_forecaster = NeuralForecaster(db=db)

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
    """Generate forecast for a symbol"""
    data = request.json
    symbol = data.get('symbol', 'AAPL')
    model_type = data.get('model', 'ensemble')  # ensemble, arima, lstm, gru
    horizon = data.get('horizon', '24h')  # 1h, 3h, 24h, 72h
    
    # Parse horizon
    horizon_map = {'1h': 1, '3h': 3, '24h': 24, '72h': 72, '1d': 1, '3d': 3, '7d': 7}
    steps = horizon_map.get(horizon, 24)
    
    # Always fetch fresh data from yfinance for real-time predictions
    print(f"Fetching fresh data for {symbol}...")
    df = data_fetcher.fetch_data(symbol, period='1y', interval='1d')
    
    if df is None or len(df) < 100:
        return jsonify({'error': 'Insufficient data for forecasting'}), 400
    
    # Extract close prices
    close_prices = df['close']
    
    # Generate forecast based on model type
    try:
        if model_type == 'arima':
            predictions, metrics = traditional_forecaster.arima_forecast(close_prices, steps=steps)
        elif model_type == 'ma':
            predictions, metrics = traditional_forecaster.moving_average_forecast(close_prices, steps=steps)
        elif model_type == 'ensemble':
            predictions, metrics = traditional_forecaster.ensemble_forecast(close_prices, steps=steps)
        elif model_type == 'lstm':
            predictions, metrics = neural_forecaster.lstm_forecast(
                close_prices, steps=steps, epochs=30, symbol=symbol, use_cache=True
            )
        elif model_type == 'gru':
            predictions, metrics = neural_forecaster.gru_forecast(
                close_prices, steps=steps, epochs=30, symbol=symbol, use_cache=True
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
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'model': model_type,
            'horizon': horizon,
            'historical_data': historical_data,
            'predictions': prediction_records,
            'metrics': metrics,
            'latest_data_time': latest_data_time,
            'prediction_time': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
