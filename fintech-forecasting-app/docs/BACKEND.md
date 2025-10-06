# Backend Documentation

## Overview

Flask-based REST API providing financial forecasting services with ML model integration, MongoDB caching, and real-time stock data fetching.

## Tech Stack

### Core Framework

- **Flask 3.0+** - Web framework
- **Python 3.11+** - Programming language
- **Flask-CORS** - Cross-origin resource sharing

### Data & ML

- **yfinance** - Stock market data
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **scikit-learn** - Machine learning utilities
- **statsmodels** - Statistical models (ARIMA)

### Database

- **MongoDB** - Document database
- **PyMongo** - MongoDB driver

### Additional Libraries

- **requests** - HTTP client
- **python-dotenv** - Environment variables
- **logging** - Application logging

## Project Structure

```
backend/
├── main.py                    # Flask application & API routes
├── run_server.py             # Server startup script
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
├── utils/
│   ├── database.py          # MongoDB operations
│   ├── config.py            # Config helper
│   └── data_loader.py       # Data loading utilities
├── ml/
│   ├── models.py            # ML model implementations
│   └── advanced.py          # Advanced model variants
├── logs/
│   └── app.log              # Application logs
└── test_endpoints.py         # API testing script
```

## API Endpoints

### 1. Health Check

```
GET /health
Response: {
  "status": "healthy",
  "timestamp": "2025-10-06T...",
  "version": "1.0.0"
}
```

### 2. Generate Forecast

```
POST /api/forecast/<symbol>
Headers: Content-Type: application/json
Body: {
  "model_type": "lstm",        # moving_average, arima, lstm, gru, transformer
  "horizon": 24,               # 1-72 hours
  "retrain": false             # optional
}

Response: {
  "symbol": "AAPL",
  "model_type": "lstm",
  "current_price": 258.01,
  "forecast_1h": {
    "direction": "up",
    "confidence": "medium",
    "predicted_price": 259.15,
    "price_range": [257.97, 260.34]
  },
  "forecast_4h": { ... },
  "forecast_24h": { ... },
  "technical_indicators": {
    "sma_7": 258.21,
    "rsi": 73.84,
    "bb_upper": 259.17,
    "bb_lower": 255.21,
    "volatility_pct": 0.55,
    "momentum_5d_pct": -0.20,
    "support_level": 253.11,
    "resistance_level": 259.23
  },
  "forecast": [
    {
      "timestamp": "2025-10-06T03:37:08...",
      "predicted_price": 259.15,
      "price_range_low": 257.72,
      "price_range_high": 260.58,
      "confidence": "high"
    },
    // ... more forecast points
  ],
  "data_points_used": 49,
  "data_period": "7 days hourly",
  "created_at": "2025-10-06T...",
  "cached": false
}
```

### 3. List Available Models

```
GET /api/models
Response: {
  "models": [
    {
      "type": "moving_average",
      "name": "Moving Average",
      "description": "Simple moving average forecast",
      "traditional": true
    },
    {
      "type": "arima",
      "name": "ARIMA",
      "description": "AutoRegressive Integrated Moving Average",
      "traditional": true
    },
    {
      "type": "lstm",
      "name": "LSTM",
      "description": "Long Short-Term Memory neural network",
      "traditional": false
    },
    {
      "type": "gru",
      "name": "GRU",
      "description": "Gated Recurrent Unit neural network",
      "traditional": false
    },
    {
      "type": "transformer",
      "name": "Transformer",
      "description": "Transformer-based neural network",
      "traditional": false
    }
  ]
}
```

### 4. Get Historical Data

```
GET /api/data/<symbol>?limit=1000
Response: {
  "symbol": "AAPL",
  "data": [
    {
      "timestamp": "2025-10-06T10:00:00",
      "open": 257.50,
      "high": 259.20,
      "low": 256.80,
      "close": 258.01,
      "volume": 1000000
    },
    // ... more data points
  ],
  "count": 49
}
```

### 5. Get Model Performance

```
GET /api/performance/<symbol>
Response: {
  "symbol": "AAPL",
  "performances": [
    {
      "model_type": "lstm",
      "metrics": {
        "mae": 2.45,
        "mse": 8.32,
        "rmse": 2.88,
        "mape": 0.95,
        "direction_accuracy": 78.5
      },
      "evaluation_date": "2025-10-06T...",
      "data_points": 100
    }
  ]
}
```

### 6. Compare Models

```
POST /api/compare-models/<symbol>
Body: {
  "models": ["moving_average", "lstm", "arima"],
  "horizon": 24
}
Response: {
  "symbol": "AAPL",
  "comparisons": [
    {
      "model_type": "lstm",
      "predictions": [...],
      "metrics": {...}
    }
  ]
}
```

### 7. Clear Cache

```
POST /api/cache/clear
Body: {
  "symbol": "AAPL"  # optional, clears all if not provided
}
Response: {
  "status": "success",
  "message": "Cache cleared for AAPL"
}
```

## MongoDB Caching System

### Architecture

#### First API Call (Fresh Data)

```
1. Client requests AAPL forecast
2. Backend checks: is_first_api_call("AAPL") → True
3. Fetch from yfinance API
4. Store data in MongoDB (stock_data_cache)
5. Train model
6. Store model in MongoDB (trained_models)
7. Return forecast
```

#### Subsequent API Calls (Cached)

```
1. Client requests AAPL forecast
2. Backend checks: is_first_api_call("AAPL") → False
3. Load cached data from MongoDB (if < 6 hours old)
4. Load cached model from MongoDB (if < 24 hours old)
5. Generate forecast
6. Return forecast (90% faster!)
```

### MongoDB Collections

#### 1. `trained_models`

```python
{
  "_id": ObjectId(...),
  "symbol": "AAPL",
  "model_type": "lstm",
  "model_data": Binary(...),  # Pickled model
  "metadata": {
    "trained_at": ISODate(...),
    "model_class": "SimpleLSTM"
  },
  "created_at": ISODate(...)
}
```

**Indexes:**

```python
[("symbol", 1), ("model_type", 1), ("created_at", -1)]
```

#### 2. `stock_data_cache`

```python
{
  "_id": ObjectId(...),
  "symbol": "AAPL",
  "data_type": "7d_hourly",
  "data": [
    {
      "timestamp": "2025-10-06 10:00:00",
      "close": 258.01,
      "high": 259.23,
      "low": 257.11,
      "volume": 1000000
    }
  ],
  "created_at": ISODate(...),
  "count": 49
}
```

**Indexes:**

```python
[("symbol", 1), ("data_type", 1), ("created_at", -1)]
```

#### 3. `forecasts`

```python
{
  "symbol": "AAPL",
  "model_type": "lstm",
  "predictions": [...],
  "metrics": {...},
  "created_at": ISODate(...)
}
```

#### 4. `historical_prices`

```python
{
  "symbol": "AAPL",
  "timestamp": ISODate(...),
  "open_price": 257.50,
  "high": 259.20,
  "low": 256.80,
  "close": 258.01,
  "volume": 1000000
}
```

### Database Functions

```python
# Model caching
store_trained_model(symbol, model_type, model_data, metadata)
get_trained_model(symbol, model_type, max_age_hours=24)

# Stock data caching
store_stock_data_cache(symbol, data, data_type="7d_hourly")
get_stock_data_cache(symbol, data_type="7d_hourly", max_age_hours=6)

# Session tracking
is_first_api_call(symbol)  # Returns True for first call in session
reset_first_api_calls()    # Reset tracker

# Historical data
store_price_data(price_data)
get_price_data(symbol, limit=1000)

# Forecasts
store_forecast(forecast)
get_latest_forecast(symbol, model_type)
```

## Configuration

### Environment Variables

```bash
# .env file
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB_NAME=fintech_forecasting
FLASK_ENV=development
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000
PORT=5000
```

### Config Class

```python
class Config:
    DATABASE_URL = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'fintech_forecasting')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.path.join(BASE_DIR, 'logs', 'app.log')
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
```

## Data Flow

### Forecast Generation Pipeline

```
1. Validate Input
   ↓
2. Check if First API Call
   ↓
3. Load/Fetch Stock Data
   - First call: yfinance API → MongoDB cache
   - Subsequent: MongoDB cache → Use if fresh
   ↓
4. Calculate Technical Indicators
   - SMA-7, RSI, Bollinger Bands
   - Volatility, Momentum
   - Support/Resistance levels
   ↓
5. Load/Train ML Model
   - Check MongoDB for cached model
   - If fresh: Use cached model
   - If stale: Train new model → MongoDB cache
   ↓
6. Generate Predictions
   - Short-term: 1h, 4h, 24h forecasts
   - Full timeline: hourly predictions
   - Confidence intervals
   ↓
7. Determine Direction & Confidence
   - Based on price change and volatility
   - Classify: up/down/sideways
   - Confidence: high/medium/low
   ↓
8. Return Forecast Response
```

## Error Handling

### Error Response Format

```python
{
  "error": "Error message",
  "details": "Detailed error information"
}
```

### HTTP Status Codes

- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found
- `500` - Internal Server Error

### Logging

```python
# Configured in main.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
```

## Performance Metrics

### Without Caching

- API call: ~2-3 seconds
- Model training: ~3-5 seconds
- Total: ~5-10 seconds per request

### With Caching

- First request: ~5-10 seconds (fresh data + training)
- Cached request: ~200-500ms (MongoDB lookup)
- **Improvement: 90-95% faster**

## Development

### Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Server

```bash
python run_server.py
```

Server runs on: http://localhost:5000

### Testing

```bash
# Test all endpoints
python test_endpoints.py

# Test specific model
python test_transformer.py

# Test all models
python test_all_models.py
```

### MongoDB Setup

```bash
# Install MongoDB locally or use Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Or use MongoDB Atlas (cloud)
# Update MONGODB_URI in .env
```

## Best Practices

1. **Always validate input parameters**
2. **Use logging for debugging**
3. **Cache aggressively but intelligently**
4. **Handle edge cases (insufficient data, API failures)**
5. **Return meaningful error messages**
6. **Monitor performance metrics**
7. **Keep models fresh (24-hour TTL)**
8. **Use transactions for critical operations**

## Security

1. **CORS Configuration**: Restrict allowed origins
2. **Input Validation**: Sanitize all user inputs
3. **Rate Limiting**: Implement in production
4. **API Keys**: Store securely in environment variables
5. **MongoDB**: Use authentication in production

## Deployment

### Production Checklist

- [ ] Set `FLASK_ENV=production`
- [ ] Configure proper MongoDB authentication
- [ ] Set up reverse proxy (nginx)
- [ ] Enable HTTPS
- [ ] Configure rate limiting
- [ ] Set up monitoring and logging
- [ ] Use production WSGI server (gunicorn)

### Docker Deployment

```bash
docker-compose up -d
```

## Troubleshooting

### MongoDB Connection Issues

```python
# Check MongoDB is running
mongosh

# Check connection string
MONGODB_URI=mongodb://localhost:27017/
```

### Model Training Errors

- Check sufficient data points (min 20)
- Verify data format (numpy arrays)
- Check for NaN values

### Cache Not Working

- Verify MongoDB indexes created
- Check collection names
- Verify TTL settings

## Future Enhancements

- [ ] WebSocket support for real-time updates
- [ ] Advanced model ensemble methods
- [ ] Backtesting framework
- [ ] Model performance tracking dashboard
- [ ] Automated model retraining
- [ ] Multi-symbol batch forecasting
- [ ] Redis caching layer
