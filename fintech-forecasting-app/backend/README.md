# Backend Structure

Clean and organized backend for the FinTech Forecasting App.

## Directory Structure

```
backend/
├── api/                    # API routes and endpoints
│   ├── __init__.py
│   └── routes.py          # Flask route handlers
│
├── ml/                     # Machine Learning models
│   ├── __init__.py
│   ├── models.py          # Main forecasting models (ARIMA, Moving Average)
│   └── advanced.py        # Advanced models (LSTM, GRU, Transformer)
│
├── utils/                  # Utilities and helpers
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── database.py        # MongoDB operations
│   └── data_loader.py     # Data fetching and processing
│
├── models/                 # Saved model files
│   └── *.pkl              # Trained model artifacts
│
├── logs/                   # Application logs
│   └── app.log
│
├── main.py                 # Flask application entry point
├── run_server.py          # Server runner
├── scheduler.py           # Background jobs
├── test_models.py         # Unit tests
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables
```

## Key Files

### `main.py`

- Flask application setup
- REST API endpoints
- Request handling

### `ml/models.py`

- ARIMA forecasting
- Moving Average models
- Model training and prediction

### `ml/advanced.py`

- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Transformer networks

### `utils/database.py`

- MongoDB connection
- Data models (PriceData, ForecastData, ModelPerformance)
- CRUD operations

### `utils/config.py`

- Environment configuration
- API keys
- Model parameters

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python main.py

# Run background scheduler
python scheduler.py

# Run tests
python test_models.py
```

## API Endpoints

- `GET /health` - Health check
- `GET /api/data/{symbol}` - Get historical data
- `POST /api/forecast/{symbol}` - Generate forecast
- `GET /api/models` - List available models
- `GET /api/performance/{symbol}` - Model performance metrics
