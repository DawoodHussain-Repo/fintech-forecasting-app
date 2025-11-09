# Stock/Crypto/ForEx Forecasting Application

AI-powered financial forecasting application using yfinance for data and multiple ML approaches including traditional time series models and neural networks.

## Features

- **Data Source**: Real-time data fetching using yfinance
- **Traditional Models**: ARIMA, Moving Average, Exponential Smoothing, Ensemble
- **Neural Models**: LSTM, GRU
- **Database**: MongoDB for storing historical data and predictions
- **Visualization**: Interactive candlestick charts with Plotly
- **Web Interface**: Clean Flask-based UI

## Architecture

```
Frontend (HTML/CSS/JS) → Flask API → ML Models → MongoDB
                                   ↓
                              yfinance API
```

## Installation

### Prerequisites
- Python 3.10+
- MongoDB (local or cloud)

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start MongoDB (if running locally):
```bash
mongod
```

3. Run the application:
```bash
python backend/app.py
```

4. Open browser to `http://localhost:5000`

## Docker Deployment

```bash
docker build -t stock-forecasting .
docker run -p 5000:5000 stock-forecasting
```

## Usage

1. Select a financial instrument (Stock/Crypto/ForEx)
2. Choose a forecasting model
3. Select forecast horizon (1h, 3h, 24h, 72h)
4. Click "Generate Forecast" to see predictions
5. Use "Compare Models" to evaluate multiple approaches

## Models Implemented

### Traditional Models
- **ARIMA**: Auto-regressive Integrated Moving Average
- **Moving Average**: Simple moving average forecasting
- **Exponential Smoothing**: Weighted moving average
- **Ensemble**: Combination of multiple traditional models

### Neural Models
- **LSTM**: Long Short-Term Memory networks
- **GRU**: Gated Recurrent Units

**Model Caching**: Neural models are automatically saved to MongoDB after training and reused for subsequent predictions, significantly improving performance.

## Performance Metrics

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

## Project Structure

```
stock-forecasting/
├── backend/
│   ├── app.py              # Flask API
│   ├── database.py         # MongoDB integration
│   ├── data_fetcher.py     # yfinance wrapper
│   └── models/
│       ├── traditional.py  # Traditional ML models
│       └── neural.py       # Neural network models
├── frontend/
│   ├── static/
│   │   ├── app.js         # Frontend logic
│   │   └── style.css      # Styling
│   └── templates/
│       └── index.html     # Main page
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

## Model Management

Trained neural models are automatically cached in MongoDB. To manage stored models:

```bash
# List all stored models
python manage_models.py list

# View model details
python manage_models.py info AAPL lstm

# Delete a specific model
python manage_models.py delete AAPL lstm

# Delete all models
python manage_models.py delete-all
```

## Testing

```bash
pytest tests/
```

## License

MIT
