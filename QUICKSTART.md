# Quick Start Guide

## Prerequisites

1. **Python 3.10+** installed
2. **MongoDB** installed and running
3. **Git** (optional, for version control)

## Installation Steps

### Option 1: Local Setup (Recommended for Development)

1. **Clone or download the project**
   ```bash
   cd stock-forecasting
   ```

2. **Install MongoDB** (if not already installed)
   - **Windows**: Download from https://www.mongodb.com/try/download/community
   - **Mac**: `brew install mongodb-community`
   - **Linux**: `sudo apt-get install mongodb`

3. **Start MongoDB**
   ```bash
   # Windows
   mongod
   
   # Mac/Linux
   sudo systemctl start mongod
   ```

4. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the application**
   ```bash
   python backend/app.py
   ```

7. **Open browser**
   Navigate to: `http://localhost:5000`

### Option 2: Docker Setup (Recommended for Production)

1. **Install Docker and Docker Compose**

2. **Build and run**
   ```bash
   docker-compose up --build
   ```

3. **Open browser**
   Navigate to: `http://localhost:5000`

## Using the Application

### Step 1: Select a Symbol
Choose from:
- **Stocks**: AAPL, GOOGL, MSFT, TSLA, AMZN
- **Crypto**: BTC-USD, ETH-USD, BNB-USD
- **ForEx**: EURUSD=X, GBPUSD=X

### Step 2: Choose a Model
- **Ensemble**: Best overall performance (recommended)
- **ARIMA**: Statistical time series model
- **Moving Average**: Simple baseline
- **LSTM**: Neural network (slower but accurate)
- **GRU**: Neural network (faster than LSTM)

### Step 3: Select Forecast Horizon
- 1 Day, 3 Days, 7 Days
- 24 Hours, 72 Hours

### Step 4: Generate Forecast
Click "Generate Forecast" to see:
- Historical candlestick chart
- Predicted values overlaid
- Performance metrics (RMSE, MAE, MAPE)

### Step 5: Compare Models (Optional)
Click "Compare Models" to evaluate all models simultaneously

## Troubleshooting

### MongoDB Connection Error
```
Error: Failed to connect to MongoDB
```
**Solution**: Ensure MongoDB is running
```bash
# Check if MongoDB is running
# Windows
tasklist | findstr mongod

# Mac/Linux
ps aux | grep mongod
```

### Module Not Found Error
```
ModuleNotFoundError: No module named 'flask'
```
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Port Already in Use
```
Error: Address already in use
```
**Solution**: Change port in `backend/app.py` or kill process using port 5000
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:5000 | xargs kill -9
```

### yfinance Data Fetch Error
```
Error: Failed to fetch data
```
**Solution**: 
- Check internet connection
- Verify symbol is correct
- Try a different symbol

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=backend

# Run specific test file
pytest tests/test_models.py
```

## Project Structure

```
stock-forecasting/
├── backend/
│   ├── app.py              # Main Flask application
│   ├── database.py         # MongoDB operations
│   ├── data_fetcher.py     # yfinance integration
│   └── models/
│       ├── traditional.py  # ARIMA, MA, Ensemble
│       └── neural.py       # LSTM, GRU
├── frontend/
│   ├── static/
│   │   ├── app.js         # Frontend JavaScript
│   │   └── style.css      # Styling
│   └── templates/
│       └── index.html     # Main HTML page
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Multi-container setup
├── README.md              # Project overview
├── REPORT.md              # Technical report
└── QUICKSTART.md          # This file
```

## Next Steps

1. **Customize Models**: Edit `backend/models/` to add new forecasting models
2. **Add Symbols**: Modify `backend/app.py` to include more financial instruments
3. **Enhance UI**: Update `frontend/` files for custom styling
4. **Deploy**: Use Docker for production deployment

## Support

For issues or questions:
1. Check the REPORT.md for technical details
2. Review test files for usage examples
3. Consult yfinance documentation: https://pypi.org/project/yfinance/

## Performance Tips

1. **First Run**: Neural models (LSTM/GRU) take 30-60 seconds for training
2. **Subsequent Runs**: Models are cached in MongoDB and load instantly!
3. **Speed**: Use traditional models (Ensemble, ARIMA) for faster results
4. **Accuracy**: Use neural models with more historical data (1y+)

## Model Management

The application automatically caches trained neural models in MongoDB:

```bash
# List all cached models
python manage_models.py list

# View details of a specific model
python manage_models.py info AAPL lstm

# Delete a cached model (will retrain on next use)
python manage_models.py delete AAPL lstm

# Clear all cached models
python manage_models.py delete-all
```

**Benefits of Model Caching:**
- First prediction: 30-60 seconds (training time)
- Subsequent predictions: < 1 second (loads from cache)
- Models are symbol-specific (AAPL model won't be used for GOOGL)
- Automatic cache invalidation when retraining

Enjoy forecasting! 📈
