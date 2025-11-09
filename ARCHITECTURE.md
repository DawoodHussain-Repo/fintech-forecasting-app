# System Architecture Documentation

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Web Browser (Chrome, Firefox, Safari, Edge)          │     │
│  │  - HTML5 + CSS3                                        │     │
│  │  - JavaScript (ES6+)                                   │     │
│  │  - Plotly.js for visualization                         │     │
│  └────────────────────────────────────────────────────────┘     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP/HTTPS
                            │ REST API (JSON)
┌───────────────────────────▼─────────────────────────────────────┐
│                      APPLICATION LAYER                           │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              Flask Web Framework                       │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │     │
│  │  │   Routes     │  │  Controllers │  │  Middleware │ │     │
│  │  │  /api/...    │  │              │  │   CORS      │ │     │
│  │  └──────────────┘  └──────────────┘  └─────────────┘ │     │
│  └────────────────────────────────────────────────────────┘     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌───────▼────────┐  ┌──────▼──────┐
│  DATA LAYER    │  │  MODEL LAYER   │  │ STORAGE     │
│                │  │                │  │ LAYER       │
│ ┌────────────┐ │  │ ┌────────────┐ │  │ ┌─────────┐ │
│ │ yfinance   │ │  │ │Traditional │ │  │ │ MongoDB │ │
│ │ API Client │ │  │ │  Models    │ │  │ │         │ │
│ │            │ │  │ │ - ARIMA    │ │  │ │ ┌─────┐ │ │
│ │ - Stocks   │ │  │ │ - MA       │ │  │ │ │Hist │ │ │
│ │ - Crypto   │ │  │ │ - Ensemble │ │  │ │ │Data │ │ │
│ │ - ForEx    │ │  │ └────────────┘ │  │ │ └─────┘ │ │
│ └────────────┘ │  │                │  │ │         │ │
│                │  │ ┌────────────┐ │  │ │ ┌─────┐ │ │
│ ┌────────────┐ │  │ │  Neural    │ │  │ │ │Pred │ │ │
│ │Data        │ │  │ │  Networks  │ │  │ │ │Data │ │ │
│ │Processor   │ │  │ │ - LSTM     │ │  │ │ └─────┘ │ │
│ │            │ │  │ │ - GRU      │ │  │ │         │ │
│ │- Normalize │ │  │ │ - PyTorch  │ │  │ │ ┌─────┐ │ │
│ │- Features  │ │  │ └────────────┘ │  │ │ │Meta │ │ │
│ └────────────┘ │  │                │  │ │ │Data │ │ │
└────────────────┘  └────────────────┘  │ │ └─────┘ │ │
                                        │ └─────────┘ │
                                        └─────────────┘
```

## Component Details

### 1. Client Layer

**Technologies:**
- HTML5 for structure
- CSS3 for styling (gradient backgrounds, responsive design)
- Vanilla JavaScript for interactivity
- Plotly.js for candlestick charts

**Responsibilities:**
- User interface rendering
- User input collection
- API communication
- Data visualization
- Real-time chart updates

**Key Files:**
- `frontend/templates/index.html` - Main HTML structure
- `frontend/static/style.css` - Styling and layout
- `frontend/static/app.js` - Client-side logic

### 2. Application Layer

**Technologies:**
- Flask 3.0 (Python web framework)
- Flask-CORS for cross-origin requests

**API Endpoints:**

```
GET  /                          → Render main page
GET  /api/symbols               → Get available symbols
POST /api/fetch_data            → Fetch historical data
POST /api/forecast              → Generate forecast
POST /api/compare_models        → Compare all models
GET  /api/historical/<symbol>   → Get stored historical data
```

**Request/Response Flow:**

```
Client Request
    ↓
Flask Route Handler
    ↓
Data Validation
    ↓
Business Logic
    ├→ Fetch Data (yfinance)
    ├→ Process Data
    ├→ Run Model
    └→ Store Results (MongoDB)
    ↓
JSON Response
    ↓
Client Receives & Renders
```

**Key Files:**
- `backend/app.py` - Main Flask application and routes

### 3. Data Layer

**Data Fetcher Module:**

```python
class DataFetcher:
    - fetch_data(symbol, period, interval)
    - get_latest_price(symbol)
    - prepare_for_model(df)
    - get_symbol_info(symbol)
```

**Data Flow:**

```
yfinance API
    ↓
Raw OHLCV Data
    ↓
Data Cleaning
    ↓
Feature Engineering
    ├→ Returns
    ├→ Moving Averages
    ├→ Volatility
    └→ RSI
    ↓
Normalized Data
    ↓
Model Input
```

**Key Files:**
- `backend/data_fetcher.py` - yfinance integration

### 4. Model Layer

**Traditional Models Architecture:**

```
Input: Time Series [t1, t2, ..., tn]
    ↓
┌─────────────────────────────────────┐
│  Traditional Forecaster             │
│                                     │
│  ┌──────────────┐                  │
│  │ Moving Avg   │ → Prediction 1   │
│  └──────────────┘                  │
│                                     │
│  ┌──────────────┐                  │
│  │ ARIMA        │ → Prediction 2   │
│  └──────────────┘                  │
│                                     │
│  ┌──────────────┐                  │
│  │ Exp Smooth   │ → Prediction 3   │
│  └──────────────┘                  │
│                                     │
│  ┌──────────────┐                  │
│  │ Ensemble     │ → Average        │
│  └──────────────┘                  │
└─────────────────────────────────────┘
    ↓
Output: Forecast [t(n+1), ..., t(n+k)]
```

**Neural Network Architecture:**

```
Input Sequence [60 time steps]
    ↓
Normalization (MinMaxScaler)
    ↓
┌─────────────────────────────────────┐
│  LSTM/GRU Model                     │
│                                     │
│  Input Layer (1 feature)            │
│         ↓                           │
│  LSTM/GRU Layer 1 (64 units)        │
│         ↓                           │
│  Dropout (0.2)                      │
│         ↓                           │
│  LSTM/GRU Layer 2 (64 units)        │
│         ↓                           │
│  Dropout (0.2)                      │
│         ↓                           │
│  Fully Connected (1 output)         │
└─────────────────────────────────────┘
    ↓
Denormalization
    ↓
Output: Next Price Prediction
```

**Key Files:**
- `backend/models/traditional.py` - Traditional forecasting models
- `backend/models/neural.py` - Neural network models

### 5. Storage Layer

**Database: MongoDB**

**Collections:**

1. **historical_data**
```json
{
  "_id": ObjectId,
  "symbol": "AAPL",
  "date": ISODate,
  "open": 150.0,
  "high": 152.0,
  "low": 149.0,
  "close": 151.0,
  "volume": 1000000
}
```
**Indexes:** `(symbol, date)`

2. **predictions**
```json
{
  "_id": ObjectId,
  "symbol": "AAPL",
  "model_name": "lstm",
  "forecast_horizon": "24h",
  "predictions": [
    {"date": ISODate, "predicted_close": 152.5},
    ...
  ],
  "metrics": {
    "rmse": 2.5,
    "mae": 1.8,
    "mape": 1.2
  },
  "created_at": ISODate
}
```
**Indexes:** `(symbol, model_name, created_at)`

3. **metadata**
```json
{
  "_id": ObjectId,
  "symbol": "AAPL",
  "name": "Apple Inc.",
  "currency": "USD",
  "exchange": "NASDAQ",
  "updated_at": ISODate
}
```
**Indexes:** `(symbol)`

**Key Files:**
- `backend/database.py` - MongoDB operations

## Data Flow Diagram

### Forecast Generation Flow

```
User Action: Click "Generate Forecast"
    ↓
1. Frontend collects inputs
   - Symbol: AAPL
   - Model: LSTM
   - Horizon: 24h
    ↓
2. POST /api/forecast
    ↓
3. Backend receives request
    ↓
4. Check MongoDB for cached data
    ├→ Found: Use cached
    └→ Not found: Fetch from yfinance
    ↓
5. Data preprocessing
   - Clean data
   - Add features
   - Normalize
    ↓
6. Model execution
   - Load/train model
   - Generate predictions
   - Calculate metrics
    ↓
7. Store results in MongoDB
    ↓
8. Return JSON response
   - Historical data (100 points)
   - Predictions (24 points)
   - Metrics (RMSE, MAE, MAPE)
    ↓
9. Frontend renders
   - Candlestick chart
   - Forecast overlay
   - Metrics display
    ↓
User sees visualization
```

## Deployment Architecture

### Development Environment

```
Developer Machine
├── Python 3.10+
├── MongoDB (local)
├── Virtual Environment
└── Flask Dev Server (port 5000)
```

### Docker Environment

```
Docker Host
├── Container: MongoDB
│   └── Port: 27017
│
└── Container: Flask App
    ├── Port: 5000
    └── Linked to MongoDB
```

### Production Environment (Recommended)

```
Cloud Platform (AWS/GCP/Azure)
├── Load Balancer
│   └── SSL/TLS Termination
│
├── Application Servers (Multiple)
│   ├── Flask + Gunicorn
│   ├── Auto-scaling
│   └── Health checks
│
├── Database Cluster
│   ├── MongoDB Atlas
│   ├── Replica Set
│   └── Automated backups
│
└── CDN
    └── Static assets (CSS, JS)
```

## Security Considerations

1. **API Security**
   - CORS configuration
   - Rate limiting (recommended)
   - Input validation
   - SQL injection prevention (using MongoDB)

2. **Data Security**
   - MongoDB authentication
   - Encrypted connections
   - Environment variables for secrets

3. **Application Security**
   - No sensitive data in client
   - Secure session management
   - HTTPS in production

## Performance Optimization

1. **Caching Strategy**
   - Store predictions in MongoDB
   - Cache historical data
   - Reuse trained models

2. **Database Optimization**
   - Proper indexing
   - Query optimization
   - Connection pooling

3. **Model Optimization**
   - Batch predictions
   - Model quantization (future)
   - GPU acceleration (optional)

## Scalability

**Horizontal Scaling:**
- Multiple Flask instances behind load balancer
- MongoDB replica set
- Stateless application design

**Vertical Scaling:**
- Increase server resources
- GPU for neural models
- More MongoDB RAM

## Monitoring & Logging

**Recommended Tools:**
- Application: Flask logging
- Database: MongoDB logs
- Performance: Prometheus + Grafana
- Errors: Sentry

**Key Metrics:**
- Request latency
- Model inference time
- Database query time
- Error rates
- Prediction accuracy

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | HTML/CSS/JS | User interface |
| Visualization | Plotly.js | Candlestick charts |
| Backend | Flask | Web framework |
| Data Source | yfinance | Financial data |
| Database | MongoDB | Data persistence |
| ML Framework | PyTorch | Neural networks |
| Statistics | statsmodels | Traditional models |
| Data Processing | pandas/numpy | Data manipulation |
| Testing | pytest | Unit tests |
| Containerization | Docker | Deployment |

## Conclusion

This architecture provides:
- **Modularity**: Easy to extend and maintain
- **Scalability**: Can handle increased load
- **Reliability**: Robust error handling
- **Performance**: Optimized data flow
- **Maintainability**: Clear separation of concerns
