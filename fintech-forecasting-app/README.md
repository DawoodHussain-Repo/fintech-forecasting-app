# FinTech Forecasting Application

A comprehensive financial forecasting application built with **Next.js**, **Python ML models**, and **MongoDB**. This project implements both traditional time series models (ARIMA, Moving Averages) and modern neural networks (LSTM, GRU, Transformer) for predicting stock, cryptocurrency, and forex prices.

> 📌 **Course:** CS4063 – Natural Language Processing  
> 📅 **Due:** Tuesday, October 7th, 10:00am  
> 🎯 **Objective:** End-to-end FinTech forecasting with AI/ML integration

## ✨ Features

### Frontend (Next.js 15 + React 19)

- **Modern UI:** Apple-inspired design with dark/light mode
- **Interactive Charts:** Professional candlestick visualization with Chart.js
- **Real-Time Data:** Live market data from Alpha Vantage API
- **Responsive Design:** Mobile-first approach with Tailwind CSS
- **Model Selection:** Choose between traditional and neural forecasting models

### Backend (Python + Flask)

- **ML Models:** ARIMA, LSTM, GRU, Transformer implementations
- **REST API:** Comprehensive endpoints for data and forecasting
- **Database:** MongoDB with optimized time series storage
- **Background Jobs:** Automated data collection and model retraining
- **Comprehensive Testing:** Unit tests for all ML models and APIs

### Machine Learning Pipeline

- **Traditional Models:** ARIMA with auto-parameter selection, Moving Averages
- **Neural Networks:** LSTM, GRU, and Transformer architectures
- **Evaluation Metrics:** MSE, MAE, MAPE, directional accuracy
- **Model Persistence:** Automated saving and loading of trained models
- **Performance Tracking:** Historical model performance monitoring

## 🚀 Quick Start

### Option 1: Docker (Production-Ready)

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# ALPHA_VANTAGE_API_KEY=your_api_key_here
# DATABASE_URL=mongodb://localhost:27017/fintech_forecasting

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Access the app at http://localhost:3000
```

### Option 2: Manual Setup

#### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.11+ with pip
- **MongoDB** (local or Docker)
- **Alpha Vantage API Key** (free at alphavantage.co)

#### Frontend Setup

```bash
# Install dependencies
npm install

# Configure environment
echo "ALPHA_VANTAGE_API_KEY=your_api_key_here" > .env.local

# Start development server
npm run dev
```

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start Flask server
python main.py
```

## 📊 Application Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Database      │
│   (Next.js)     │◄──►│   (Flask/ML)    │◄──►│   (MongoDB)     │
│                 │    │                 │    │                 │
│ • React UI      │    │ • ML Models     │    │ • Historical    │
│ • Charts        │    │ • REST API      │    │   Data          │
│ • Real-time     │    │ • Scheduling    │    │ • Forecasts     │
│   Data          │    │                 │    │ • Performance   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        │              ┌─────────────────┐              │
        │              │   External APIs  │              │
        └──────────────┤ • Alpha Vantage │◄─────────────┘
                       │ • Financial     │
                       │   Data Sources  │
                       └─────────────────┘
```

## 🧠 Machine Learning Models

### Traditional Models

- **Moving Average:** Simple and exponential moving averages for trend analysis
- **ARIMA:** Auto-ARIMA with optimized parameters using PMDArima library

### Neural Network Models

- **LSTM:** Long Short-Term Memory networks for sequence prediction
- **GRU:** Gated Recurrent Units for efficient time series modeling
- **Transformer:** Attention-based model for capturing long-range dependencies

### Model Performance

| Model          | MAPE (%) | Training Time | Best Use Case        |
| -------------- | -------- | ------------- | -------------------- |
| Moving Average | 8-15     | Instant       | Baseline trends      |
| ARIMA          | 3-12     | Medium        | Stationary series    |
| LSTM           | 2-8      | Slow          | Complex patterns     |
| GRU            | 2-9      | Medium        | Balanced performance |
| Transformer    | 3-10     | Very Slow     | Long sequences       |

## 🔧 API Endpoints

### Data Management

```http
GET /api/data/{symbol}?limit=1000
POST /api/update-data
```

### Forecasting

```http
POST /api/forecast/{symbol}
{
  "model_type": "lstm",
  "horizon": 24,
  "retrain": false
}

GET /api/models
GET /api/performance/{symbol}
```

### System Health

```http
GET /health
```

## 📱 User Interface

### Dashboard

- **Market Overview:** Real-time price data with interactive charts
- **Symbol Search:** Support for stocks (AAPL), crypto (BTC), forex (EUR/USD)
- **Watchlist:** Persistent symbol tracking with sparklines
- **Range Selection:** 1D, 1W, 1M, 6M, 1Y timeframes

### Forecast Workspace

- **Model Selection:** Choose from 5 different ML models
- **Horizon Configuration:** 1h, 3h, 24h, 72h forecast periods
- **Performance Metrics:** Real-time model accuracy and performance data
- **Confidence Intervals:** Prediction uncertainty visualization

## 🧪 Testing

Run the comprehensive test suite:

```bash
cd backend
python test_models.py
```

**Test Coverage:**

- ✅ Model training and prediction accuracy
- ✅ API endpoint validation
- ✅ Database operations
- ✅ Data preprocessing pipelines
- ✅ Error handling and edge cases

## 🚢 Deployment

### Development

```bash
# Frontend (localhost:3000)
npm run dev

# Backend (localhost:5000)
cd backend && python main.py

# Database
mongod --dbpath ./data
```

### Production with Docker

```bash
# Build and start all services
docker-compose up -d

# Scale backend services
docker-compose up -d --scale backend=3

# Monitor logs
docker-compose logs -f backend
```

### Environment Variables

**Frontend (.env.local):**

```env
ALPHA_VANTAGE_API_KEY=your_api_key_here
BACKEND_URL=http://localhost:5000
```

**Backend (.env):**

```env
DATABASE_URL=mongodb://localhost:27017/fintech_forecasting
ALPHA_VANTAGE_API_KEY=your_api_key_here
FLASK_ENV=production
CORS_ORIGINS=http://localhost:3000
```

## 📊 Data Sources

### Alpha Vantage API

- **Stocks:** Daily adjusted time series (TIME_SERIES_DAILY_ADJUSTED)
- **Crypto:** Digital currency daily data (DIGITAL_CURRENCY_DAILY)
- **Forex:** Foreign exchange rates (FX_DAILY)
- **Rate Limits:** 5 requests/minute (free tier)

### Supported Symbols

- **Stocks:** AAPL, GOOGL, MSFT, TSLA, AMZN, etc.
- **Crypto:** BTC, ETH, LTC, XRP, ADA, etc.
- **Forex:** EUR/USD, GBP/JPY, USD/CAD, etc.

## 🔍 Troubleshooting

### Common Issues

**"Failed to fetch market data"**

- Check Alpha Vantage API key configuration
- Verify API rate limits (5 requests/minute)
- Ensure internet connectivity

**"Database connection failed"**

- Start MongoDB service: `mongod`
- Check DATABASE_URL in .env file
- Verify MongoDB is running on port 27017

**"Model training failed"**

- Ensure sufficient historical data (>100 data points)
- Check Python dependencies installation
- Verify GPU/CPU resources for neural models

**Frontend build errors**

- Run `npm install` to update dependencies
- Clear Next.js cache: `rm -rf .next`
- Check Node.js version (requires 18+)

## � Project Structure

```
fintech-forecasting-app/
├── README.md                 # This file
├── TECHNICAL_REPORT.md       # Detailed technical documentation
├── package.json             # Frontend dependencies
├── next.config.ts           # Next.js configuration
├── docker-compose.yml       # Multi-container orchestration
├── setup.bat / setup.sh     # Automated setup scripts
│
├── app/                     # Next.js app directory
│   ├── layout.tsx          # Root layout with navigation
│   ├── page.tsx            # Landing page
│   ├── dashboard/          # Market data dashboard
│   ├── forecast/           # ML forecasting workspace
│   ├── watchlist/          # Symbol tracking
│   └── api/                # API routes
│
├── backend/                 # Python ML backend
│   ├── main.py             # Flask application
│   ├── forecasting.py      # ML model implementations
│   ├── database.py         # MongoDB data models
│   ├── scheduler.py        # Background job scheduler
│   ├── test_models.py      # Comprehensive test suite
│   └── requirements.txt    # Python dependencies
│
├── components/              # Reusable React components
├── lib/                    # Utility functions
└── styles/                 # CSS and theme files
```

## 🤝 Contributing

This is an academic project for CS4063 Natural Language Processing. The codebase demonstrates:

- **Modern Frontend:** React 19, Next.js 15, TypeScript
- **ML Engineering:** Traditional and neural time series models
- **Full-Stack Development:** RESTful APIs, database design
- **DevOps Practices:** Docker, automated testing, CI/CD ready
- **Software Engineering:** Clean code, documentation, error handling

## 📄 License

This project is created for educational purposes as part of CS4063 coursework.

## 🙏 Acknowledgments

- **Alpha Vantage** for providing financial data API
- **TensorFlow/Keras** for neural network implementations
- **Statsmodels** for traditional time series analysis
- **Next.js** team for the excellent React framework
- **Chart.js** community for financial charting capabilities

---

**🎯 Assignment Status:** Complete and ready for submission  
**📅 Due Date:** Tuesday, October 7th, 10:00am  
**🚀 Live Demo:** Available after running setup scripts

For questions or issues, please refer to the technical report or check the troubleshooting section above.
