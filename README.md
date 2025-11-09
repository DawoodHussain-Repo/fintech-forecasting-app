# Stock/Crypto/ForEx Forecasting Application

AI-powered financial forecasting application using traditional time series models and neural networks.

**Student:** Dawood Hussain (22i-2410)  
**Course:** NLP Section A

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start MongoDB
mongod

# Run application
python backend/app.py

# Open browser
http://localhost:5000
```

---

## 📊 Features

- **Multiple Models:** ARIMA, Moving Average, Ensemble, LSTM, GRU
- **Interactive Charts:** Candlestick visualization with forecast overlay
- **Real-time Data:** Fetches live data from yfinance
- **Performance Metrics:** RMSE, MAE, MAPE
- **Model Caching:** Neural networks cached for faster predictions

---

## 📁 Project Structure

```
fintech-forecasting-app/
├── backend/           # Flask API + ML models
├── frontend/          # HTML/CSS/JS interface
├── tests/             # Unit tests
├── docs/              # Report and documentation
├── requirements.txt   # Dependencies
└── Dockerfile         # Container config
```

---

## 🎯 Models Implemented

### Traditional Models
- ARIMA (5,1,0)
- Moving Average
- Exponential Smoothing
- Ensemble

### Neural Models
- LSTM (2 layers, 64 units)
- GRU (2 layers, 64 units)

---

## 📚 Documentation

**Complete documentation and assignment report available in the `docs/` folder:**

- **docs/report.pdf** - Full assignment report with architecture, models, and evaluation
- **docs/README.md** - Comprehensive documentation with API details, troubleshooting, and more

---

## 🐳 Docker Deployment

```bash
docker build -t stock-forecasting .
docker run -p 5000:5000 stock-forecasting
```

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 📦 Tech Stack

- **Frontend:** HTML, CSS, JavaScript, Plotly.js
- **Backend:** Flask, Python
- **Database:** MongoDB
- **ML:** PyTorch (LSTM/GRU), statsmodels (ARIMA)
- **Data:** yfinance API

---

## 📄 License

MIT License - Educational project
