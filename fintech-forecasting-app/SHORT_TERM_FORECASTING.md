# Short-Term Forecasting Strategy

## 📊 Data Used

### Recent Data Window

- **Period**: Last **7 days** (rolling window)
- **Interval**: **Hourly** prices (1h)
- **Data Points**: ~168 hours (7 days × 24 hours)
- **OHLCV**: Open, High, Low, Close, Volume

### Why 7 Days?

✅ Captures **recent market behavior**
✅ Includes **short-term trends** and momentum
✅ Reflects **current volatility** levels
✅ Provides enough data without noise from old patterns

---

## 📈 Technical Indicators Calculated

### 1. **7-Day Simple Moving Average (SMA-7)**

- Average of last 7 closing prices
- Shows immediate trend direction

### 2. **RSI (Relative Strength Index) - 14 periods**

- Measures momentum
- **>70**: Overbought (potential reversal down)
- **<30**: Oversold (potential reversal up)
- **30-70**: Neutral

### 3. **Bollinger Bands (20 periods, 2σ)**

- **Upper Band**: Resistance level
- **Lower Band**: Support level
- Price touching bands = potential reversal

### 4. **Recent Volatility (24h)**

- Standard deviation of last 24 hours
- Higher volatility = wider price ranges

### 5. **Momentum (5-day %)**

- Price change over last 5 days
- Positive = uptrend, Negative = downtrend

### 6. **Support/Resistance Levels**

- **Support**: Recent 24h low
- **Resistance**: Recent 24h high

---

## 🎯 Forecast Horizons

### 1-Hour Forecast

- **Direction**: up/down/sideways
- **Confidence**: high/medium/low
- **Price Range**: ±2% from predicted

### 4-Hour Forecast

- **Direction**: up/down/sideways
- **Confidence**: high/medium/low
- **Price Range**: ±4% from predicted

### 24-Hour Forecast

- **Direction**: up/down/sideways
- **Confidence**: high/medium/low
- **Price Range**: ±7% from predicted

---

## 🧠 ML Model Training

### Input Data

```python
Recent 7 days hourly closes = [150.2, 151.3, 150.8, ..., 175.4]
# ~168 data points
```

### Training Process

1. **No historical data >7 days** is used
2. Model learns **recent patterns** only
3. Technical indicators provide **context**
4. Predictions are **hourly** for next 1-72 hours

### Models Available

- **Moving Average**: Fast, trend-following
- **ARIMA**: Statistical, auto-tuned
- **LSTM**: Pattern-based, recent sequences
- **GRU**: Momentum-weighted patterns
- **Transformer**: Attention on recent prices

---

## 📤 Response Format

```json
{
  "symbol": "AAPL",
  "model_type": "arima",
  "current_price": 175.5,

  "forecast_1h": {
    "direction": "up",
    "confidence": "high",
    "predicted_price": 176.2,
    "price_range": [172.68, 179.72]
  },

  "forecast_4h": {
    "direction": "up",
    "confidence": "medium",
    "predicted_price": 177.1,
    "price_range": [170.02, 184.18]
  },

  "forecast_24h": {
    "direction": "sideways",
    "confidence": "low",
    "predicted_price": 175.8,
    "price_range": [163.49, 188.11]
  },

  "technical_indicators": {
    "sma_7": 174.25,
    "rsi": 58.3,
    "bb_upper": 180.5,
    "bb_lower": 170.2,
    "volatility_pct": 2.15,
    "momentum_5d_pct": 1.8,
    "support_level": 172.1,
    "resistance_level": 178.9
  },

  "forecast": [
    {
      "timestamp": "2025-10-06T12:00:00Z",
      "predicted_price": 176.2,
      "price_range_low": 174.1,
      "price_range_high": 178.3,
      "confidence": "high"
    }
    // ... hourly predictions up to 72h
  ],

  "data_points_used": 168,
  "data_period": "7 days hourly"
}
```

---

## 🎨 Confidence Levels

### High Confidence

- **Low volatility** (<1%) + **Clear direction** (>2% change)
- **Short horizon** (1-4 hours)
- **Strong technical signals** (RSI extremes, BB touches)

### Medium Confidence

- **Medium volatility** (1-3%) + **Moderate direction**
- **Medium horizon** (4-12 hours)
- **Mixed technical signals**

### Low Confidence

- **High volatility** (>3%)
- **Long horizon** (>12 hours)
- **Sideways market** (no clear trend)

---

## 🔄 Rolling Window Strategy

### Database Storage (Future Implementation)

```
Keep only last 7-14 days of OHLCV data per stock
Delete data older than 14 days automatically
This keeps database lean and focused on recent trends
```

### Current Implementation

- Fetch fresh 7 days hourly data from yfinance on each request
- No stale data
- Always uses most recent market behavior

---

## ✅ Benefits

1. **Focused on Recent Trends** - No noise from old data
2. **Technical Indicators** - RSI, Bollinger Bands, Support/Resistance
3. **Price Ranges** - Not single-point predictions
4. **Confidence Levels** - Honest about uncertainty
5. **Multiple Horizons** - 1h, 4h, 24h forecasts
6. **Fast Training** - Only 168 data points to process

---

## 🚀 Usage

```bash
# Start the app
npm run app

# Visit stock page
http://localhost:3000/stock/AAPL

# Click any model button
# See short-term forecast with:
# - Direction (up/down/sideways)
# - Confidence (high/medium/low)
# - Price ranges
# - Technical indicators
```

---

## 📊 Example Output

**AAPL at $175.50**

**1h Forecast**: ↑ UP (High Confidence)

- Predicted: $176.20
- Range: $174.10 - $178.30
- RSI: 58 (Neutral), Momentum: +1.8%

**4h Forecast**: ↑ UP (Medium Confidence)

- Predicted: $177.10
- Range: $170.02 - $184.18
- Volatility: 2.15%

**24h Forecast**: → SIDEWAYS (Low Confidence)

- Predicted: $175.80
- Range: $163.49 - $188.11
- Note: High uncertainty over 24h
