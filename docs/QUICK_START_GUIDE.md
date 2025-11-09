# Quick Start Guide - Adaptive Learning System

**Student:** Dawood Hussain (22i-2410)  
**Course:** NLP Section A

---

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required:**
- Python 3.10+
- MongoDB

### Step 2: Start MongoDB

**Windows:**
```bash
mongod
```

**Mac:**
```bash
brew services start mongodb-community
```

**Linux:**
```bash
sudo systemctl start mongod
```

### Step 3: Run the System

**Automated (Recommended):**
```bash
python start_adaptive_system.py
```

This will:
- ✅ Check MongoDB connection
- ✅ Initialize the system (first time only)
- ✅ Start the Flask application
- ✅ Open monitoring dashboard

**Manual:**
```bash
# Initialize (first time only)
python backend/initialize_adaptive_system.py

# Start Flask
python backend/app.py
```

---

## 🌐 Access the Application

### Main Forecasting App
```
http://localhost:5000
```

**Features:**
- Generate forecasts for stocks/crypto/forex
- Compare multiple models
- Interactive candlestick charts
- Performance metrics

### Adaptive Learning Monitor
```
http://localhost:5000/monitor
```

**Features:**
- Real-time performance tracking
- Model version history
- Ensemble weight visualization
- Training logs
- Manual retrain triggers

---

## 📊 Using the Monitor Dashboard

### Overview Tab
- **Scheduler Status:** Check if adaptive learning is running
- **Model Statistics:** Total predictions, training count
- **Current Performance:** Recent vs baseline MAPE

### Performance Tab
- **Trend Chart:** 30-day MAPE history
- **Visual Degradation:** See performance changes over time

### Versions Tab
- **Version History:** All model versions with metrics
- **Active Version:** Currently deployed model
- **Rollback:** Can revert to previous versions

### Ensemble Tab
- **Current Weights:** Model contribution distribution
- **Weight Evolution:** How weights change over time
- **Rebalancing:** Automatic optimization

### Training Logs Tab
- **Event History:** All training events
- **Triggers:** Why retraining occurred
- **Duration:** Training time and epochs
- **Status:** Success/failure tracking

---

## 🎮 Interactive Controls

### Refresh Data
Click **🔄 Refresh Data** to update all metrics immediately.

### Trigger Retrain
1. Select symbol (e.g., AAPL)
2. Select model (LSTM or GRU)
3. Click **🔧 Trigger Retrain**
4. Wait 2-5 minutes for completion
5. Refresh to see new version

### Rebalance Ensemble
1. Select symbol
2. Click **⚖️ Rebalance Ensemble**
3. View updated weights in Ensemble tab

---

## 🔄 How Adaptive Learning Works

### Automatic Retraining

The system automatically retrains models when:

1. **Performance Degrades**
   - Recent MAPE > Baseline MAPE × 1.2
   - Example: Baseline 2%, Recent 2.5% → Retrain

2. **Consecutive Failures**
   - 3+ predictions with MAPE > 5%
   - Indicates model drift

3. **Scheduled Monthly**
   - 30+ days since last training
   - Routine maintenance

4. **Scheduled Weekly**
   - 7+ days + MAPE > 2.5%
   - Light fine-tuning

### Retraining Process

```
1. Detect Trigger
   ↓
2. Fetch Recent Data (365 days)
   ↓
3. Load Existing Model
   ↓
4. Transfer Learning
   - Freeze early layers
   - Lower learning rate (0.0001)
   - Train 10 epochs
   ↓
5. Evaluate Performance
   ↓
6. Save New Version (v1.0.1 → v1.0.2)
   ↓
7. Update Ensemble Weights
   ↓
8. Log Event
```

### Ensemble Rebalancing

Every 24 hours, the system:

1. Calculates recent error for each model (last 7 days)
2. Computes inverse-error weights
3. Applies minimum threshold (5%)
4. Normalizes to sum to 100%
5. Updates ensemble predictions

**Example:**
```
LSTM:  MAPE 1.5% → Weight 35%
GRU:   MAPE 1.6% → Weight 30%
ARIMA: MAPE 2.1% → Weight 20%
MA:    MAPE 3.2% → Weight 10%
```

---

## 📈 Monitoring Performance

### Key Metrics

**MAPE (Mean Absolute Percentage Error)**
- Primary metric for accuracy
- Lower is better
- < 2% = Excellent
- 2-5% = Good
- > 5% = Needs improvement

**RMSE (Root Mean Squared Error)**
- Absolute error in price units
- Penalizes large errors more

**MAE (Mean Absolute Error)**
- Average absolute error
- More robust to outliers

### Performance Status

- **✅ Good:** Recent MAPE ≤ Baseline × 1.2
- **⚠️ Degraded:** Recent MAPE > Baseline × 1.2
- **❌ Poor:** Recent MAPE > Baseline × 1.5

---

## 🛠️ Troubleshooting

### MongoDB Connection Error

**Problem:** "MongoDB is not running"

**Solution:**
```bash
# Check if MongoDB is running
mongod --version

# Start MongoDB
mongod
```

### No Data in Monitor

**Problem:** All metrics show "N/A"

**Solution:**
```bash
# Run initialization script
python backend/initialize_adaptive_system.py

# Or use the main forecasting app first
# Generate some predictions at http://localhost:5000
```

### Scheduler Not Running

**Problem:** Scheduler status shows "🔴 Stopped"

**Solution:**
The scheduler starts automatically when Flask starts. If stopped:
1. Restart the Flask application
2. Check logs for errors
3. Ensure MongoDB is running

### Slow Retraining

**Problem:** Retraining takes too long

**Explanation:**
- First training: 2-5 minutes (full training)
- Subsequent: 1-2 minutes (transfer learning)
- This is normal for neural networks

**Tips:**
- Use transfer learning (enabled by default)
- Reduce epochs (default: 10)
- Use GPU if available

---

## 💡 Tips & Best Practices

### For Best Results

1. **Let it Run:** Allow 24-48 hours for initial data collection
2. **Monitor Regularly:** Check dashboard daily
3. **Trust the System:** Automatic retraining is optimized
4. **Manual Override:** Use sparingly, only when needed

### Performance Optimization

1. **Ensemble Weights:** Check weekly, should be balanced
2. **Version History:** Keep eye on MAPE trends
3. **Training Logs:** Review for failures
4. **Activity Feed:** Monitor for anomalies

### Data Quality

1. **Symbol Selection:** Use liquid, actively traded symbols
2. **Data Freshness:** System fetches real-time data
3. **Historical Depth:** Needs 100+ days for training
4. **Market Hours:** Best predictions during trading hours

---

## 📚 Additional Resources

### Documentation
- `ADAPTIVE_LEARNING_APPROACH.md` - Detailed algorithms
- `IMPLEMENTATION_STATUS.md` - Technical details
- `MILESTONE2_COMPLETE.md` - Complete feature list

### Code Structure
```
backend/
├── adaptive_learning/      # Core adaptive learning modules
├── models/                 # LSTM, GRU, traditional models
├── app.py                  # Flask API with endpoints
├── database.py             # MongoDB operations
└── data_fetcher.py         # yfinance integration

frontend/
├── templates/
│   ├── index.html         # Main forecasting page
│   └── monitor.html       # Adaptive learning monitor
└── static/
    ├── app.js             # Main app logic
    └── monitor.js         # Monitor dashboard logic
```

### API Endpoints

**Adaptive Learning:**
- `GET /api/adaptive/scheduler/status`
- `POST /api/adaptive/retrain`
- `GET /api/adaptive/performance/<symbol>/<model>`
- `GET /api/adaptive/versions/<symbol>/<model>`
- `POST /api/adaptive/ensemble/rebalance`

---

## 🎯 What to Expect

### First Hour
- System initializes with sample data
- Models train on historical data
- Ensemble weights calculated
- Dashboard shows initial metrics

### First Day
- Scheduler runs at 2 AM
- Performance tracked continuously
- Ensemble rebalances automatically
- Activity feed shows events

### First Week
- Models adapt to recent market conditions
- Version history builds up
- Performance trends become visible
- Optimal ensemble weights emerge

### First Month
- Full adaptive learning cycle complete
- Multiple retraining events logged
- Clear performance improvements
- System fully optimized

---

## ✅ Success Checklist

Before considering the system operational:

- [ ] MongoDB running and accessible
- [ ] Flask application started successfully
- [ ] Monitor dashboard loads without errors
- [ ] At least one symbol initialized
- [ ] Performance metrics showing data
- [ ] Scheduler status shows "Running"
- [ ] Can trigger manual retrain
- [ ] Ensemble weights displayed
- [ ] Training logs visible

---

## 🆘 Getting Help

### Common Issues

1. **Import Errors:** Run `pip install -r requirements.txt`
2. **MongoDB Errors:** Ensure MongoDB is running
3. **Port Conflicts:** Change port in `backend/app.py`
4. **Memory Issues:** Reduce batch size in neural models

### Debug Mode

Enable detailed logging:
```python
# In backend/app.py
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Check Logs

Monitor console output for:
- Training events
- Rebalancing operations
- Error messages
- Performance metrics

---

## 🎓 Learning Objectives

By using this system, you'll understand:

1. **Adaptive Learning:** How models evolve with new data
2. **Transfer Learning:** Fast fine-tuning vs full retraining
3. **Ensemble Methods:** Dynamic model combination
4. **Performance Monitoring:** Real-time accuracy tracking
5. **Version Control:** Model lifecycle management

---

**Ready to start?** Run `python start_adaptive_system.py` and visit http://localhost:5000/monitor! 🚀
