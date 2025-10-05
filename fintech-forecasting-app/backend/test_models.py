"""
Unit tests for the FinTech Forecasting application.
Tests ML models, API endpoints, and database operations.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import application modules
from config import Config
from database import PriceData, ForecastData, ModelPerformance
from forecasting import (
    MovingAverageModel, ARIMAModel, LSTMModel, 
    create_model, ForecastModel
)

class TestPriceData(unittest.TestCase):
    """Test PriceData model."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = PriceData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open_price=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000
        )
    
    def test_price_data_creation(self):
        """Test PriceData object creation."""
        self.assertEqual(self.sample_data.symbol, "AAPL")
        self.assertEqual(self.sample_data.close, 154.0)
        self.assertIsInstance(self.sample_data.timestamp, datetime)
    
    def test_price_data_to_dict(self):
        """Test PriceData serialization."""
        data_dict = self.sample_data.to_dict()
        
        self.assertIn("symbol", data_dict)
        self.assertIn("close", data_dict)
        self.assertIn("timestamp", data_dict)
        self.assertIn("created_at", data_dict)
        
        self.assertEqual(data_dict["symbol"], "AAPL")
        self.assertEqual(data_dict["close"], 154.0)
    
    def test_price_data_from_dict(self):
        """Test PriceData deserialization."""
        data_dict = self.sample_data.to_dict()
        reconstructed = PriceData.from_dict(data_dict)
        
        self.assertEqual(reconstructed.symbol, self.sample_data.symbol)
        self.assertEqual(reconstructed.close, self.sample_data.close)

class TestForecastData(unittest.TestCase):
    """Test ForecastData model."""
    
    def setUp(self):
        """Set up test forecast data."""
        self.sample_forecast = ForecastData(
            symbol="AAPL",
            model_type="LSTM",
            forecast_horizon=24,
            predictions=[
                {"timestamp": "2023-10-01T00:00:00Z", "value": 155.0},
                {"timestamp": "2023-10-01T01:00:00Z", "value": 156.0}
            ],
            metrics={"mse": 0.5, "mae": 0.3}
        )
    
    def test_forecast_data_creation(self):
        """Test ForecastData object creation."""
        self.assertEqual(self.sample_forecast.symbol, "AAPL")
        self.assertEqual(self.sample_forecast.model_type, "LSTM")
        self.assertEqual(self.sample_forecast.forecast_horizon, 24)
        self.assertEqual(len(self.sample_forecast.predictions), 2)
    
    def test_forecast_data_serialization(self):
        """Test ForecastData serialization."""
        data_dict = self.sample_forecast.to_dict()
        
        self.assertIn("symbol", data_dict)
        self.assertIn("model_type", data_dict)
        self.assertIn("predictions", data_dict)
        self.assertIn("metrics", data_dict)

class TestMovingAverageModel(unittest.TestCase):
    """Test Moving Average forecasting model."""
    
    def setUp(self):
        """Set up test data for moving average model."""
        self.model = MovingAverageModel("AAPL", window=10)
        
        # Create sample price data
        base_time = datetime.now(timezone.utc)
        self.price_data = []
        
        for i in range(50):
            price = 100 + np.sin(i * 0.1) * 10 + np.random.normal(0, 1)
            self.price_data.append(PriceData(
                symbol="AAPL",
                timestamp=base_time + timedelta(days=i),
                open_price=price - 1,
                high=price + 2,
                low=price - 2,
                close=price,
                volume=1000000
            ))
    
    def test_model_creation(self):
        """Test model creation."""
        self.assertEqual(self.model.symbol, "AAPL")
        self.assertEqual(self.model.window, 10)
        self.assertEqual(self.model.model_type, "MovingAverage")
    
    def test_prepare_data(self):
        """Test data preparation."""
        prices, _ = self.model.prepare_data(self.price_data)
        
        self.assertEqual(len(prices), len(self.price_data))
        self.assertIsInstance(prices, np.ndarray)
    
    def test_training(self):
        """Test model training."""
        metrics = self.model.train(self.price_data)
        
        self.assertTrue(self.model.is_trained)
        self.assertIn("training_samples", metrics)
        self.assertEqual(metrics["training_samples"], len(self.price_data))
    
    def test_prediction(self):
        """Test model prediction."""
        self.model.train(self.price_data)
        predictions, confidence_intervals = self.model.predict(self.price_data, 5)
        
        self.assertEqual(len(predictions), 5)
        self.assertIsInstance(predictions, np.ndarray)
        # Moving average doesn't provide confidence intervals
        self.assertIsNone(confidence_intervals)
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        actual = np.array([100, 101, 102, 103, 104])
        predicted = np.array([99, 100, 101, 104, 105])
        
        metrics = self.model.evaluate(actual, predicted)
        
        self.assertIn("mse", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("mape", metrics)
        self.assertIn("directional_accuracy", metrics)
        
        self.assertGreater(metrics["mse"], 0)
        self.assertGreater(metrics["rmse"], 0)
        self.assertGreater(metrics["mae"], 0)

class TestARIMAModel(unittest.TestCase):
    """Test ARIMA forecasting model."""
    
    def setUp(self):
        """Set up test data for ARIMA model."""
        self.model = ARIMAModel("AAPL")
        
        # Create more realistic time series data
        base_time = datetime.now(timezone.utc)
        self.price_data = []
        
        for i in range(100):
            # Create trending data with some seasonality
            trend = 100 + i * 0.1
            seasonal = np.sin(i * 0.1) * 5
            noise = np.random.normal(0, 2)
            price = trend + seasonal + noise
            
            self.price_data.append(PriceData(
                symbol="AAPL",
                timestamp=base_time + timedelta(days=i),
                open_price=price - 1,
                high=price + 2,
                low=price - 2,
                close=price,
                volume=1000000
            ))
    
    def test_model_creation(self):
        """Test ARIMA model creation."""
        self.assertEqual(self.model.symbol, "AAPL")
        self.assertEqual(self.model.model_type, "ARIMA")
        self.assertIsNone(self.model.order)  # Auto-determine
    
    def test_data_preparation(self):
        """Test ARIMA data preparation."""
        ts_values, _ = self.model.prepare_data(self.price_data)
        
        self.assertEqual(len(ts_values), len(self.price_data))
        self.assertIsInstance(ts_values, np.ndarray)
    
    @patch('pmdarima.auto_arima')
    def test_training_with_mock(self, mock_auto_arima):
        """Test ARIMA training with mocked pmdarima."""
        # Mock the ARIMA model
        mock_model = MagicMock()
        mock_model.aic.return_value = 100.0
        mock_model.bic.return_value = 105.0
        mock_model.fittedvalues.return_value = np.random.normal(100, 5, 99)
        mock_auto_arima.return_value = mock_model
        
        metrics = self.model.train(self.price_data)
        
        self.assertTrue(self.model.is_trained)
        self.assertIn("training_samples", metrics)
        self.assertIn("aic", metrics)
        self.assertIn("bic", metrics)
        
        mock_auto_arima.assert_called_once()

class TestLSTMModel(unittest.TestCase):
    """Test LSTM forecasting model."""
    
    def setUp(self):
        """Set up test data for LSTM model."""
        self.model = LSTMModel("AAPL", sequence_length=10, lstm_units=32)
        
        # Create sample data (more than sequence_length)
        base_time = datetime.now(timezone.utc)
        self.price_data = []
        
        for i in range(50):
            price = 100 + np.sin(i * 0.1) * 10 + np.random.normal(0, 1)
            self.price_data.append(PriceData(
                symbol="AAPL",
                timestamp=base_time + timedelta(days=i),
                open_price=price - 1,
                high=price + 2,
                low=price - 2,
                close=price,
                volume=1000000 + np.random.randint(-100000, 100000)
            ))
    
    def test_model_creation(self):
        """Test LSTM model creation."""
        self.assertEqual(self.model.symbol, "AAPL")
        self.assertEqual(self.model.sequence_length, 10)
        self.assertEqual(self.model.lstm_units, 32)
        self.assertEqual(self.model.model_type, "LSTM")
    
    @patch('sklearn.preprocessing.MinMaxScaler')
    def test_data_preparation(self, mock_scaler):
        """Test LSTM data preparation."""
        # Mock the scaler
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.random((50, 5))
        mock_scaler.return_value = mock_scaler_instance
        
        X, y = self.model.prepare_data(self.price_data)
        
        # Should create sequences
        expected_sequences = len(self.price_data) - self.model.sequence_length
        self.assertEqual(len(X), expected_sequences)
        self.assertEqual(len(y), expected_sequences)
        self.assertEqual(X.shape[1], self.model.sequence_length)
        self.assertEqual(X.shape[2], 5)  # OHLCV features

class TestModelFactory(unittest.TestCase):
    """Test model factory function."""
    
    def test_create_moving_average_model(self):
        """Test creating moving average model."""
        model = create_model("moving_average", "AAPL")
        self.assertIsInstance(model, MovingAverageModel)
        self.assertEqual(model.symbol, "AAPL")
    
    def test_create_arima_model(self):
        """Test creating ARIMA model."""
        model = create_model("arima", "AAPL")
        self.assertIsInstance(model, ARIMAModel)
        self.assertEqual(model.symbol, "AAPL")
    
    def test_create_lstm_model(self):
        """Test creating LSTM model."""
        model = create_model("lstm", "AAPL")
        self.assertIsInstance(model, LSTMModel)
        self.assertEqual(model.symbol, "AAPL")
    
    def test_create_invalid_model(self):
        """Test creating invalid model type."""
        with self.assertRaises(ValueError):
            create_model("invalid_model", "AAPL")

class TestConfig(unittest.TestCase):
    """Test configuration module."""
    
    def test_config_attributes(self):
        """Test that required config attributes exist."""
        self.assertTrue(hasattr(Config, 'DATABASE_URL'))
        self.assertTrue(hasattr(Config, 'FINNHUB_API_KEY'))
        self.assertTrue(hasattr(Config, 'FLASK_PORT'))
        self.assertTrue(hasattr(Config, 'MODEL_UPDATE_INTERVAL'))
    
    def test_config_validation(self):
        """Test configuration validation."""
        # This will return False with demo API key
        is_valid = Config.validate_config()
        self.assertIsInstance(is_valid, bool)
    
    def test_directory_creation(self):
        """Test that Config creates necessary directories."""
        # The directories should be created on import
        self.assertTrue(os.path.exists(Config.MODELS_DIR))

class TestAPIEndpoints(unittest.TestCase):
    """Test Flask API endpoints."""
    
    def setUp(self):
        """Set up test Flask app."""
        from main import app
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
    
    def test_models_endpoint(self):
        """Test models listing endpoint."""
        response = self.app.get('/api/models')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertIn('models', data)
        self.assertIsInstance(data['models'], list)
        
        # Check that expected models are present
        model_types = [model['type'] for model in data['models']]
        self.assertIn('moving_average', model_types)
        self.assertIn('arima', model_types)
        self.assertIn('lstm', model_types)
    
    def test_not_found_endpoint(self):
        """Test 404 handling."""
        response = self.app.get('/api/nonexistent')
        self.assertEqual(response.status_code, 404)
        
        data = response.get_json()
        self.assertIn('error', data)

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestPriceData,
        TestForecastData,
        TestMovingAverageModel,
        TestARIMAModel,
        TestLSTMModel,
        TestModelFactory,
        TestConfig,
        TestAPIEndpoints
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
