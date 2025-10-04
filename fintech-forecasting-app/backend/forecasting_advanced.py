"""
Advanced Forecasting Models for Financial Time Series
Implements ARIMA, LSTM, GRU, Transformer, and other models with proper evaluation metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, MultiHeadAttention, LayerNormalization
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

class ModelMetrics:
    """Calculate and store evaluation metrics for forecasting models"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate comprehensive forecast evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Directional accuracy (for financial forecasting)
        direction_true = np.diff(y_true) >= 0
        direction_pred = np.diff(y_pred) >= 0
        direction_accuracy = np.mean(direction_true == direction_pred) * 100
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy)
        }

class AdvancedARIMA:
    """Enhanced ARIMA model with automatic parameter selection"""
    
    def __init__(self, max_p=5, max_d=2, max_q=5):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.model = None
        self.fitted_model = None
        self.best_params = None
        
    def auto_arima(self, data):
        """Automatic ARIMA parameter selection using AIC"""
        best_aic = float('inf')
        best_params = None
        
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_params = (p, d, q)
                    except:
                        continue
        
        return best_params
    
    def fit(self, data):
        """Fit ARIMA model with automatic parameter selection"""
        self.best_params = self.auto_arima(data)
        if self.best_params:
            self.model = ARIMA(data, order=self.best_params)
            self.fitted_model = self.model.fit()
        return self
    
    def predict(self, steps=1):
        """Generate forecasts"""
        if self.fitted_model:
            forecast = self.fitted_model.forecast(steps=steps)
            return forecast
        return np.zeros(steps)

class MovingAverageModel:
    """Enhanced Moving Average with multiple techniques"""
    
    def __init__(self, window_sizes=[5, 10, 20], use_exponential=True):
        self.window_sizes = window_sizes
        self.use_exponential = use_exponential
        self.data = None
        self.exp_models = {}
        
    def fit(self, data):
        """Fit moving average models"""
        self.data = data
        
        if self.use_exponential:
            # Fit exponential smoothing models
            for window in self.window_sizes:
                try:
                    model = ExponentialSmoothing(
                        data, 
                        trend='add', 
                        seasonal=None,
                        seasonal_periods=window
                    )
                    self.exp_models[window] = model.fit()
                except:
                    # Fallback to simple exponential smoothing
                    model = ExponentialSmoothing(data, trend=None, seasonal=None)
                    self.exp_models[window] = model.fit()
        
        return self
    
    def predict(self, steps=1):
        """Generate ensemble prediction from multiple moving averages"""
        predictions = []
        
        # Simple moving averages
        for window in self.window_sizes:
            if len(self.data) >= window:
                ma_pred = np.mean(self.data[-window:])
                predictions.extend([ma_pred] * steps)
        
        # Exponential smoothing predictions
        if self.use_exponential:
            for window in self.window_sizes:
                if window in self.exp_models:
                    exp_pred = self.exp_models[window].forecast(steps)
                    predictions.extend(exp_pred)
        
        # Ensemble average
        if predictions:
            prediction_array = np.array(predictions).reshape(-1, steps)
            return np.mean(prediction_array, axis=0)
        
        return np.full(steps, self.data[-1])  # Fallback to last value

class LSTMModel:
    """Advanced LSTM model for time series forecasting"""
    
    def __init__(self, sequence_length=60, units=50, layers=2, dropout=0.2):
        self.sequence_length = sequence_length
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def prepare_data(self, data):
        """Prepare data for LSTM training"""
        # Scale data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.units,
            return_sequences=True if self.layers > 1 else False,
            input_shape=input_shape
        ))
        model.add(Dropout(self.dropout))
        
        # Additional LSTM layers
        for i in range(1, self.layers):
            model.add(LSTM(
                units=self.units,
                return_sequences=True if i < self.layers - 1 else False
            ))
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def fit(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """Train LSTM model"""
        X, y = self.prepare_data(data)
        
        if len(X) == 0:
            return self
        
        # Reshape for LSTM input
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build and train model
        self.model = self.build_model((X.shape[1], 1))
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            shuffle=False
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, data, steps=1):
        """Generate LSTM predictions"""
        if not self.is_fitted or self.model is None:
            return np.zeros(steps)
        
        # Use last sequence_length points for prediction
        if len(data) < self.sequence_length:
            return np.full(steps, data[-1])
        
        # Scale the input data
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        
        predictions = []
        current_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        for _ in range(steps):
            pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence for next prediction
            new_sequence = np.append(current_sequence[0, 1:, 0], pred[0, 0])
            current_sequence = new_sequence.reshape(1, self.sequence_length, 1)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()

class GRUModel:
    """GRU model similar to LSTM but with simplified architecture"""
    
    def __init__(self, sequence_length=60, units=50, layers=2, dropout=0.2):
        self.sequence_length = sequence_length
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def prepare_data(self, data):
        """Prepare data for GRU training"""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build GRU model architecture"""
        model = Sequential()
        
        # First GRU layer
        model.add(GRU(
            units=self.units,
            return_sequences=True if self.layers > 1 else False,
            input_shape=input_shape
        ))
        model.add(Dropout(self.dropout))
        
        # Additional GRU layers
        for i in range(1, self.layers):
            model.add(GRU(
                units=self.units,
                return_sequences=True if i < self.layers - 1 else False
            ))
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def fit(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """Train GRU model"""
        X, y = self.prepare_data(data)
        
        if len(X) == 0:
            return self
        
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        self.model = self.build_model((X.shape[1], 1))
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            shuffle=False
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, data, steps=1):
        """Generate GRU predictions"""
        if not self.is_fitted or self.model is None:
            return np.zeros(steps)
        
        if len(data) < self.sequence_length:
            return np.full(steps, data[-1])
        
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        
        predictions = []
        current_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        for _ in range(steps):
            pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            new_sequence = np.append(current_sequence[0, 1:, 0], pred[0, 0])
            current_sequence = new_sequence.reshape(1, self.sequence_length, 1)
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()

class SimpleTransformerModel:
    """Simplified Transformer model for time series forecasting"""
    
    def __init__(self, sequence_length=60, d_model=64, num_heads=4, dropout=0.1):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def prepare_data(self, data):
        """Prepare data for Transformer training"""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build simplified Transformer model"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # Project to d_model dimensions
        x = Dense(self.d_model)(inputs)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout
        )(x, x)
        
        # Add & Norm
        x = LayerNormalization()(x + attention_output)
        
        # Feed Forward
        ffn_output = Dense(self.d_model * 2, activation='relu')(x)
        ffn_output = Dropout(self.dropout)(ffn_output)
        ffn_output = Dense(self.d_model)(ffn_output)
        
        # Add & Norm
        x = LayerNormalization()(x + ffn_output)
        
        # Global pooling and output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = Dense(1)(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def fit(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """Train Transformer model"""
        X, y = self.prepare_data(data)
        
        if len(X) == 0:
            return self
        
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        self.model = self.build_model((X.shape[1], 1))
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            shuffle=False
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, data, steps=1):
        """Generate Transformer predictions"""
        if not self.is_fitted or self.model is None:
            return np.zeros(steps)
        
        if len(data) < self.sequence_length:
            return np.full(steps, data[-1])
        
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        
        predictions = []
        current_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        for _ in range(steps):
            pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            new_sequence = np.append(current_sequence[0, 1:, 0], pred[0, 0])
            current_sequence = new_sequence.reshape(1, self.sequence_length, 1)
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()

class ModelEnsemble:
    """Ensemble multiple models for improved accuracy"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        
    def add_model(self, name, model, weight=1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
    def fit(self, data, **kwargs):
        """Fit all models in the ensemble"""
        for name, model in self.models.items():
            try:
                model.fit(data, **kwargs)
            except Exception as e:
                print(f"Error fitting {name}: {e}")
        return self
    
    def predict(self, data, steps=1):
        """Generate ensemble prediction"""
        predictions = []
        total_weight = 0
        
        for name, model in self.models.items():
            try:
                pred = model.predict(data, steps)
                if pred is not None and len(pred) == steps:
                    weight = self.weights[name]
                    predictions.append(pred * weight)
                    total_weight += weight
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
        
        if predictions and total_weight > 0:
            ensemble_pred = np.sum(predictions, axis=0) / total_weight
            return ensemble_pred
        
        # Fallback
        return np.full(steps, data[-1] if len(data) > 0 else 0)

# Factory function for creating models
def create_model(model_type, **kwargs):
    """Factory function to create different model types"""
    
    if model_type.lower() == 'arima':
        return AdvancedARIMA(**kwargs)
    elif model_type.lower() == 'moving_average':
        return MovingAverageModel(**kwargs)
    elif model_type.lower() == 'lstm':
        return LSTMModel(**kwargs)
    elif model_type.lower() == 'gru':
        return GRUModel(**kwargs)
    elif model_type.lower() == 'transformer':
        return SimpleTransformerModel(**kwargs)
    elif model_type.lower() == 'ensemble':
        ensemble = ModelEnsemble()
        # Add default models to ensemble
        ensemble.add_model('arima', AdvancedARIMA(), weight=0.3)
        ensemble.add_model('lstm', LSTMModel(), weight=0.4)
        ensemble.add_model('gru', GRUModel(), weight=0.3)
        return ensemble
    else:
        raise ValueError(f"Unknown model type: {model_type}")