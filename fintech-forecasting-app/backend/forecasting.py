"""
Forecasting module containing all ML models for financial time series prediction.
Implements traditional models (ARIMA, Moving Average) and neural models (LSTM, GRU, Transformer).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import joblib
import warnings
warnings.filterwarnings('ignore')

# Traditional ML imports
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Input, 
    MultiHeadAttention, LayerNormalization, 
    GlobalAveragePooling1D, Embedding
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from database import PriceData, ForecastData, ModelPerformance
from config import Config

logger = logging.getLogger(__name__)

class ForecastModel(ABC):
    """Abstract base class for all forecasting models."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model = None
        self.is_trained = False
        self.model_type = self.__class__.__name__
    
    @abstractmethod
    def prepare_data(self, data: List[PriceData]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training/prediction."""
        pass
    
    @abstractmethod
    def train(self, data: List[PriceData]) -> Dict[str, float]:
        """Train the model and return training metrics."""
        pass
    
    @abstractmethod
    def predict(self, data: List[PriceData], horizon: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions. Returns (predictions, confidence_intervals)."""
        pass
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Calculate directional accuracy
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "directional_accuracy": float(directional_accuracy)
        }

class MovingAverageModel(ForecastModel):
    """Simple Moving Average forecasting model."""
    
    def __init__(self, symbol: str, window: int = 20):
        super().__init__(symbol)
        self.window = window
        self.model_type = "MovingAverage"
    
    def prepare_data(self, data: List[PriceData]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for moving average calculation."""
        prices = np.array([d.close for d in sorted(data, key=lambda x: x.timestamp)])
        return prices, prices  # No separate features needed
    
    def train(self, data: List[PriceData]) -> Dict[str, float]:
        """Moving average doesn't require training."""
        self.is_trained = True
        return {"training_samples": len(data)}
    
    def predict(self, data: List[PriceData], horizon: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict using simple moving average."""
        prices, _ = self.prepare_data(data)
        
        if len(prices) < self.window:
            # Use available data
            last_value = prices[-1] if len(prices) > 0 else 0
            return np.full(horizon, last_value), None
        
        # Calculate moving average
        ma = np.mean(prices[-self.window:])
        
        # Simple forecast: extend the moving average
        predictions = np.full(horizon, ma)
        
        # Add some trend if available
        if len(prices) >= self.window * 2:
            recent_ma = np.mean(prices[-self.window:])
            older_ma = np.mean(prices[-2*self.window:-self.window])
            trend = (recent_ma - older_ma) / self.window
            
            for i in range(horizon):
                predictions[i] = ma + trend * (i + 1)
        
        return predictions, None

class ARIMAModel(ForecastModel):
    """ARIMA (AutoRegressive Integrated Moving Average) forecasting model."""
    
    def __init__(self, symbol: str, order: Optional[Tuple[int, int, int]] = None):
        super().__init__(symbol)
        self.order = order  # If None, will auto-determine
        self.model_type = "ARIMA"
    
    def prepare_data(self, data: List[PriceData]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for ARIMA."""
        sorted_data = sorted(data, key=lambda x: x.timestamp)
        prices = np.array([d.close for d in sorted_data])
        
        # Create time index
        timestamps = pd.DatetimeIndex([d.timestamp for d in sorted_data])
        ts = pd.Series(prices, index=timestamps)
        
        return ts.values, ts.values
    
    def train(self, data: List[PriceData]) -> Dict[str, float]:
        """Train ARIMA model."""
        try:
            sorted_data = sorted(data, key=lambda x: x.timestamp)
            prices = [d.close for d in sorted_data]
            ts = pd.Series(prices)
            
            if self.order is None:
                # Auto-determine ARIMA parameters
                self.model = pm.auto_arima(
                    ts,
                    start_p=0, start_q=0, max_p=3, max_q=3,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore'
                )
            else:
                # Use specified order
                arima = ARIMA(ts, order=self.order)
                self.model = arima.fit()
            
            self.is_trained = True
            
            # Calculate training metrics
            fitted_values = self.model.fittedvalues()
            training_mse = mean_squared_error(ts[1:], fitted_values)  # Skip first value
            
            return {
                "training_samples": len(data),
                "training_mse": float(training_mse),
                "aic": float(self.model.aic()),
                "bic": float(self.model.bic())
            }
            
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            raise
    
    def predict(self, data: List[PriceData], horizon: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make ARIMA predictions with confidence intervals."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Get forecast with confidence intervals
            forecast_result = self.model.predict(n_periods=horizon, return_conf_int=True)
            
            if isinstance(forecast_result, tuple):
                predictions, conf_int = forecast_result
            else:
                predictions = forecast_result
                conf_int = None
            
            confidence_intervals = None
            if conf_int is not None:
                confidence_intervals = conf_int
            
            return np.array(predictions), confidence_intervals
            
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            raise

class LSTMModel(ForecastModel):
    """LSTM (Long Short-Term Memory) neural network forecasting model."""
    
    def __init__(self, symbol: str, sequence_length: int = 60, 
                 lstm_units: int = 128, dropout_rate: float = 0.2):
        super().__init__(symbol)
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.scaler = None
        self.model_type = "LSTM"
    
    def prepare_data(self, data: List[PriceData]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training."""
        from sklearn.preprocessing import MinMaxScaler
        
        sorted_data = sorted(data, key=lambda x: x.timestamp)
        
        # Use multiple features: OHLCV
        features = np.array([
            [d.open_price, d.high, d.low, d.close, d.volume] 
            for d in sorted_data
        ])
        
        # Scale the data
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            scaled_features = self.scaler.fit_transform(features)
        else:
            scaled_features = self.scaler.transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(scaled_features[i, 3])  # Close price index
        
        return np.array(X), np.array(y)
    
    def train(self, data: List[PriceData]) -> Dict[str, float]:
        """Train LSTM model."""
        try:
            X, y = self.prepare_data(data)
            
            if len(X) < 10:  # Need minimum data
                raise ValueError("Insufficient data for LSTM training")
            
            # Split data for validation
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build LSTM model
            self.model = Sequential([
                LSTM(self.lstm_units, return_sequences=True, 
                     input_shape=(self.sequence_length, X.shape[2])),
                Dropout(self.dropout_rate),
                LSTM(self.lstm_units // 2, return_sequences=False),
                Dropout(self.dropout_rate),
                Dense(25),
                Dense(1)
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Train with early stopping
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            history = self.model.fit(
                X_train, y_train,
                epochs=Config.EPOCHS,
                batch_size=Config.BATCH_SIZE,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_trained = True
            
            # Calculate training metrics
            train_pred = self.model.predict(X_train, verbose=0)
            val_pred = self.model.predict(X_val, verbose=0)
            
            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            
            return {
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
                "final_train_loss": float(history.history['loss'][-1]),
                "final_val_loss": float(history.history['val_loss'][-1])
            }
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            raise
    
    def predict(self, data: List[PriceData], horizon: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make LSTM predictions."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Prepare the last sequence for prediction
            sorted_data = sorted(data, key=lambda x: x.timestamp)
            features = np.array([
                [d.open_price, d.high, d.low, d.close, d.volume] 
                for d in sorted_data[-self.sequence_length:]
            ])
            
            if len(features) < self.sequence_length:
                raise ValueError("Insufficient data for prediction")
            
            scaled_features = self.scaler.transform(features)
            
            predictions = []
            current_sequence = scaled_features.copy()
            
            for _ in range(horizon):
                # Predict next value
                X_pred = current_sequence[-self.sequence_length:].reshape(1, self.sequence_length, -1)
                pred = self.model.predict(X_pred, verbose=0)[0, 0]
                predictions.append(pred)
                
                # Update sequence for next prediction
                # For simplicity, repeat the last features with new predicted close price
                new_row = current_sequence[-1].copy()
                new_row[3] = pred  # Update close price
                current_sequence = np.vstack([current_sequence, new_row])
            
            # Inverse transform predictions (only close price)
            dummy_features = np.zeros((len(predictions), 5))
            dummy_features[:, 3] = predictions
            inverse_predictions = self.scaler.inverse_transform(dummy_features)[:, 3]
            
            return inverse_predictions, None
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            raise

class GRUModel(ForecastModel):
    """GRU (Gated Recurrent Unit) neural network forecasting model."""
    
    def __init__(self, symbol: str, sequence_length: int = 60, 
                 gru_units: int = 128, dropout_rate: float = 0.2):
        super().__init__(symbol)
        self.sequence_length = sequence_length
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.scaler = None
        self.model_type = "GRU"
    
    def prepare_data(self, data: List[PriceData]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for GRU training (similar to LSTM)."""
        from sklearn.preprocessing import MinMaxScaler
        
        sorted_data = sorted(data, key=lambda x: x.timestamp)
        
        features = np.array([
            [d.open_price, d.high, d.low, d.close, d.volume] 
            for d in sorted_data
        ])
        
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            scaled_features = self.scaler.fit_transform(features)
        else:
            scaled_features = self.scaler.transform(features)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(scaled_features[i, 3])  # Close price
        
        return np.array(X), np.array(y)
    
    def train(self, data: List[PriceData]) -> Dict[str, float]:
        """Train GRU model."""
        try:
            X, y = self.prepare_data(data)
            
            if len(X) < 10:
                raise ValueError("Insufficient data for GRU training")
            
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build GRU model
            self.model = Sequential([
                GRU(self.gru_units, return_sequences=True, 
                    input_shape=(self.sequence_length, X.shape[2])),
                Dropout(self.dropout_rate),
                GRU(self.gru_units // 2, return_sequences=False),
                Dropout(self.dropout_rate),
                Dense(25),
                Dense(1)
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            history = self.model.fit(
                X_train, y_train,
                epochs=Config.EPOCHS,
                batch_size=Config.BATCH_SIZE,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_trained = True
            
            train_pred = self.model.predict(X_train, verbose=0)
            val_pred = self.model.predict(X_val, verbose=0)
            
            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            
            return {
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
                "final_train_loss": float(history.history['loss'][-1]),
                "final_val_loss": float(history.history['val_loss'][-1])
            }
            
        except Exception as e:
            logger.error(f"GRU training failed: {e}")
            raise
    
    def predict(self, data: List[PriceData], horizon: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make GRU predictions (similar to LSTM)."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        try:
            sorted_data = sorted(data, key=lambda x: x.timestamp)
            features = np.array([
                [d.open_price, d.high, d.low, d.close, d.volume] 
                for d in sorted_data[-self.sequence_length:]
            ])
            
            if len(features) < self.sequence_length:
                raise ValueError("Insufficient data for prediction")
            
            scaled_features = self.scaler.transform(features)
            
            predictions = []
            current_sequence = scaled_features.copy()
            
            for _ in range(horizon):
                X_pred = current_sequence[-self.sequence_length:].reshape(1, self.sequence_length, -1)
                pred = self.model.predict(X_pred, verbose=0)[0, 0]
                predictions.append(pred)
                
                new_row = current_sequence[-1].copy()
                new_row[3] = pred
                current_sequence = np.vstack([current_sequence, new_row])
            
            dummy_features = np.zeros((len(predictions), 5))
            dummy_features[:, 3] = predictions
            inverse_predictions = self.scaler.inverse_transform(dummy_features)[:, 3]
            
            return inverse_predictions, None
            
        except Exception as e:
            logger.error(f"GRU prediction failed: {e}")
            raise

class TransformerModel(ForecastModel):
    """Transformer-based forecasting model."""
    
    def __init__(self, symbol: str, sequence_length: int = 60,
                 d_model: int = 128, num_heads: int = 8, dropout_rate: float = 0.1):
        super().__init__(symbol)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.scaler = None
        self.model_type = "Transformer"
    
    def prepare_data(self, data: List[PriceData]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for Transformer training."""
        from sklearn.preprocessing import MinMaxScaler
        
        sorted_data = sorted(data, key=lambda x: x.timestamp)
        
        features = np.array([
            [d.open_price, d.high, d.low, d.close, d.volume] 
            for d in sorted_data
        ])
        
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            scaled_features = self.scaler.fit_transform(features)
        else:
            scaled_features = self.scaler.transform(features)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(scaled_features[i, 3])  # Close price
        
        return np.array(X), np.array(y)
    
    def train(self, data: List[PriceData]) -> Dict[str, float]:
        """Train Transformer model."""
        try:
            X, y = self.prepare_data(data)
            
            if len(X) < 10:
                raise ValueError("Insufficient data for Transformer training")
            
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build Transformer model
            inputs = Input(shape=(self.sequence_length, X.shape[2]))
            
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=self.num_heads, 
                key_dim=self.d_model // self.num_heads
            )(inputs, inputs)
            
            # Add & Norm
            attention_output = LayerNormalization()(inputs + attention_output)
            
            # Feed forward
            ffn_output = Dense(self.d_model, activation='relu')(attention_output)
            ffn_output = Dropout(self.dropout_rate)(ffn_output)
            ffn_output = Dense(X.shape[2])(ffn_output)
            
            # Add & Norm
            ffn_output = LayerNormalization()(attention_output + ffn_output)
            
            # Global pooling and output
            pooled = GlobalAveragePooling1D()(ffn_output)
            outputs = Dense(1)(pooled)
            
            self.model = Model(inputs=inputs, outputs=outputs)
            
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(patience=7, factor=0.5)
            ]
            
            history = self.model.fit(
                X_train, y_train,
                epochs=Config.EPOCHS,
                batch_size=Config.BATCH_SIZE,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_trained = True
            
            train_pred = self.model.predict(X_train, verbose=0)
            val_pred = self.model.predict(X_val, verbose=0)
            
            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            
            return {
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
                "final_train_loss": float(history.history['loss'][-1]),
                "final_val_loss": float(history.history['val_loss'][-1])
            }
            
        except Exception as e:
            logger.error(f"Transformer training failed: {e}")
            raise
    
    def predict(self, data: List[PriceData], horizon: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make Transformer predictions."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        try:
            sorted_data = sorted(data, key=lambda x: x.timestamp)
            features = np.array([
                [d.open_price, d.high, d.low, d.close, d.volume] 
                for d in sorted_data[-self.sequence_length:]
            ])
            
            if len(features) < self.sequence_length:
                raise ValueError("Insufficient data for prediction")
            
            scaled_features = self.scaler.transform(features)
            
            predictions = []
            current_sequence = scaled_features.copy()
            
            for _ in range(horizon):
                X_pred = current_sequence[-self.sequence_length:].reshape(1, self.sequence_length, -1)
                pred = self.model.predict(X_pred, verbose=0)[0, 0]
                predictions.append(pred)
                
                new_row = current_sequence[-1].copy()
                new_row[3] = pred
                current_sequence = np.vstack([current_sequence, new_row])
            
            dummy_features = np.zeros((len(predictions), 5))
            dummy_features[:, 3] = predictions
            inverse_predictions = self.scaler.inverse_transform(dummy_features)[:, 3]
            
            return inverse_predictions, None
            
        except Exception as e:
            logger.error(f"Transformer prediction failed: {e}")
            raise

# Model factory function
def create_model(model_type: str, symbol: str, **kwargs) -> ForecastModel:
    """Factory function to create forecasting models."""
    models = {
        "moving_average": MovingAverageModel,
        "arima": ARIMAModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
        "transformer": TransformerModel
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type.lower()](symbol, **kwargs)

def save_model(model: ForecastModel, filename: str):
    """Save a trained model to disk."""
    try:
        if hasattr(model.model, 'save'):  # Keras model
            model.model.save(f"{Config.MODELS_DIR}/{filename}")
        else:  # Other models
            joblib.dump(model, f"{Config.MODELS_DIR}/{filename}")
        logger.info(f"Model saved: {filename}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

def load_model(filename: str) -> ForecastModel:
    """Load a trained model from disk."""
    try:
        return joblib.load(f"{Config.MODELS_DIR}/{filename}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
