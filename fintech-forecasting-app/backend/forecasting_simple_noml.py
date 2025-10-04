"""
Simplified Forecasting Models for Financial Time Series
Implements ARIMA, Moving Average, and basic models without TensorFlow dependency
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

class ModelMetrics:
    """Calculate and store evaluation metrics for forecasting models"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate comprehensive forecast evaluation metrics"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        if len(y_true) == 0:
            return {
                'mse': 0.0,
                'rmse': 0.0,
                'mae': 0.0,
                'mape': 0.0,
                'direction_accuracy': 0.0
            }
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Mean Absolute Percentage Error (handle division by zero)
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        # Directional accuracy (for financial forecasting)
        if len(y_true) > 1:
            direction_true = np.diff(y_true) >= 0
            direction_pred = np.diff(y_pred) >= 0
            direction_accuracy = np.mean(direction_true == direction_pred) * 100
        else:
            direction_accuracy = 0.0
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy)
        }

class AdvancedARIMA:
    """Enhanced ARIMA model with automatic parameter selection"""
    
    def __init__(self, max_p=3, max_d=2, max_q=3):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.model = None
        self.fitted_model = None
        self.best_params = None
        
    def auto_arima(self, data):
        """Automatic ARIMA parameter selection using AIC"""
        best_aic = float('inf')
        best_params = (1, 1, 1)  # Default fallback
        
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    try:
                        if len(data) < 10:  # Not enough data
                            continue
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
        try:
            self.best_params = self.auto_arima(data)
            self.model = ARIMA(data, order=self.best_params)
            self.fitted_model = self.model.fit()
        except Exception as e:
            print(f"ARIMA fit error: {e}")
            # Fallback to simple model
            self.best_params = (1, 1, 1)
            try:
                self.model = ARIMA(data, order=self.best_params)
                self.fitted_model = self.model.fit()
            except:
                self.fitted_model = None
        return self
    
    def predict(self, data, steps=1):
        """Generate forecasts"""
        if self.fitted_model:
            try:
                forecast = self.fitted_model.forecast(steps=steps)
                return np.array(forecast)
            except:
                pass
        
        # Fallback: return last value
        if len(data) > 0:
            return np.full(steps, data[-1])
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
        self.data = np.array(data)
        
        if self.use_exponential and len(data) > 10:
            # Fit exponential smoothing models
            for window in self.window_sizes:
                try:
                    if len(data) >= window * 2:
                        model = ExponentialSmoothing(
                            data, 
                            trend='add', 
                            seasonal=None
                        )
                        self.exp_models[window] = model.fit()
                except:
                    # Fallback to simple exponential smoothing
                    try:
                        model = ExponentialSmoothing(data, trend=None, seasonal=None)
                        self.exp_models[window] = model.fit()
                    except:
                        continue
        
        return self
    
    def predict(self, data, steps=1):
        """Generate ensemble prediction from multiple moving averages"""
        if len(data) == 0:
            return np.zeros(steps)
            
        predictions = []
        
        # Simple moving averages
        for window in self.window_sizes:
            if len(data) >= window:
                ma_pred = np.mean(data[-window:])
                predictions.append(ma_pred)
        
        # Exponential smoothing predictions
        if self.use_exponential:
            for window in self.window_sizes:
                if window in self.exp_models:
                    try:
                        exp_pred = self.exp_models[window].forecast(steps)
                        if hasattr(exp_pred, '__iter__'):
                            predictions.extend(exp_pred[:steps])
                        else:
                            predictions.append(exp_pred)
                    except:
                        continue
        
        # Ensemble average
        if predictions:
            base_pred = np.mean(predictions)
            # Add some variation for multi-step predictions
            result = []
            for i in range(steps):
                # Add small random walk
                pred = base_pred * (1 + np.random.normal(0, 0.01))
                result.append(pred)
                base_pred = pred  # Use previous prediction as base
            return np.array(result)
        
        # Fallback to last value with small trend
        if len(data) > 1:
            trend = (data[-1] - data[-2]) / data[-2] if data[-2] != 0 else 0
            result = []
            last_val = data[-1]
            for i in range(steps):
                next_val = last_val * (1 + trend * 0.1)  # Reduced trend
                result.append(next_val)
                last_val = next_val
            return np.array(result)
        
        return np.full(steps, data[-1] if len(data) > 0 else 0)

class SimpleLSTM:
    """Simplified LSTM-like model using statistical methods"""
    
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model_params = None
        
    def fit(self, data):
        """Fit a statistical approximation of LSTM behavior"""
        data = np.array(data)
        if len(data) < self.sequence_length:
            self.sequence_length = max(1, len(data) // 2)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Extract patterns (simplified LSTM-like memory)
        patterns = []
        targets = []
        
        for i in range(self.sequence_length, len(scaled_data)):
            pattern = scaled_data[i-self.sequence_length:i]
            target = scaled_data[i]
            patterns.append(pattern)
            targets.append(target)
        
        if patterns:
            self.model_params = {
                'patterns': np.array(patterns),
                'targets': np.array(targets),
                'last_sequence': scaled_data[-self.sequence_length:]
            }
        
        return self
    
    def predict(self, data, steps=1):
        """Generate predictions using pattern matching"""
        if self.model_params is None or len(data) == 0:
            return np.full(steps, data[-1] if len(data) > 0 else 0)
        
        # Scale input data
        try:
            scaled_data = self.scaler.transform(np.array(data).reshape(-1, 1)).flatten()
        except:
            return np.full(steps, data[-1] if len(data) > 0 else 0)
        
        predictions = []
        current_sequence = scaled_data[-self.sequence_length:] if len(scaled_data) >= self.sequence_length else self.model_params['last_sequence']
        
        for _ in range(steps):
            # Find most similar pattern
            patterns = self.model_params['patterns']
            targets = self.model_params['targets']
            
            if len(patterns) > 0:
                # Calculate similarity (simplified)
                similarities = []
                for pattern in patterns:
                    if len(pattern) == len(current_sequence):
                        similarity = 1 / (1 + np.mean((pattern - current_sequence) ** 2))
                        similarities.append(similarity)
                    else:
                        similarities.append(0)
                
                # Weighted prediction
                similarities = np.array(similarities)
                if np.sum(similarities) > 0:
                    weights = similarities / np.sum(similarities)
                    pred = np.sum(targets * weights)
                else:
                    pred = targets[-1] if len(targets) > 0 else current_sequence[-1]
            else:
                pred = current_sequence[-1]
            
            predictions.append(pred)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], pred)
        
        # Inverse transform
        try:
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        except:
            # Fallback
            predictions = np.array(predictions) * np.std(data) + np.mean(data)
        
        return predictions

class SimpleGRU:
    """Simplified GRU-like model using statistical methods"""
    
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model_params = None
        
    def fit(self, data):
        """Fit a statistical approximation of GRU behavior"""
        # Similar to SimpleLSTM but with different pattern weighting
        data = np.array(data)
        if len(data) < self.sequence_length:
            self.sequence_length = max(1, len(data) // 2)
        
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Extract recent patterns with exponential weighting
        patterns = []
        targets = []
        weights = []
        
        for i in range(self.sequence_length, len(scaled_data)):
            pattern = scaled_data[i-self.sequence_length:i]
            target = scaled_data[i]
            weight = np.exp(-0.1 * (len(scaled_data) - i))  # Recent patterns weighted more
            
            patterns.append(pattern)
            targets.append(target)
            weights.append(weight)
        
        if patterns:
            self.model_params = {
                'patterns': np.array(patterns),
                'targets': np.array(targets),
                'weights': np.array(weights),
                'last_sequence': scaled_data[-self.sequence_length:]
            }
        
        return self
    
    def predict(self, data, steps=1):
        """Generate predictions using weighted pattern matching"""
        if self.model_params is None or len(data) == 0:
            return np.full(steps, data[-1] if len(data) > 0 else 0)
        
        try:
            scaled_data = self.scaler.transform(np.array(data).reshape(-1, 1)).flatten()
        except:
            return np.full(steps, data[-1] if len(data) > 0 else 0)
        
        predictions = []
        current_sequence = scaled_data[-self.sequence_length:] if len(scaled_data) >= self.sequence_length else self.model_params['last_sequence']
        
        for _ in range(steps):
            patterns = self.model_params['patterns']
            targets = self.model_params['targets']
            pattern_weights = self.model_params['weights']
            
            if len(patterns) > 0:
                # Calculate similarity with recency weighting
                similarities = []
                for i, pattern in enumerate(patterns):
                    if len(pattern) == len(current_sequence):
                        similarity = 1 / (1 + np.mean((pattern - current_sequence) ** 2))
                        similarity *= pattern_weights[i]  # Apply recency weight
                        similarities.append(similarity)
                    else:
                        similarities.append(0)
                
                similarities = np.array(similarities)
                if np.sum(similarities) > 0:
                    weights = similarities / np.sum(similarities)
                    pred = np.sum(targets * weights)
                else:
                    pred = targets[-1] if len(targets) > 0 else current_sequence[-1]
            else:
                pred = current_sequence[-1]
            
            predictions.append(pred)
            current_sequence = np.append(current_sequence[1:], pred)
        
        try:
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        except:
            predictions = np.array(predictions) * np.std(data) + np.mean(data)
        
        return predictions

class SimpleTransformer:
    """Simplified Transformer-like model using attention mechanisms"""
    
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.attention_patterns = None
        
    def fit(self, data):
        """Fit attention-based patterns"""
        data = np.array(data)
        if len(data) < self.sequence_length:
            self.sequence_length = max(1, len(data) // 2)
        
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create attention patterns
        patterns = []
        targets = []
        
        for i in range(self.sequence_length, len(scaled_data)):
            sequence = scaled_data[i-self.sequence_length:i]
            target = scaled_data[i]
            
            # Calculate self-attention weights (simplified)
            attention_weights = np.softmax(sequence)  # Simple attention
            attended_sequence = sequence * attention_weights
            
            patterns.append(attended_sequence)
            targets.append(target)
        
        if patterns:
            self.attention_patterns = {
                'patterns': np.array(patterns),
                'targets': np.array(targets),
                'last_sequence': scaled_data[-self.sequence_length:]
            }
        
        return self
    
    def predict(self, data, steps=1):
        """Generate predictions using attention patterns"""
        if self.attention_patterns is None or len(data) == 0:
            return np.full(steps, data[-1] if len(data) > 0 else 0)
        
        try:
            scaled_data = self.scaler.transform(np.array(data).reshape(-1, 1)).flatten()
        except:
            return np.full(steps, data[-1] if len(data) > 0 else 0)
        
        predictions = []
        current_sequence = scaled_data[-self.sequence_length:] if len(scaled_data) >= self.sequence_length else self.attention_patterns['last_sequence']
        
        for _ in range(steps):
            # Apply attention to current sequence
            attention_weights = np.softmax(current_sequence)
            attended_current = current_sequence * attention_weights
            
            # Find best matching attention pattern
            patterns = self.attention_patterns['patterns']
            targets = self.attention_patterns['targets']
            
            if len(patterns) > 0:
                similarities = []
                for pattern in patterns:
                    if len(pattern) == len(attended_current):
                        # Cosine similarity for attention patterns
                        norm_pattern = pattern / (np.linalg.norm(pattern) + 1e-8)
                        norm_current = attended_current / (np.linalg.norm(attended_current) + 1e-8)
                        similarity = np.dot(norm_pattern, norm_current)
                        similarities.append(max(0, similarity))
                    else:
                        similarities.append(0)
                
                similarities = np.array(similarities)
                if np.sum(similarities) > 0:
                    weights = similarities / np.sum(similarities)
                    pred = np.sum(targets * weights)
                else:
                    pred = targets[-1] if len(targets) > 0 else current_sequence[-1]
            else:
                pred = current_sequence[-1]
            
            predictions.append(pred)
            current_sequence = np.append(current_sequence[1:], pred)
        
        try:
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        except:
            predictions = np.array(predictions) * np.std(data) + np.mean(data)
        
        return predictions

# Factory function for creating models
def create_model(model_type, **kwargs):
    """Factory function to create different model types"""
    
    model_type = model_type.lower()
    
    if model_type == 'arima':
        return AdvancedARIMA(**kwargs)
    elif model_type == 'moving_average':
        return MovingAverageModel(**kwargs)
    elif model_type == 'lstm':
        return SimpleLSTM(**kwargs)
    elif model_type == 'gru':
        return SimpleGRU(**kwargs)
    elif model_type == 'transformer':
        return SimpleTransformer(**kwargs)
    else:
        # Default fallback
        return MovingAverageModel(**kwargs)