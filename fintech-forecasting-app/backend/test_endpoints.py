"""
Test script for backend endpoints
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("\n=== Testing /health endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_models():
    """Test models list endpoint"""
    print("\n=== Testing /api/models endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/api/models")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Available models: {len(data.get('models', []))}")
        for model in data.get('models', []):
            print(f"  - {model['type']}: {model['name']}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_forecast(symbol="AAPL", model_type="moving_average", horizon=24):
    """Test forecast endpoint"""
    print(f"\n=== Testing /api/forecast/{symbol} with {model_type} ===")
    try:
        payload = {
            "model_type": model_type,
            "horizon": horizon
        }
        response = requests.post(f"{BASE_URL}/api/forecast/{symbol}", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Symbol: {data.get('symbol')}")
            print(f"Model: {data.get('model_type')}")
            print(f"Current Price: ${data.get('current_price', 'N/A')}")
            
            # Check forecast results
            forecast_1h = data.get('forecast_1h', {})
            print(f"\n1h Forecast:")
            print(f"  Direction: {forecast_1h.get('direction', 'N/A')}")
            print(f"  Confidence: {forecast_1h.get('confidence', 'N/A')}")
            print(f"  Predicted Price: ${forecast_1h.get('predicted_price', 'N/A')}")
            
            forecast_4h = data.get('forecast_4h', {})
            print(f"\n4h Forecast:")
            print(f"  Direction: {forecast_4h.get('direction', 'N/A')}")
            print(f"  Confidence: {forecast_4h.get('confidence', 'N/A')}")
            print(f"  Predicted Price: ${forecast_4h.get('predicted_price', 'N/A')}")
            
            forecast_24h = data.get('forecast_24h', {})
            print(f"\n24h Forecast:")
            print(f"  Direction: {forecast_24h.get('direction', 'N/A')}")
            print(f"  Confidence: {forecast_24h.get('confidence', 'N/A')}")
            print(f"  Predicted Price: ${forecast_24h.get('predicted_price', 'N/A')}")
            
            # Technical indicators
            indicators = data.get('technical_indicators', {})
            print(f"\nTechnical Indicators:")
            print(f"  SMA-7: ${indicators.get('sma_7', 'N/A')}")
            print(f"  RSI: {indicators.get('rsi', 'N/A')}")
            print(f"  Volatility: {indicators.get('volatility_pct', 'N/A')}%")
            
            print(f"\nData Points Used: {data.get('data_points_used', 'N/A')}")
            print(f"Data Period: {data.get('data_period', 'N/A')}")
            
        else:
            print(f"Error Response: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("BACKEND ENDPOINT TESTING")
    print("="*60)
    
    # Test health
    health_ok = test_health()
    
    # Test models list
    models_ok = test_models()
    
    # Test forecast with different models
    models_to_test = [
        "moving_average",
        "arima",
        "lstm",
        "gru",
        "transformer"
    ]
    
    forecast_results = {}
    for model in models_to_test:
        forecast_results[model] = test_forecast(symbol="AAPL", model_type=model, horizon=24)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Health Endpoint: {'✓ PASS' if health_ok else '✗ FAIL'}")
    print(f"Models Endpoint: {'✓ PASS' if models_ok else '✗ FAIL'}")
    print("\nForecast Endpoint Tests:")
    for model, result in forecast_results.items():
        print(f"  {model:20s}: {'✓ PASS' if result else '✗ FAIL'}")
    
    all_passed = health_ok and models_ok and all(forecast_results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("="*60)

if __name__ == "__main__":
    run_all_tests()
