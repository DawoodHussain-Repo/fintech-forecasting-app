"""Quick test for Transformer model"""
import requests
import json

print("Testing Transformer model...")
try:
    r = requests.post(
        'http://localhost:5000/api/forecast/AAPL',
        json={'model_type': 'transformer', 'horizon': 24}
    )
    print(f"Status Code: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        print(f"✓ SUCCESS - Transformer model working!")
        print(f"  Symbol: {data.get('symbol')}")
        print(f"  Current Price: ${data.get('current_price')}")
        print(f"  1h Forecast: {data.get('forecast_1h', {}).get('direction')} ({data.get('forecast_1h', {}).get('confidence')})")
        print(f"  4h Forecast: {data.get('forecast_4h', {}).get('direction')} ({data.get('forecast_4h', {}).get('confidence')})")
        print(f"  24h Forecast: {data.get('forecast_24h', {}).get('direction')} ({data.get('forecast_24h', {}).get('confidence')})")
    else:
        print(f"✗ FAILED")
        print(f"  Error: {r.json()}")
        
except Exception as e:
    print(f"✗ ERROR: {e}")
