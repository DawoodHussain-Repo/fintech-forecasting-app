"""
Test model saving through ACTUAL backend API
"""

import requests
import json
import os
from datetime import datetime
import time

BASE_URL = "http://localhost:5000"

def test_via_backend():
    """Test model saving through real API calls"""
    
    print("="*70)
    print("TESTING MODEL SAVING VIA BACKEND API")
    print("="*70)
    
    # Test cases
    test_cases = [
        {"symbol": "TSLA", "model": "moving_average"},
        {"symbol": "MSFT", "model": "arima"},
        {"symbol": "GOOGL", "model": "lstm"}
    ]
    
    for i, test in enumerate(test_cases, 1):
        symbol = test["symbol"]
        model_type = test["model"]
        
        print(f"\n{'─'*70}")
        print(f"Test {i}/3: {symbol} with {model_type}")
        print(f"{'─'*70}")
        
        # Make API request
        url = f"{BASE_URL}/api/forecast/{symbol}"
        payload = {
            "model_type": model_type,
            "horizon": 24
        }
        
        print(f"  ➤ POST {url}")
        print(f"     Payload: {json.dumps(payload)}")
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=120)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ Response: {response.status_code} ({elapsed:.2f}s)")
                print(f"     Symbol: {data.get('symbol')}")
                print(f"     Model: {data.get('model_type')}")
                print(f"     Current Price: ${data.get('current_price', 'N/A')}")
                print(f"     Data Points: {data.get('data_points_used', 'N/A')}")
                print(f"     Cached: {data.get('cached', False)}")
                
                # Check if model file was created
                today = datetime.now().strftime("%Y%m%d")
                model_path = f"models/{symbol}_{model_type}_{today}.pkl"
                
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
                    age_seconds = time.time() - os.path.getmtime(model_path)
                    
                    print(f"  ✓ MODEL FILE SAVED!")
                    print(f"     Path: {model_path}")
                    print(f"     Size: {file_size:,} bytes")
                    print(f"     Created: {mod_time.strftime('%H:%M:%S')} ({age_seconds:.1f}s ago)")
                else:
                    print(f"  ✗ MODEL FILE NOT FOUND: {model_path}")
                
            else:
                print(f"  ✗ Request failed: {response.status_code}")
                print(f"     Error: {response.text[:200]}")
                
        except requests.exceptions.ConnectionError:
            print(f"  ✗ CONNECTION ERROR!")
            print(f"     Backend not running at {BASE_URL}")
            print(f"\n     ⚠ START THE BACKEND FIRST:")
            print(f"     cd backend && python run_server.py")
            return
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
    
    # List all models
    print(f"\n{'='*70}")
    print("ALL MODELS IN backend/models/")
    print(f"{'='*70}")
    
    if os.path.exists("models"):
        files = sorted(os.listdir("models"), key=lambda x: os.path.getmtime(os.path.join("models", x)), reverse=True)
        if files:
            print(f"{'File':<45} {'Size':>12}  {'Modified':<20}")
            print(f"{'-'*45} {'-'*12}  {'-'*20}")
            for f in files:
                full_path = os.path.join("models", f)
                size = os.path.getsize(full_path)
                mod_time = datetime.fromtimestamp(os.path.getmtime(full_path))
                print(f"{f:<45} {size:>10,} B  {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\nTotal: {len(files)} model files")
        else:
            print("  (empty)")
    else:
        print("  models/ directory does not exist!")

if __name__ == "__main__":
    test_via_backend()
