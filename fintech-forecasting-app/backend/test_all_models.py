"""
Quick validation test for all models
"""
import requests
import json

models = ['moving_average', 'arima', 'lstm', 'gru', 'transformer']
results = {}

print("=" * 60)
print("TESTING ALL MODELS FOR AAPL")
print("=" * 60)

for model in models:
    print(f"\nTesting {model}...", end=" ")
    try:
        r = requests.post(
            'http://localhost:5000/api/forecast/AAPL',
            json={'model_type': model, 'horizon': 24},
            timeout=30
        )
        
        if r.status_code == 200:
            data = r.json()
            forecast_1h = data.get('forecast_1h', {})
            forecast_4h = data.get('forecast_4h', {})
            forecast_24h = data.get('forecast_24h', {})
            
            # Check if we have valid predictions
            has_valid_data = (
                forecast_1h.get('direction') and
                forecast_1h.get('confidence') and
                forecast_1h.get('predicted_price') and
                forecast_4h.get('direction') and
                forecast_24h.get('direction')
            )
            
            if has_valid_data:
                print(f"✓ PASS")
                results[model] = {
                    'status': 'PASS',
                    '1h': f"{forecast_1h.get('direction')} ({forecast_1h.get('confidence')})",
                    '4h': f"{forecast_4h.get('direction')} ({forecast_4h.get('confidence')})",
                    '24h': f"{forecast_24h.get('direction')} ({forecast_24h.get('confidence')})"
                }
            else:
                print(f"✗ FAIL - Invalid data")
                results[model] = {'status': 'FAIL', 'reason': 'Missing predictions'}
        else:
            print(f"✗ FAIL - Status {r.status_code}")
            error = r.json().get('error', 'Unknown error')
            results[model] = {'status': 'FAIL', 'reason': error}
            
    except Exception as e:
        print(f"✗ ERROR - {str(e)[:50]}")
        results[model] = {'status': 'ERROR', 'reason': str(e)[:100]}

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for model, result in results.items():
    if result['status'] == 'PASS':
        print(f"✓ {model:20s} | 1h: {result['1h']:20s} | 4h: {result['4h']:20s} | 24h: {result['24h']}")
    else:
        print(f"✗ {model:20s} | {result.get('reason', 'Unknown error')}")

passed = sum(1 for r in results.values() if r['status'] == 'PASS')
total = len(results)

print(f"\n{'='*60}")
print(f"RESULT: {passed}/{total} models passed")
if passed == total:
    print("✓ ALL TESTS PASSED - Backend is fully functional!")
else:
    print(f"✗ {total - passed} model(s) failed")
print("=" * 60)
