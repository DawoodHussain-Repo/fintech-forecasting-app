"""
Direct test script - trains models and checks if they're saved to backend/models/
"""

import os
import sys
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_saving_direct():
    """Test model saving directly without API"""
    
    print("="*70)
    print("TESTING MODEL SAVING TO backend/models/")
    print("="*70)
    
    # Import after path setup
    from ml.models import create_model
    import numpy as np
    
    # Test data
    test_symbols = ["NVDA", "AAPL", "TSLA"]
    test_models = ["moving_average", "arima"]
    
    # Generate fake price data
    np.random.seed(42)
    fake_prices = np.random.normal(100, 5, 100)  # 100 price points
    
    for symbol in test_symbols:
        for model_type in test_models:
            print(f"\n{'─'*70}")
            print(f"Testing: {symbol} - {model_type}")
            print(f"{'─'*70}")
            
            try:
                # Create model
                print(f"  ➤ Creating {model_type} model...")
                model = create_model(model_type)
                
                # Train model
                print(f"  ➤ Training model on {len(fake_prices)} data points...")
                model = model.fit(fake_prices)
                
                # Save model
                print(f"  ➤ Saving model to backend/models/...")
                save_success = model.save_model(symbol, model_type)
                
                if save_success:
                    # Check if file exists
                    today = datetime.now().strftime("%Y%m%d")
                    model_path = f"models/{symbol}_{model_type}_{today}.pkl"
                    
                    if os.path.exists(model_path):
                        file_size = os.path.getsize(model_path)
                        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
                        
                        print(f"  ✓ SUCCESS!")
                        print(f"     File: {model_path}")
                        print(f"     Size: {file_size:,} bytes")
                        print(f"     Modified: {mod_time.strftime('%H:%M:%S')}")
                    else:
                        print(f"  ✗ FAILED - File not found: {model_path}")
                else:
                    print(f"  ✗ FAILED - save_model returned False")
                    
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # List all saved models
    print(f"\n{'='*70}")
    print("ALL MODELS IN backend/models/")
    print(f"{'='*70}")
    
    if os.path.exists("models"):
        files = sorted(os.listdir("models"))
        if files:
            for f in files:
                full_path = os.path.join("models", f)
                size = os.path.getsize(full_path)
                mod_time = datetime.fromtimestamp(os.path.getmtime(full_path))
                print(f"  {f:40s} {size:>10,} bytes  {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\nTotal: {len(files)} model files")
        else:
            print("  (empty)")
    else:
        print("  models/ directory does not exist!")

if __name__ == "__main__":
    test_model_saving_direct()
