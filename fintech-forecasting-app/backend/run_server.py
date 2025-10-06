"""
Alternative Flask runner for Windows compatibility.
This version avoids the threading issues that can occur on Windows.
"""

import os
import sys
import signal
import logging
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Set environment variables for Windows compatibility
os.environ['FLASK_ENV'] = 'development'
os.environ['FLASK_DEBUG'] = 'False'  # Disable debug to avoid threading issues

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def cleanup_handler(signum, frame):
    """Handle cleanup on shutdown."""
    logger.info("Shutting down server...")
    try:
        logger.info("Server shutdown complete")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    sys.exit(0)

if __name__ == '__main__':
    try:
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)
        
        # Import after setting environment
        from main import app
        
        print("Starting FinTech Forecasting API...")
        print("Backend: http://localhost:5000")
        print("Logs: Check console for API requests")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        # Use simple Flask server without debug mode
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except KeyboardInterrupt:
        cleanup_handler(None, None)
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Try running: pip install flask flask-cors pymongo python-dotenv")
        input("Press Enter to exit...")
