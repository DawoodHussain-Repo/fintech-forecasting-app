"""
Background scheduler for automated data collection and model updates.
Periodically fetches new data and retrains models.
"""

import logging
import time
import schedule
from datetime import datetime, timezone
import sys

from utils.config import Config
from main import fetch_and_store_data
from utils.database import get_price_data
from ml.models import create_model, save_model

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def update_data_job():
    """Job to update data for all configured symbols."""
    logger.info("Starting scheduled data update")
    
    success_count = 0
    total_count = len(Config.DATA_COLLECTION_SYMBOLS)
    
    for symbol in Config.DATA_COLLECTION_SYMBOLS:
        try:
            success = fetch_and_store_data(symbol)
            if success:
                success_count += 1
                logger.info(f"Successfully updated data for {symbol}")
            else:
                logger.warning(f"Failed to update data for {symbol}")
                
            # Small delay to avoid rate limiting
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error updating data for {symbol}: {e}")
    
    logger.info(f"Data update completed: {success_count}/{total_count} symbols updated")

def retrain_models_job():
    """Job to retrain models for configured symbols."""
    logger.info("Starting scheduled model retraining")
    
    model_types = ['lstm', 'arima', 'gru']  # Focus on key models
    
    for symbol in Config.DATA_COLLECTION_SYMBOLS:
        # Check if we have enough data
        price_data = get_price_data(symbol, 500)
        
        if len(price_data) < 100:
            logger.warning(f"Insufficient data for {symbol}, skipping model training")
            continue
        
        for model_type in model_types:
            try:
                logger.info(f"Retraining {model_type} model for {symbol}")
                
                # Create and train model
                model = create_model(model_type, symbol)
                training_metrics = model.train(price_data)
                
                # Save model
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                filename = f"{symbol}_{model_type}_{timestamp}.pkl"
                save_model(model, filename)
                
                logger.info(f"Successfully retrained {model_type} for {symbol}")
                logger.info(f"Training metrics: {training_metrics}")
                
                # Small delay between model trainings
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error retraining {model_type} for {symbol}: {e}")
    
    logger.info("Model retraining completed")

def health_check_job():
    """Job to perform system health checks."""
    logger.info("Performing health check")
    
    try:
        # Check database connection
        from utils.database import db_manager
        db_manager.client.admin.command('ping')
        logger.info("✓ Database connection healthy")
        
        # Check data freshness
        total_symbols = len(Config.DATA_COLLECTION_SYMBOLS)
        fresh_data_count = 0
        
        for symbol in Config.DATA_COLLECTION_SYMBOLS:
            price_data = get_price_data(symbol, 1)
            if price_data:
                latest_data = price_data[0]
                hours_old = (datetime.now(timezone.utc) - latest_data.timestamp).total_seconds() / 3600
                if hours_old < 48:  # Data less than 48 hours old
                    fresh_data_count += 1
        
        logger.info(f"✓ Fresh data available for {fresh_data_count}/{total_symbols} symbols")
        
        if fresh_data_count < total_symbols * 0.5:
            logger.warning("⚠ More than 50% of symbols have stale data")
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")

def main():
    """Main scheduler function."""
    logger.info("Starting FinTech Forecasting Scheduler")
    logger.info(f"Update interval: {Config.MODEL_UPDATE_INTERVAL} seconds")
    logger.info(f"Monitoring symbols: {Config.DATA_COLLECTION_SYMBOLS}")
    
    # Schedule jobs
    
    # Update data every hour
    schedule.every().hour.at(":05").do(update_data_job)
    
    # Retrain models every 6 hours
    schedule.every(6).hours.do(retrain_models_job)
    
    # Health check every 30 minutes
    schedule.every(30).minutes.do(health_check_job)
    
    # Daily data update at specific times (when markets are closed/opening)
    schedule.every().day.at("01:00").do(update_data_job)  # After US market close
    schedule.every().day.at("07:00").do(update_data_job)  # Before EU market open
    schedule.every().day.at("13:00").do(update_data_job)  # After EU market close
    
    # Weekly model retraining on Sundays
    schedule.every().sunday.at("02:00").do(retrain_models_job)
    
    logger.info("Scheduler jobs configured:")
    for job in schedule.jobs:
        logger.info(f"  - {job}")
    
    # Run initial jobs
    logger.info("Running initial data update...")
    update_data_job()
    
    logger.info("Running initial health check...")
    health_check_job()
    
    # Main scheduler loop
    logger.info("Scheduler started - waiting for scheduled jobs...")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        raise

if __name__ == '__main__':
    main()
