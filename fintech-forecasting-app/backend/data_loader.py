"""
Optimized data loader for fetching and storing market data using Finnhub API.
"""

import logging
import requests
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import threading

from config import Config
from database import PriceData, store_price_data, get_price_data

logger = logging.getLogger(__name__)

class DataLoader:
    """Optimized data loader for market data using Finnhub API."""
    
    def __init__(self):
        self.api_key = Config.FINNHUB_API_KEY
        self.base_url = "https://finnhub.io/api/v1"
        self.rate_limit_delay = 1  # Finnhub free tier: 60 calls per minute
        
    def load_initial_data(self):
        """Load initial data for popular symbols."""
        popular_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        logger.info("Starting initial data load...")
        
        for i, symbol in enumerate(popular_symbols):
            try:
                # Check if we already have recent data
                existing_data = get_price_data(symbol, 10)
                if existing_data:
                    latest_date = max(pd.timestamp for pd in existing_data)
                    days_old = (datetime.now(timezone.utc) - latest_date).days
                    
                    if days_old < 1:  # Data is less than 1 day old
                        logger.info(f"Recent data exists for {symbol}, skipping")
                        continue
                
                logger.info(f"Fetching data for {symbol}...")
                success = self._fetch_and_store_symbol(symbol)
                
                if success:
                    logger.info(f"Successfully loaded data for {symbol}")
                else:
                    logger.warning(f"Failed to load data for {symbol}")
                
                # Rate limiting
                if i < len(popular_symbols) - 1:
                    logger.info(f"Waiting {self.rate_limit_delay}s for rate limit...")
                    time.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
                continue
        
        logger.info("Initial data load completed")
    
    def _fetch_and_store_symbol(self, symbol: str) -> bool:
        """Fetch and store data for a single symbol using Finnhub API."""
        try:
            # Calculate date range (get last 365 days of data)
            to_timestamp = int(time.time())
            from_timestamp = to_timestamp - (365 * 24 * 60 * 60)  # 1 year ago
            
            # Fetch daily candles from Finnhub
            url = f"{self.base_url}/stock/candle"
            params = {
                'symbol': symbol,
                'resolution': 'D',  # Daily resolution
                'from': from_timestamp,
                'to': to_timestamp,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if data.get('s') == 'no_data':
                logger.error(f"No data available for {symbol}")
                return False
            
            if 'error' in data:
                logger.error(f"API error for {symbol}: {data['error']}")
                return False
            
            # Parse candle data
            if not all(key in data for key in ['t', 'o', 'h', 'l', 'c', 'v']):
                logger.error(f"Incomplete data received for {symbol}")
                return False
            
            stored_count = 0
            
            # Store data points
            for i in range(len(data['t'])):
                try:
                    timestamp = datetime.fromtimestamp(data['t'][i], tz=timezone.utc)
                    
                    price_data = PriceData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open_price=float(data['o'][i]),
                        high=float(data['h'][i]),
                        low=float(data['l'][i]),
                        close=float(data['c'][i]),
                        volume=int(float(data['v'][i]))
                    )
                    
                    if store_price_data(price_data):
                        stored_count += 1
                        
                except (ValueError, KeyError, IndexError) as e:
                    logger.warning(f"Error parsing data for {symbol} at index {i}: {e}")
                    continue
            
            logger.info(f"Stored {stored_count} data points for {symbol}")
            return stored_count > 0
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return False
    
    def fetch_symbol_if_needed(self, symbol: str) -> bool:
        """Fetch data for a symbol if not already available."""
        try:
            # Check if we have any data for this symbol
            existing_data = get_price_data(symbol, 1)
            
            if not existing_data:
                logger.info(f"No data found for {symbol}, fetching...")
                return self._fetch_and_store_symbol(symbol)
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking/fetching data for {symbol}: {e}")
            return False

# Global data loader instance
data_loader = DataLoader()

def load_data_async():
    """Load data asynchronously to avoid blocking app startup."""
    try:
        data_loader.load_initial_data()
    except Exception as e:
        logger.error(f"Error in async data loading: {e}")

def ensure_symbol_data(symbol: str) -> bool:
    """Ensure data exists for a symbol, fetch if needed."""
    return data_loader.fetch_symbol_if_needed(symbol)