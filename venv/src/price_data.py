# src/price_data.py
import requests
import pandas as pd
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
import sys
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from config.settings import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('price_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockPriceData:
    def __init__(self):
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.base_url = BASE_URL
        self.data_directory = PRICE_DATA_DIR
        
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        
        self.last_call_time = None

    def _wait_for_rate_limit(self):
        """Handle API rate limiting"""
        if self.last_call_time is not None:
            elapsed = time.time() - self.last_call_time
            if elapsed < RATE_LIMIT_DELAY:
                time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_call_time = time.time()

    def get_daily_adjusted(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get daily adjusted stock prices with detailed error handling"""
        try:
            logger.info(f"Getting daily adjusted data for {symbol}")
            
            # Construct cache file path
            cache_file = os.path.join(self.data_directory, f"{symbol}_daily.csv")
            
            # Check cache
            if os.path.exists(cache_file):
                logger.info(f"Found cached data for {symbol}")
                df = pd.read_csv(cache_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Check if data is recent enough (less than 1 day old)
                if datetime.now() - df['timestamp'].max() < timedelta(days=1):
                    logger.info(f"Using cached data for {symbol}")
                    return df
                else:
                    logger.info("Cache is old, fetching new data")

            # Wait for rate limit
            self._wait_for_rate_limit()
            
            # Prepare API request
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': 'full',
                'apikey': self.api_key
            }
            
            # Log API request (without API key)
            debug_params = params.copy()
            debug_params['apikey'] = '***'
            logger.info(f"Making API request with params: {debug_params}")

            # Make API request
            response = requests.get(self.base_url, params=params)
            
            # Log response status
            logger.info(f"Response status code: {response.status_code}")
            
            # Check response status
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}")
                logger.error(f"Response text: {response.text}")
                return None

            # Parse JSON response
            data = response.json()
            
            # Log response keys for debugging
            logger.debug(f"Response keys: {data.keys()}")
            
            # Check for API errors
            if "Error Message" in data:
                logger.error(f"API returned error: {data['Error Message']}")
                return None
                
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                if "API call frequency" in data["Note"]:
                    logger.info("Rate limit reached, waiting...")
                    time.sleep(60)  # Wait for 60 seconds
                    return self.get_daily_adjusted(symbol)  # Retry

            # Check for time series data
            if "Time Series (Daily)" not in data:
                logger.error("No time series data found in response")
                logger.debug(f"Full response: {data}")
                return None

            # Convert data to DataFrame
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            
            # Rename columns
            df.columns = [
                'open', 'high', 'low', 'close', 
                'adjusted_close', 'volume', 
                'dividend_amount', 'split_coefficient'
            ]
            
            # Process index and timestamp
            df.index = pd.to_datetime(df.index)
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'timestamp'}, inplace=True)
            
            # Convert numeric columns
            numeric_columns = [col for col in df.columns if col != 'timestamp']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Sort by date
            df = df.sort_values('timestamp')
            
            # Log data info
            logger.info(f"Retrieved {len(df)} rows of data")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            # Save to cache
            df.to_csv(cache_file, index=False)
            logger.info(f"Saved data to {cache_file}")

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {symbol}: {str(e)}")
            return None
        except ValueError as e:
            logger.error(f"JSON parsing error for {symbol}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {symbol}: {str(e)}")
            return None

def test_price_data():
    """Test function to verify price data retrieval"""
    price_data = StockPriceData()
    
    # Test with a single symbol
    symbol = 'AAPL'
    logger.info(f"\nTesting price data retrieval for {symbol}")
    
    df = price_data.get_daily_adjusted(symbol)
    
    if df is not None:
        logger.info("\nData retrieved successfully!")
        logger.info(f"Shape: {df.shape}")
        logger.info("\nFirst few rows:")
        logger.info(f"\n{df.head()}")
        logger.info("\nLast few rows:")
        logger.info(f"\n{df.tail()}")
    else:
        logger.error("Failed to retrieve data")

if __name__ == "__main__":
    test_price_data()