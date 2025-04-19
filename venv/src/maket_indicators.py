# src/market_indicators.py
import requests
import pandas as pd
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
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
        logging.FileHandler('market_indicators.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketIndicators:
    def __init__(self):
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.base_url = BASE_URL
        self.data_directory = os.path.join(BASE_DATA_DIR, 'market_data')
        
        # Create directory if it doesn't exist
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

    def get_index_data(self, index_symbol: str = 'NDAQ') -> Optional[pd.DataFrame]:
        """Get market index data"""
        try:
            logger.info(f"Getting market index data for {index_symbol}")
            
            cache_file = os.path.join(self.data_directory, f"{index_symbol}_daily.csv")
            
            # Check cache
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if datetime.now() - df['timestamp'].max() < timedelta(days=1):
                    logger.info(f"Using cached data for {index_symbol}")
                    return df

            self._wait_for_rate_limit()
            
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': index_symbol,
                'outputsize': 'full',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Failed to get index data: {response.status_code}")
                return None

            data = response.json()
            
            if "Time Series (Daily)" not in data:
                logger.error("No time series data found in response")
                return None

            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            
            # Rename columns
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
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
            
            # Save to cache
            df.to_csv(cache_file, index=False)
            logger.info(f"Saved index data to {cache_file}")

            return df

        except Exception as e:
            logger.error(f"Error getting index data: {str(e)}")
            return None

    def get_market_beta(self, symbol: str) -> Optional[float]:
        """Get stock's beta from OVERVIEW endpoint"""
        try:
            logger.info(f"Getting beta for {symbol}")
            
            cache_file = os.path.join(self.data_directory, f"{symbol}_overview.json")
            
            # Check cache
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    import json
                    data = json.load(f)
                    if data.get('last_updated'):
                        last_updated = datetime.fromisoformat(data['last_updated'])
                        if datetime.now() - last_updated < timedelta(days=1):
                            logger.info(f"Using cached overview data for {symbol}")
                            return float(data.get('Beta', 0))

            self._wait_for_rate_limit()
            
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Failed to get overview data: {response.status_code}")
                return None

            data = response.json()
            
            # Add timestamp and save to cache
            data['last_updated'] = datetime.now().isoformat()
            with open(cache_file, 'w') as f:
                import json
                json.dump(data, f)
            
            beta = data.get('Beta')
            if beta:
                return float(beta)
            else:
                logger.error(f"Beta not found for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Error getting beta: {str(e)}")
            return None

    def get_market_correlation(self, symbol: str, index_symbol: str = 'NDAQ') -> Optional[float]:
        """Calculate correlation between stock and market index"""
        try:
            # Get stock data
            stock_data = self.get_daily_stock_data(symbol)
            if stock_data is None:
                return None

            # Get index data
            index_data = self.get_index_data(index_symbol)
            if index_data is None:
                return None

            # Align dates
            merged_data = pd.merge(
                stock_data[['timestamp', 'close']].rename(columns={'close': 'stock'}),
                index_data[['timestamp', 'close']].rename(columns={'close': 'index'}),
                on='timestamp'
            )

            # Calculate correlation
            correlation = merged_data['stock'].corr(merged_data['index'])
            
            logger.info(f"Correlation between {symbol} and {index_symbol}: {correlation:.4f}")
            
            return correlation

        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return None

    def get_daily_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get daily stock data for correlation calculation"""
        try:
            self._wait_for_rate_limit()
            
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'outputsize': 'full',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            
            if response.status_code != 200:
                return None

            data = response.json()
            
            if "Time Series (Daily)" not in data:
                return None

            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df.index)
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'timestamp'}, inplace=True)
            
            for col in df.columns:
                if col != 'timestamp':
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return df.sort_values('timestamp')

        except Exception as e:
            logger.error(f"Error getting stock data: {str(e)}")
            return None

def test_market_indicators():
    """Test function"""
    mi = MarketIndicators()
    
    # Test index data
    logger.info("\nTesting NASDAQ index data:")
    index_df = mi.get_index_data()
    if index_df is not None:
        logger.info(f"Retrieved {len(index_df)} rows of index data")
        logger.info(f"Date range: {index_df['timestamp'].min()} to {index_df['timestamp'].max()}")
    
    # Test beta
    symbol = 'AAPL'
    logger.info(f"\nTesting beta for {symbol}:")
    beta = mi.get_market_beta(symbol)
    if beta is not None:
        logger.info(f"Beta: {beta}")
    
    # Test correlation
    logger.info(f"\nTesting market correlation for {symbol}:")
    correlation = mi.get_market_correlation(symbol)
    if correlation is not None:
        logger.info(f"Correlation: {correlation:.4f}")

if __name__ == "__main__":
    test_market_indicators()