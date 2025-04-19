# src/volume_indicators.py
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
        logging.FileHandler('volume_indicators.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VolumeIndicators:
    def __init__(self):
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.base_url = BASE_URL
        self.data_directory = os.path.join(BASE_DATA_DIR, 'volume_data')
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        
        self.last_call_time = None
        self.current_time_utc = datetime.utcnow()
        self.user = "trinhtuanphong123"

    def _wait_for_rate_limit(self):
        """Handle API rate limiting"""
        if self.last_call_time is not None:
            elapsed = time.time() - self.last_call_time
            if elapsed < RATE_LIMIT_DELAY:
                time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_call_time = time.time()

    def get_obv(self, symbol: str, interval: str = 'daily') -> Optional[pd.DataFrame]:
        """Get On Balance Volume (OBV) indicator"""
        try:
            logger.info(f"Getting OBV data for {symbol}")
            logger.info(f"UTC Time: {self.current_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"User: {self.user}")
            
            # Define cache file
            cache_file = os.path.join(self.data_directory, f"{symbol}_OBV_{interval}.csv")
            
            # Check cache
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if datetime.now() - df['timestamp'].max() < timedelta(days=1):
                    logger.info(f"Using cached OBV data for {symbol}")
                    return df

            # Wait for rate limit
            self._wait_for_rate_limit()
            
            # Make API request
            params = {
                'function': 'OBV',
                'symbol': symbol,
                'interval': interval,
                'apikey': self.api_key
            }
            
            logger.info(f"Requesting OBV data for {symbol}")
            response = requests.get(self.base_url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Failed to get OBV data: {response.status_code}")
                return None

            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                logger.error(f"API Error: {data['Error Message']}")
                return None
                
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                if "API call frequency" in data["Note"]:
                    logger.info("Rate limit reached, waiting...")
                    time.sleep(60)
                    return self.get_obv(symbol, interval)

            # Process OBV data
            if 'Technical Analysis: OBV' not in data:
                logger.error("No OBV data found in response")
                return None

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data['Technical Analysis: OBV'], orient='index')
            df.index = pd.to_datetime(df.index)
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'timestamp', 'OBV': 'obv'}, inplace=True)
            
            # Convert OBV to numeric
            df['obv'] = pd.to_numeric(df['obv'], errors='coerce')
            
            # Sort by date
            df = df.sort_values('timestamp')
            
            # Add analysis columns
            df['obv_sma_20'] = df['obv'].rolling(window=20).mean()
            df['obv_sma_50'] = df['obv'].rolling(window=50).mean()
            df['obv_trend'] = df['obv'].diff().apply(lambda x: 'Up' if x > 0 else 'Down' if x < 0 else 'Neutral')
            
            # Save to cache
            df.to_csv(cache_file, index=False)
            logger.info(f"Saved OBV data to {cache_file}")
            
            return df

        except Exception as e:
            logger.error(f"Error getting OBV data for {symbol}: {str(e)}")
            return None

    def analyze_obv(self, symbol: str, interval: str = 'daily') -> Dict:
        """Analyze OBV indicator and provide insights"""
        df = self.get_obv(symbol, interval)
        
        if df is None or df.empty:
            return {
                'symbol': symbol,
                'status': 'error',
                'message': 'Failed to retrieve OBV data'
            }

        try:
            # Get latest data points
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Calculate trends
            short_term_trend = 'Up' if latest['obv'] > latest['obv_sma_20'] else 'Down'
            long_term_trend = 'Up' if latest['obv'] > latest['obv_sma_50'] else 'Down'
            
            # Calculate momentum
            obv_momentum = ((latest['obv'] - previous['obv']) / previous['obv']) * 100
            
            # Determine divergence
            price_trend = self.get_price_trend(symbol)
            divergence = 'None'
            if price_trend:
                if price_trend == 'Up' and short_term_trend == 'Down':
                    divergence = 'Bearish'
                elif price_trend == 'Down' and short_term_trend == 'Up':
                    divergence = 'Bullish'
            
            analysis = {
                'symbol': symbol,
                'timestamp': latest['timestamp'],
                'current_obv': latest['obv'],
                'short_term_trend': short_term_trend,
                'long_term_trend': long_term_trend,
                'momentum': f"{obv_momentum:.2f}%",
                'divergence': divergence,
                'status': 'success'
            }
            
            # Add interpretation
            if short_term_trend == long_term_trend:
                analysis['interpretation'] = f"Strong {short_term_trend.lower()} trend confirmed by both short and long-term indicators"
            else:
                analysis['interpretation'] = "Mixed signals between short and long-term trends"
            
            if abs(obv_momentum) > 5:
                analysis['momentum_interpretation'] = "Strong momentum" if obv_momentum > 0 else "Strong negative momentum"
            else:
                analysis['momentum_interpretation'] = "Weak momentum"
            
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing OBV for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'status': 'error',
                'message': str(e)
            }

    def get_price_trend(self, symbol: str) -> Optional[str]:
        """Helper function to get price trend for divergence analysis"""
        try:
            # Wait for rate limit
            self._wait_for_rate_limit()
            
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            
            if response.status_code != 200:
                return None

            data = response.json()
            
            if "Time Series (Daily)" not in data:
                return None

            # Get last two days of data
            time_series = data["Time Series (Daily)"]
            dates = sorted(time_series.keys(), reverse=True)[:2]
            
            if len(dates) < 2:
                return None

            latest_close = float(time_series[dates[0]]['4. close'])
            previous_close = float(time_series[dates[1]]['4. close'])
            
            return 'Up' if latest_close > previous_close else 'Down'

        except Exception as e:
            logger.error(f"Error getting price trend: {str(e)}")
            return None

def test_volume_indicators():
    """Test function"""
    vi = VolumeIndicators()
    
    # Test with multiple symbols
    symbols = ['AAPL']
    
    for symbol in symbols:
        logger.info(f"\nAnalyzing OBV for {symbol}")
        
        # Get OBV analysis
        analysis = vi.analyze_obv(symbol)
        
        if analysis['status'] == 'success':
            logger.info("\nOBV Analysis Results:")
            logger.info(f"Symbol: {analysis['symbol']}")
            logger.info(f"Timestamp: {analysis['timestamp']}")
            logger.info(f"Current OBV: {analysis['current_obv']:,.0f}")
            logger.info(f"Short-term Trend: {analysis['short_term_trend']}")
            logger.info(f"Long-term Trend: {analysis['long_term_trend']}")
            logger.info(f"Momentum: {analysis['momentum']}")
            logger.info(f"Divergence: {analysis['divergence']}")
            logger.info(f"Interpretation: {analysis['interpretation']}")
            logger.info(f"Momentum Analysis: {analysis['momentum_interpretation']}")
        else:
            logger.error(f"Failed to analyze {symbol}: {analysis['message']}")
        
        # Wait before next request
        time.sleep(RATE_LIMIT_DELAY)

if __name__ == "__main__":
    # Run test
    test_volume_indicators()