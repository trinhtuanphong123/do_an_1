import requests
import pandas as pd
import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple
import sys
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_metrics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
CURRENT_UTC = datetime.strptime("2025-03-28 18:06:47", "%Y-%m-%d %H:%M:%S")
CURRENT_USER = "trinhtuanphong123"

class FinancialMetrics:
    def __init__(self):
        self.api_key = "YOUR_ALPHA_VANTAGE_API_KEY"  # Replace with your API key
        self.base_url = "https://www.alphavantage.co/query"
        self.current_utc = CURRENT_UTC
        self.user_login = CURRENT_USER
        
        # Setup directories
        self.base_dir = Path("data")
        self.metrics_dir = self.base_dir / "financial_metrics"
        self.earnings_dir = self.metrics_dir / "earnings"
        self.volatility_dir = self.metrics_dir / "volatility"
        
        # Create directories
        for dir_path in [self.base_dir, self.metrics_dir, self.earnings_dir, self.volatility_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.last_call_time = None
        logger.info(f"Initialized FinancialMetrics for user {self.user_login} at {self.current_utc}")

    def _wait_for_rate_limit(self):
        """Handle API rate limiting"""
        if self.last_call_time is not None:
            elapsed = time.time() - self.last_call_time
            if elapsed < 12:  # Alpha Vantage rate limit
                time.sleep(12 - elapsed)
        self.last_call_time = time.time()

    def get_earnings_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get earnings data and save to CSV"""
        try:
            logger.info(f"Fetching earnings data for {symbol}")
            
            # Define file paths
            quarterly_file = self.earnings_dir / f"{symbol}_quarterly_earnings.csv"
            annual_file = self.earnings_dir / f"{symbol}_annual_earnings.csv"
            
            self._wait_for_rate_limit()
            
            # Make API request
            params = {
                "function": "EARNINGS",
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            # Process quarterly earnings
            if "quarterlyEarnings" in data:
                quarterly_df = pd.DataFrame(data["quarterlyEarnings"])
                quarterly_df.to_csv(quarterly_file, index=False)
                logger.info(f"Saved quarterly earnings to {quarterly_file}")
            else:
                quarterly_df = pd.DataFrame()
                logger.warning(f"No quarterly earnings data found for {symbol}")
            
            # Process annual earnings
            if "annualEarnings" in data:
                annual_df = pd.DataFrame(data["annualEarnings"])
                annual_df.to_csv(annual_file, index=False)
                logger.info(f"Saved annual earnings to {annual_file}")
            else:
                annual_df = pd.DataFrame()
                logger.warning(f"No annual earnings data found for {symbol}")
            
            return quarterly_df, annual_df
            
        except Exception as e:
            logger.error(f"Error getting earnings data for {symbol}: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def get_atr_data(self, symbol: str, interval: str = "daily") -> pd.DataFrame:
        """Get ATR data and save to CSV"""
        try:
            logger.info(f"Fetching ATR data for {symbol}")
            
            # Define file path
            atr_file = self.volatility_dir / f"{symbol}_ATR_{interval}.csv"
            
            self._wait_for_rate_limit()
            
            # Make API request
            params = {
                "function": "ATR",
                "symbol": symbol,
                "interval": interval,
                "time_period": "14",
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if "Technical Analysis: ATR" in data:
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(data["Technical Analysis: ATR"], orient="index")
                df.index = pd.to_datetime(df.index)
                df.reset_index(inplace=True)
                df.rename(columns={"index": "timestamp", "ATR": "atr"}, inplace=True)
                
                # Save to CSV
                df.to_csv(atr_file, index=False)
                logger.info(f"Saved ATR data to {atr_file}")
                return df
            else:
                logger.warning(f"No ATR data found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting ATR data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def save_session_info(self):
        """Save session information"""
        try:
            session_info = {
                "timestamp_utc": self.current_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "user_login": self.user_login,
                "data_directory": str(self.metrics_dir),
                "earnings_directory": str(self.earnings_dir),
                "volatility_directory": str(self.volatility_dir)
            }
            
            session_file = self.metrics_dir / f"session_{self.current_utc.strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(session_file, "w") as f:
                json.dump(session_info, f, indent=2)
            
            logger.info(f"Saved session info to {session_file}")
            return session_file
            
        except Exception as e:
            logger.error(f"Error saving session info: {str(e)}")
            return None

def main():
    try:
        # Initialize
        fm = FinancialMetrics()
        
        # Save session info
        session_file = fm.save_session_info()
        
        # Test symbols
        symbols = ["AAPL"]
        
        for symbol in symbols:
            logger.info(f"\nProcessing {symbol}")
            
            # Get earnings data
            quarterly_df, annual_df = fm.get_earnings_data(symbol)
            if not quarterly_df.empty:
                logger.info(f"Retrieved {len(quarterly_df)} quarterly earnings records")
            if not annual_df.empty:
                logger.info(f"Retrieved {len(annual_df)} annual earnings records")
            
            # Get ATR data
            atr_df = fm.get_atr_data(symbol)
            if not atr_df.empty:
                logger.info(f"Retrieved {len(atr_df)} ATR records")
            
            logger.info(f"Completed processing {symbol}")
            
        logger.info("\nData collection completed successfully")
        
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")

if __name__ == "__main__":
    main()