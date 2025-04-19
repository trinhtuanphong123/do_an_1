# src/beta_analysis.py
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
        logging.FileHandler('beta_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BetaAnalysis:
    def __init__(self):
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.base_url = BASE_URL
        self.data_directory = os.path.join(BASE_DATA_DIR, 'beta_data')
        
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

    def get_beta(self, symbol: str) -> Optional[float]:
        """Get Beta coefficient for a stock"""
        try:
            logger.info(f"Getting Beta for {symbol}")
            
            # Define cache file
            cache_file = os.path.join(self.data_directory, f"{symbol}_beta.json")
            
            # Check cache
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    import json
                    data = json.load(f)
                    if data.get('last_updated'):
                        last_updated = datetime.fromisoformat(data['last_updated'])
                        # Use cache if less than 24 hours old
                        if datetime.now() - last_updated < timedelta(hours=24):
                            logger.info(f"Using cached Beta data for {symbol}")
                            return float(data.get('Beta', 0))

            # Wait for rate limit
            self._wait_for_rate_limit()
            
            # Make API request
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            logger.info(f"Requesting Beta data for {symbol}")
            response = requests.get(self.base_url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Failed to get Beta data: {response.status_code}")
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
                    return self.get_beta(symbol)

            # Get Beta value
            beta = data.get('Beta')
            
            if beta is None:
                logger.error(f"Beta not found in response for {symbol}")
                return None

            # Cache the data
            cache_data = {
                'Symbol': symbol,
                'Beta': beta,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(cache_file, 'w') as f:
                import json
                json.dump(cache_data, f)
            
            logger.info(f"Beta for {symbol}: {beta}")
            return float(beta)

        except Exception as e:
            logger.error(f"Error getting Beta for {symbol}: {str(e)}")
            return None

    def analyze_beta(self, symbol: str) -> Dict:
        """Analyze Beta value and provide interpretation"""
        beta = self.get_beta(symbol)
        
        if beta is None:
            return {
                'symbol': symbol,
                'beta': None,
                'status': 'error',
                'message': 'Failed to retrieve Beta'
            }

        # Analyze Beta value
        analysis = {
            'symbol': symbol,
            'beta': beta,
            'status': 'success',
            'volatility': '',
            'market_sensitivity': '',
            'risk_level': ''
        }

        # Interpret Beta
        if beta == 1.0:
            analysis['volatility'] = 'Equal to market'
            analysis['market_sensitivity'] = 'Moves in line with market'
            analysis['risk_level'] = 'Medium'
        elif beta > 1.0:
            analysis['volatility'] = 'More volatile than market'
            analysis['market_sensitivity'] = 'Amplified market movements'
            analysis['risk_level'] = 'High' if beta > 1.5 else 'Medium-High'
        elif beta < 1.0 and beta > 0:
            analysis['volatility'] = 'Less volatile than market'
            analysis['market_sensitivity'] = 'Dampened market movements'
            analysis['risk_level'] = 'Low' if beta < 0.5 else 'Medium-Low'
        elif beta < 0:
            analysis['volatility'] = 'Inverse to market'
            analysis['market_sensitivity'] = 'Moves opposite to market'
            analysis['risk_level'] = 'Special Case'

        return analysis

def test_beta_analysis():
    """Test function"""
    ba = BetaAnalysis()
    
    # Test with multiple symbols
    symbols = ['AAPL']
    
    for symbol in symbols:
        logger.info(f"\nAnalyzing Beta for {symbol}")
        analysis = ba.analyze_beta(symbol)
        
        if analysis['status'] == 'success':
            logger.info(f"\nBeta Analysis for {symbol}:")
            logger.info(f"Beta: {analysis['beta']:.3f}")
            logger.info(f"Volatility: {analysis['volatility']}")
            logger.info(f"Market Sensitivity: {analysis['market_sensitivity']}")
            logger.info(f"Risk Level: {analysis['risk_level']}")
        else:
            logger.error(f"Failed to analyze {symbol}: {analysis['message']}")
        
        # Wait before next request
        time.sleep(RATE_LIMIT_DELAY)

if __name__ == "__main__":
    # Current time in UTC
    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Analysis started at (UTC): {current_time}")
    logger.info(f"User: trinhtuanphong123")
    
    test_beta_analysis()