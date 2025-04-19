# main.py
import logging
import os
from datetime import datetime
from src.price_data import StockPriceData
from src.technical_indicators import TechnicalIndicators
from config.settings import *

# Setup logging
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f'stock_analysis_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        # Check API key
        if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "YOUR_API_KEY_HERE":
            logger.error("Please set your Alpha Vantage API key in config/settings.py")
            return

        # Initialize classes
        price_data = StockPriceData()
        technical = TechnicalIndicators()
        
        # Process each symbol
        for symbol in SYMBOLS:
            try:
                logger.info(f"\nProcessing {symbol}")
                
                # Get price data
                df_price = price_data.get_daily_adjusted(symbol)
                if df_price is not None:
                    logger.info(f"Successfully retrieved price data for {symbol}")
                    logger.info(f"Date range: {df_price['timestamp'].min()} to {df_price['timestamp'].max()}")
                    logger.info(f"Total rows: {len(df_price)}")
                
                # Get technical indicators
                technical.get_all_indicators(symbol)
                
                logger.info(f"Completed processing {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")

if __name__ == "__main__":
    main()