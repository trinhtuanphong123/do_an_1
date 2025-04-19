# config/settings.py
ALPHA_VANTAGE_API_KEY = "YS9EOVC069FOEFWK"  # Replace with your API key

# API Settings
BASE_URL = "https://www.alphavantage.co/query"
RATE_LIMIT_DELAY = 15  # seconds between API calls
MAX_RETRIES = 3

# Data Directories
BASE_DATA_DIR = "data"
PRICE_DATA_DIR = f"{BASE_DATA_DIR}/price_data"
TECHNICAL_DATA_DIR = f"{BASE_DATA_DIR}/technical_data"
LOGS_DIR = "logs"

# Technical Indicators Parameters
SMA_PERIODS = [20, 50, 200]
RSI_PERIOD = 14
MACD_PARAMS = {
    'fastperiod': '12',
    'slowperiod': '26',
    'signalperiod': '9'
}
BBANDS_PERIOD = 20

# Symbols to analyze
SYMBOLS = ['AAPL']


# config/settings.py
# Thêm vào phần cài đặt hiện có:

# Market Data Settings
MARKET_INDEX_SYMBOL = 'NDAQ'  # NASDAQ index
MARKET_DATA_UPDATE_INTERVAL = 24  # hours
CORRELATION_WINDOW = 252  # Trading days for correlation calculation (1 year)