# src/technical_indicators.py
import requests
import pandas as pd
import os
import time
import logging
from datetime import datetime
from typing import Optional, Dict
from config.settings import *

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    def __init__(self):
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.base_url = BASE_URL
        
        # Create directories
        self.directories = {
            'sma': os.path.join(TECHNICAL_DATA_DIR, 'sma'),
            'rsi': os.path.join(TECHNICAL_DATA_DIR, 'rsi'),
            'macd': os.path.join(TECHNICAL_DATA_DIR, 'macd'),
            'bbands': os.path.join(TECHNICAL_DATA_DIR, 'bbands')
        }
        
        for directory in self.directories.values():
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        self.last_call_time = None

    def _wait_for_rate_limit(self):
        if self.last_call_time is not None:
            elapsed = time.time() - self.last_call_time
            if elapsed < RATE_LIMIT_DELAY:
                time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_call_time = time.time()

    def get_sma(self, symbol: str, period: int) -> Optional[pd.DataFrame]:
        """Get Simple Moving Average"""
        try:
            cache_file = os.path.join(self.directories['sma'], f"{symbol}_SMA_{period}.csv")
            
            self._wait_for_rate_limit()
            
            params = {
                'function': 'SMA',
                'symbol': symbol,
                'interval': 'daily',
                'time_period': period,
                'series_type': 'close',
                'apikey': self.api_key
            }

            response = requests.get(self.base_url, params=params)
            data = response.json()

            if 'Technical Analysis: SMA' in data:
                df = pd.DataFrame.from_dict(data['Technical Analysis: SMA'], orient='index')
                df.index = pd.to_datetime(df.index)
                df.columns = [f'SMA_{period}']
                df = df.sort_index()
                
                df.to_csv(cache_file)
                logger.info(f"Saved SMA {period} data for {symbol}")
                
                return df
                
            return None

        except Exception as e:
            logger.error(f"Error getting SMA for {symbol}: {str(e)}")
            return None

    def get_rsi(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get Relative Strength Index"""
        try:
            cache_file = os.path.join(self.directories['rsi'], f"{symbol}_RSI.csv")
            
            self._wait_for_rate_limit()
            
            params = {
                'function': 'RSI',
                'symbol': symbol,
                'interval': 'daily',
                'time_period': RSI_PERIOD,
                'series_type': 'close',
                'apikey': self.api_key
            }

            response = requests.get(self.base_url, params=params)
            data = response.json()

            if 'Technical Analysis: RSI' in data:
                df = pd.DataFrame.from_dict(data['Technical Analysis: RSI'], orient='index')
                df.index = pd.to_datetime(df.index)
                df.columns = ['RSI']
                df = df.sort_index()
                
                df.to_csv(cache_file)
                logger.info(f"Saved RSI data for {symbol}")
                
                return df
                
            return None

        except Exception as e:
            logger.error(f"Error getting RSI for {symbol}: {str(e)}")
            return None

    def get_macd(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get MACD"""
        try:
            cache_file = os.path.join(self.directories['macd'], f"{symbol}_MACD.csv")
            
            self._wait_for_rate_limit()
            
            params = {
                'function': 'MACD',
                'symbol': symbol,
                'interval': 'daily',
                'series_type': 'close',
                **MACD_PARAMS,
                'apikey': self.api_key
            }

            response = requests.get(self.base_url, params=params)
            data = response.json()

            if 'Technical Analysis: MACD' in data:
                df = pd.DataFrame.from_dict(data['Technical Analysis: MACD'], orient='index')
                df.index = pd.to_datetime(df.index)
                df.columns = ['MACD', 'MACD_Hist', 'MACD_Signal']
                df = df.sort_index()
                
                df.to_csv(cache_file)
                logger.info(f"Saved MACD data for {symbol}")
                
                return df
                
            return None

        except Exception as e:
            logger.error(f"Error getting MACD for {symbol}: {str(e)}")
            return None

    def get_bbands(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get Bollinger Bands"""
        try:
            cache_file = os.path.join(self.directories['bbands'], f"{symbol}_BBANDS.csv")
            
            self._wait_for_rate_limit()
            
            params = {
                'function': 'BBANDS',
                'symbol': symbol,
                'interval': 'daily',
                'time_period': BBANDS_PERIOD,
                'series_type': 'close',
                'nbdevup': 2,
                'nbdevdn': 2,
                'apikey': self.api_key
            }

            response = requests.get(self.base_url, params=params)
            data = response.json()

            if 'Technical Analysis: BBANDS' in data:
                df = pd.DataFrame.from_dict(data['Technical Analysis: BBANDS'], orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                df.to_csv(cache_file)
                logger.info(f"Saved Bollinger Bands data for {symbol}")
                
                return df
                
            return None

        except Exception as e:
            logger.error(f"Error getting Bollinger Bands for {symbol}: {str(e)}")
            return None

    def get_all_indicators(self, symbol: str):
        """Get all technical indicators"""
        try:
            logger.info(f"\nFetching all technical indicators for {symbol}")
            
            # Get SMA for different periods
            for period in SMA_PERIODS:
                sma = self.get_sma(symbol, period)
                if sma is not None:
                    logger.info(f"Successfully retrieved SMA {period} for {symbol}")
            
            # Get RSI
            rsi = self.get_rsi(symbol)
            if rsi is not None:
                logger.info(f"Successfully retrieved RSI for {symbol}")
            
            # Get MACD
            macd = self.get_macd(symbol)
            if macd is not None:
                logger.info(f"Successfully retrieved MACD for {symbol}")
            
            # Get Bollinger Bands
            bbands = self.get_bbands(symbol)
            if bbands is not None:
                logger.info(f"Successfully retrieved Bollinger Bands for {symbol}")
            
        except Exception as e:
            logger.error(f"Error getting indicators for {symbol}: {str(e)}")