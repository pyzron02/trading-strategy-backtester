#!/usr/bin/env python3
# data_management.py - Centralized data management system

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import re
from functools import lru_cache
import hashlib

class DataManager:
    """
    Centralized data management system for the trading strategy backtester.
    
    This class handles data loading, caching, preprocessing, and distribution
    to ensure consistent data access across all testing modules.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern to ensure only one data manager exists."""
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, data_dir=None, cache_dir=None):
        """
        Initialize the DataManager.
        
        Args:
            data_dir (str): Directory containing input data files
            cache_dir (str): Directory for caching preprocessed data
        """
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
            
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
        # Set data and cache directories
        self.data_dir = data_dir or os.path.join(project_root, 'input')
        self.cache_dir = cache_dir or os.path.join(project_root, 'cache')
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize data cache
        self.data_cache = {}
        self.available_tickers = set()
        self.date_range = None
        
        # Load available data files
        self._scan_data_files()
        
        self._initialized = True
    
    def _scan_data_files(self):
        """Scan data directory for available data files and identify tickers."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"Created data directory: {self.data_dir}")
            return
            
        # Look for stock_data.csv
        stock_csv = os.path.join(self.data_dir, 'stock_data.csv')
        if os.path.exists(stock_csv):
            # Read CSV to identify available tickers
            try:
                df = pd.read_csv(stock_csv)
                # Extract unique ticker symbols from column names (format: TICKER_Field)
                ticker_pattern = re.compile(r'([A-Z]+)_(?:Open|High|Low|Close|Volume)')
                
                for col in df.columns:
                    match = ticker_pattern.match(col)
                    if match:
                        ticker = match.group(1)
                        if ticker != 'SP500':  # Exclude SP500 as it's often used as benchmark
                            self.available_tickers.add(ticker)
                
                # Determine date range
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    self.date_range = (df['Date'].min(), df['Date'].max())
                
                print(f"Found {len(self.available_tickers)} tickers in stock_data.csv")
                print(f"Date range: {self.date_range[0].strftime('%Y-%m-%d')} to {self.date_range[1].strftime('%Y-%m-%d')}")
            except Exception as e:
                print(f"Error scanning stock_data.csv: {e}")
    
    def _get_cache_key(self, tickers, start_date, end_date, include_benchmark=True):
        """
        Generate a unique cache key based on data parameters.
        
        Args:
            tickers (list): List of ticker symbols
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            include_benchmark (bool): Whether to include benchmark data
            
        Returns:
            str: Unique cache key
        """
        # Sort tickers to ensure consistent cache keys
        sorted_tickers = sorted(tickers)
        
        # Create a string representation of the parameters
        param_str = f"{','.join(sorted_tickers)}|{start_date}|{end_date}|{include_benchmark}"
        
        # Generate MD5 hash as cache key
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def load_data(self, tickers=None, start_date=None, end_date=None, include_benchmark=True, force_reload=False):
        """
        Load and preprocess data for the specified tickers and date range.
        
        Args:
            tickers (list): List of ticker symbols to load
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            include_benchmark (bool): Whether to include benchmark data
            force_reload (bool): Whether to force reload from disk
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Use all available tickers if none specified
        if tickers is None:
            tickers = list(self.available_tickers)
        
        # Generate cache key
        cache_key = self._get_cache_key(tickers, start_date, end_date, include_benchmark)
        
        # Check if data is already in memory cache
        if not force_reload and cache_key in self.data_cache:
            print(f"Using in-memory cached data for {len(tickers)} tickers")
            return self.data_cache[cache_key]
        
        # Check if data is in disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if not force_reload and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"Loaded cached data from {cache_file}")
                
                # Store in memory cache
                self.data_cache[cache_key] = data
                return data
            except Exception as e:
                print(f"Error loading cached data: {e}")
        
        # Load data from CSV
        data = self._load_from_csv(tickers, start_date, end_date, include_benchmark)
        
        # Store in memory and disk cache
        self.data_cache[cache_key] = data
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Cached data to {cache_file}")
        except Exception as e:
            print(f"Error caching data: {e}")
        
        return data
    
    def _load_from_csv(self, tickers, start_date, end_date, include_benchmark):
        """
        Load data from CSV file.
        
        Args:
            tickers (list): List of ticker symbols
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            include_benchmark (bool): Whether to include benchmark data
            
        Returns:
            pd.DataFrame: Loaded and preprocessed data
        """
        stock_csv = os.path.join(self.data_dir, 'stock_data.csv')
        if not os.path.exists(stock_csv):
            raise FileNotFoundError(f"Stock data file not found at {stock_csv}")
        
        # Read CSV
        df = pd.read_csv(stock_csv)
        
        # Convert date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            raise ValueError("CSV file must contain a 'Date' column")
        
        # Filter by date range
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
        
        # Verify all required columns exist for each ticker
        valid_tickers = []
        for ticker in tickers:
            required_cols = [f'{ticker}_{field}' for field in ['Open', 'High', 'Low', 'Close', 'Volume']]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Skipping {ticker} due to missing columns: {missing_cols}")
            else:
                valid_tickers.append(ticker)
        
        # Include benchmark if requested
        if include_benchmark and 'SP500_Close' in df.columns:
            # Calculate daily returns for benchmark
            df['SP500_Return'] = df['SP500_Close'].pct_change()
        
        # Calculate returns for each ticker
        for ticker in valid_tickers:
            df[f'{ticker}_Return'] = df[f'{ticker}_Close'].pct_change()
        
        # Drop rows with NaN returns (first row)
        df = df.dropna(subset=[f'{ticker}_Return' for ticker in valid_tickers])
        
        return df
    
    def get_available_tickers(self):
        """Get list of available ticker symbols."""
        return list(self.available_tickers)
    
    def get_date_range(self):
        """Get available date range."""
        return self.date_range
    
    def clear_cache(self):
        """Clear all cached data."""
        self.data_cache = {}
        
        # Clear disk cache
        for file in os.listdir(self.cache_dir):
            if file.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, file))
        
        print("Cleared all cached data")
    
    def get_ticker_data(self, ticker, start_date=None, end_date=None):
        """
        Get OHLCV data for a specific ticker.
        
        Args:
            ticker (str): Ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: OHLCV data for the ticker
        """
        # Load data for the ticker
        df = self.load_data([ticker], start_date, end_date)
        
        # Extract OHLCV columns for the ticker
        ticker_df = pd.DataFrame({
            'Date': df['Date'],
            'Open': df[f'{ticker}_Open'],
            'High': df[f'{ticker}_High'],
            'Low': df[f'{ticker}_Low'],
            'Close': df[f'{ticker}_Close'],
            'Volume': df[f'{ticker}_Volume'],
            'Return': df[f'{ticker}_Return']
        })
        
        return ticker_df 