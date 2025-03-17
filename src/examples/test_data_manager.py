#!/usr/bin/env python3
# test_data_manager.py - Test script for the DataManager class

import os
import sys
import time

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from engine.data_management import DataManager

def main():
    """Test the DataManager class."""
    print("\n" + "="*80)
    print("DataManager Test")
    print("="*80 + "\n")
    
    # Create DataManager instance
    data_manager = DataManager()
    
    # Print available tickers
    tickers = data_manager.get_available_tickers()
    print(f"Available tickers: {tickers}")
    
    # Print date range
    date_range = data_manager.get_date_range()
    if date_range:
        print(f"Date range: {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")
    
    # Test data loading
    if tickers:
        # Select first 3 tickers or all if less than 3
        test_tickers = tickers[:min(3, len(tickers))]
        
        print(f"\nLoading data for {test_tickers}...")
        start_time = time.time()
        df = data_manager.load_data(test_tickers, '2020-01-01', '2021-12-31')
        load_time = time.time() - start_time
        print(f"Data loaded in {load_time:.2f} seconds")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        
        # Test caching
        print("\nLoading data again (should be faster due to caching)...")
        start_time = time.time()
        df2 = data_manager.load_data(test_tickers, '2020-01-01', '2021-12-31')
        cache_time = time.time() - start_time
        print(f"Data loaded in {cache_time:.2f} seconds")
        print(f"Cache speedup: {load_time / max(cache_time, 0.001):.1f}x")
        
        # Test getting ticker data
        if test_tickers:
            ticker = test_tickers[0]
            print(f"\nGetting data for {ticker}...")
            ticker_df = data_manager.get_ticker_data(ticker, '2020-01-01', '2021-12-31')
            print(f"Ticker data shape: {ticker_df.shape}")
            print(f"Ticker columns: {ticker_df.columns.tolist()}")
            print(f"First few rows:\n{ticker_df.head()}")
    
    print("\n" + "="*80)
    print("DataManager Test Complete")
    print("="*80 + "\n")

if __name__ == '__main__':
    main() 