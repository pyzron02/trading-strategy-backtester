#!/usr/bin/env python3
"""
Download stock data for walkforward analysis
"""

import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime

def download_stock_data():
    """Download complete stock data for walkforward analysis"""
    
    # Configuration from walkforward_workflow_config.json
    tickers = ["SPY", "GOOGL", "NVDA", "QQQ", "IWM"]  # Include additional tickers from existing data
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    print(f"Downloading stock data for {tickers}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Try downloading all tickers at once to avoid rate limiting
    print("Attempting bulk download...")
    try:
        # Download all tickers at once
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=True)
        
        if data.empty:
            print("No data downloaded with bulk method")
            return
        
        # Restructure the data
        combined_data = {}
        
        for ticker in tickers:
            if ticker in data.columns.levels[0]:
                ticker_data = data[ticker].dropna()
                if len(ticker_data) > 0:
                    # Rename columns to include ticker prefix
                    ticker_data.columns = [f"{ticker}_{col}" for col in ticker_data.columns]
                    combined_data[ticker] = ticker_data
                    print(f"Processed {len(ticker_data)} rows for {ticker}")
                else:
                    print(f"No valid data for {ticker}")
            else:
                print(f"No data found for {ticker} in bulk download")
        
        if not combined_data:
            print("No valid data after processing bulk download")
            return
            
    except Exception as e:
        print(f"Bulk download failed: {e}")
        print("Falling back to individual downloads with delays...")
        
        # Fallback: Download individually with delays
        combined_data = {}
        
        for i, ticker in enumerate(tickers):
            print(f"Downloading {ticker} ({i+1}/{len(tickers)})...")
            try:
                # Add delay to avoid rate limiting
                if i > 0:
                    time.sleep(2)
                
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if len(hist) == 0:
                    print(f"Warning: No data found for {ticker}")
                    continue
                    
                # Rename columns to include ticker prefix
                hist.columns = [f"{ticker}_{col}" for col in hist.columns]
                combined_data[ticker] = hist
                print(f"Downloaded {len(hist)} rows for {ticker}")
                
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                continue
    
    if not combined_data:
        print("No data downloaded successfully!")
        return
    
    # Combine all data into single DataFrame
    print("\nCombining data...")
    combined_df = pd.concat(combined_data.values(), axis=1, join='outer')
    
    # Reset index to make Date a column
    combined_df.reset_index(inplace=True)
    
    # Format date column
    combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Reorder columns to put Date first
    cols = ['Date'] + [col for col in combined_df.columns if col != 'Date']
    combined_df = combined_df[cols]
    
    # Fill NaN values with forward fill, then backward fill
    print("Filling missing values...")
    combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
    
    # Save to CSV file
    output_file = "/home/pyzron02/trading-strategy-backtester/input/stock_data.csv"
    
    # Backup existing file
    if os.path.exists(output_file):
        backup_file = output_file.replace('.csv', '_backup.csv')
        print(f"Backing up existing file to {backup_file}")
        pd.read_csv(output_file).to_csv(backup_file, index=False)
    
    combined_df.to_csv(output_file, index=False)
    print(f"\nData saved to {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    
    # Show sample of SPY data to verify
    print(f"\nSample SPY data:")
    spy_cols = [col for col in combined_df.columns if 'SPY' in col]
    if spy_cols:
        print(combined_df[['Date'] + spy_cols].head())
        print(f"\nSPY data coverage:")
        for col in spy_cols:
            non_null = combined_df[col].notna().sum()
            print(f"  {col}: {non_null}/{len(combined_df)} ({non_null/len(combined_df)*100:.1f}%)")
    
    print("\nStock data download completed successfully!")

if __name__ == "__main__":
    # Check if yfinance is available
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance package not found. Installing...")
        os.system("pip install yfinance")
        import yfinance as yf
    
    download_stock_data()