#!/usr/bin/env python3
"""
Generate sample stock data for walkforward analysis
Creates realistic price data using random walk with realistic parameters
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_stock_data(ticker, start_date, end_date, initial_price, annual_return=0.08, volatility=0.20):
    """Generate realistic stock price data using geometric Brownian motion"""
    
    # Convert dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create date range (business days only)
    dates = pd.date_range(start=start, end=end, freq='B')
    n_days = len(dates)
    
    # Parameters for geometric Brownian motion
    dt = 1/252  # Daily time step (252 trading days per year)
    mu = annual_return  # Expected annual return
    sigma = volatility  # Annual volatility
    
    # Generate random returns
    np.random.seed(42 + hash(ticker) % 1000)  # Deterministic but different for each ticker
    random_returns = np.random.normal(0, 1, n_days)
    
    # Calculate price path using geometric Brownian motion
    returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_returns
    price_multipliers = np.exp(returns)
    
    # Generate prices
    prices = [initial_price]
    for multiplier in price_multipliers[1:]:
        prices.append(prices[-1] * multiplier)
    
    prices = np.array(prices)
    
    # Generate OHLC data
    # Open: previous close with small random gap
    np.random.seed(43 + hash(ticker) % 1000)
    gap_returns = np.random.normal(0, 0.005, n_days)  # Small overnight gaps
    opens = np.roll(prices, 1) * (1 + gap_returns)
    opens[0] = initial_price
    
    # High/Low: based on intraday volatility
    np.random.seed(44 + hash(ticker) % 1000)
    intraday_range = np.random.uniform(0.01, 0.03, n_days)  # 1-3% intraday range
    highs = np.maximum(opens, prices) * (1 + intraday_range/2)
    lows = np.minimum(opens, prices) * (1 - intraday_range/2)
    
    # Volume: realistic trading volume
    np.random.seed(45 + hash(ticker) % 1000)
    base_volume = 50000000 if ticker == 'SPY' else 25000000  # SPY has higher volume
    volume_multiplier = np.random.lognormal(0, 0.5, n_days)
    volumes = (base_volume * volume_multiplier).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Date': dates,
        f'{ticker}_Open': opens,
        f'{ticker}_High': highs,
        f'{ticker}_Low': lows,
        f'{ticker}_Close': prices,
        f'{ticker}_Volume': volumes
    })
    
    return data

def generate_complete_dataset():
    """Generate complete dataset for all required tickers"""
    
    # Configuration
    tickers_config = {
        'SPY': {'initial_price': 320.0, 'return': 0.08, 'volatility': 0.18},
        'QQQ': {'initial_price': 280.0, 'return': 0.12, 'volatility': 0.25},
        'GOOGL': {'initial_price': 1400.0, 'return': 0.10, 'volatility': 0.30},
        'NVDA': {'initial_price': 60.0, 'return': 0.25, 'volatility': 0.45},
        'IWM': {'initial_price': 170.0, 'return': 0.07, 'volatility': 0.22}
    }
    
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    print(f"Generating sample stock data for {list(tickers_config.keys())}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Generate data for each ticker
    all_data = []
    
    for ticker, config in tickers_config.items():
        print(f"Generating data for {ticker}...")
        ticker_data = generate_stock_data(
            ticker, 
            start_date, 
            end_date,
            config['initial_price'],
            config['return'],
            config['volatility']
        )
        all_data.append(ticker_data)
    
    # Combine all data
    print("Combining data...")
    combined_df = all_data[0]
    
    for data in all_data[1:]:
        # Merge on Date
        combined_df = pd.merge(combined_df, data, on='Date', how='outer')
    
    # Sort by date and fill any missing values
    combined_df = combined_df.sort_values('Date')
    combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
    
    # Format date
    combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    output_file = "/home/pyzron02/trading-strategy-backtester/input/stock_data.csv"
    
    # Backup existing file
    if os.path.exists(output_file):
        backup_file = output_file.replace('.csv', '_backup.csv')
        print(f"Backing up existing file to {backup_file}")
        if os.path.exists(backup_file):
            os.remove(backup_file)
        os.rename(output_file, backup_file)
    
    combined_df.to_csv(output_file, index=False)
    print(f"\nData saved to {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    
    # Show sample of SPY data
    print(f"\nSample SPY data:")
    spy_cols = [col for col in combined_df.columns if 'SPY' in col]
    print(combined_df[['Date'] + spy_cols].head())
    
    # Verify data completeness
    print(f"\nData completeness check:")
    for ticker in tickers_config.keys():
        ticker_cols = [col for col in combined_df.columns if ticker in col]
        if ticker_cols:
            close_col = f"{ticker}_Close"
            if close_col in combined_df.columns:
                non_null = combined_df[close_col].notna().sum()
                print(f"  {ticker}: {non_null}/{len(combined_df)} ({non_null/len(combined_df)*100:.1f}%)")
    
    print("\nSample stock data generation completed successfully!")
    print("Note: This is realistic sample data for testing. For production use, replace with real market data.")

if __name__ == "__main__":
    generate_complete_dataset()