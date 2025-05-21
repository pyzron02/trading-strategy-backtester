#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct test of pairs trading strategy with backtrader.
"""

import os
import sys
import pandas as pd
import backtrader as bt
from datetime import datetime

# Add project root to path to enable imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our pairs trading strategy
from src.strategies.pairs_trading_strategy import PairsTradingStrategy

# Function to create a pandas dataframe from input CSV
def prepare_data(ticker):
    """Prepare data for backtrader from CSV file."""
    input_file = "input/stock_data.csv"
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        return None
        
    # Read CSV file
    data = pd.read_csv(input_file)
    
    # Ensure we have the required columns for this ticker
    required_cols = [f"{ticker}_Open", f"{ticker}_High", f"{ticker}_Low", f"{ticker}_Close", f"{ticker}_Volume"]
    for col in required_cols:
        if col not in data.columns:
            print(f"Error: Required column {col} not found in input data")
            return None
    
    # Create dataframe with backtrader expected format
    df = pd.DataFrame()
    df['datetime'] = pd.to_datetime(data['Date'])
    df['open'] = data[f"{ticker}_Open"]
    df['high'] = data[f"{ticker}_High"] 
    df['low'] = data[f"{ticker}_Low"]
    df['close'] = data[f"{ticker}_Close"]
    df['volume'] = data[f"{ticker}_Volume"]
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    print(f"Prepared data for {ticker}: {len(df)} rows")
    return df

# Main function
def run_backtest():
    """Run a direct backtrader backtest."""
    # Create a cerebro instance
    cerebro = bt.Cerebro()
    
    # Set our desired cash start
    cerebro.broker.set_cash(100000.0)
    
    # Set commission
    cerebro.broker.setcommission(commission=0.001)
    
    # Prepare data for each ticker
    tickers = ["TSLA", "NVDA"]
    for ticker in tickers:
        df = prepare_data(ticker)
        if df is None:
            print(f"Skipping {ticker} due to data preparation error")
            continue
            
        # Create a data feed
        data = bt.feeds.PandasData(
            dataname=df,
            name=ticker
        )
        
        # Add the data feed to cerebro
        cerebro.adddata(data)
        print(f"Added data feed for {ticker}")
    
    # Add our strategy
    cerebro.addstrategy(
        PairsTradingStrategy,
        lookback_period=60,
        entry_threshold=2.0,
        exit_threshold=0.5,
        position_size=100,
        rebalance_freq=20,
        stop_loss=0.05
    )
    
    # Print starting portfolio value
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    
    # Run the backtest
    print("\nRunning backtest...")
    cerebro.run()
    
    # Print final portfolio value
    print(f"\nFinal Portfolio Value: {cerebro.broker.getvalue():.2f}")
    
    # Plot if requested
    #cerebro.plot()

if __name__ == "__main__":
    run_backtest()