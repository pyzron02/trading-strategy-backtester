#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the pairs trading strategy.
"""

import os
import sys

# Add project root to path to enable imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import strategy
from src.strategies.pairs_trading_strategy import PairsTradingStrategy

# Register strategy manually
from src.strategies.registry import register_strategy
register_strategy('PairsTrading', PairsTradingStrategy, version="1.0.0")

# Import required components for backtest
from src.engine.run_backtest import run_backtest

# Run a simple backtest
print("Running pairs trading backtest...")
result = run_backtest(
    strategy_name="PairsTrading",
    parameters={
        "lookback_period": 60,
        "entry_threshold": 2.0,
        "exit_threshold": 0.5,
        "position_size": 100,
        "rebalance_freq": 20,
        "stop_loss": 0.05
    },
    tickers=["AAPL", "MSFT"],
    start_date="2020-01-01",
    end_date="2023-12-31",
    initial_capital=100000.0,
    commission=0.001,
    output_dir="output/pairs_test",
    data_dir="input",
    verbose=True
)

if result:
    print(f"Backtest completed with status: {result.get('status', 'unknown')}")
    if 'metrics' in result:
        print("\nPerformance Metrics:")
        for key, value in result['metrics'].items():
            print(f"  {key}: {value}")
else:
    print("Backtest failed or returned no results.")