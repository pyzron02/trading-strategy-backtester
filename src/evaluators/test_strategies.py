#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify that all strategy implementations work with the complete workflow.

This script will test each strategy with a simple backtest to ensure they are correctly
implemented and work with the complete workflow.
"""

import os
import sys
from datetime import datetime

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import the run_backtest function
from engine.run_backtest import run_backtest

def test_strategy(strategy_name, parameters=None):
    """
    Test a strategy with a simple backtest.
    
    Args:
        strategy_name (str): Name of the strategy to test
        parameters (dict, optional): Strategy parameters
        
    Returns:
        dict: Backtest results
    """
    print(f"\n{'='*80}\nTesting strategy: {strategy_name}\n{'='*80}")
    
    # Create a timestamp for the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/test_{strategy_name}_{timestamp}"
    
    # Default parameters if none provided
    if parameters is None:
        if strategy_name == 'SimpleStock':
            parameters = {'sma_period': 20, 'position_size': 10}
        elif strategy_name == 'MACrossover':
            parameters = {'fast_period': 10, 'slow_period': 30, 'position_size': 100}
        elif strategy_name == 'MultiPosition':
            parameters = {'sma_period': 20, 'position_size': 100}
        elif strategy_name == 'AuctionMarket':
            parameters = {'param_preset': 'default', 'position_size': 100}
    
    # Run the backtest
    results = run_backtest(
        output_dir=output_dir,
        strategy_name=strategy_name,
        tickers=['AAPL'],
        parameters=parameters,
        start_date='2020-01-01',
        end_date='2021-12-31'
    )
    
    # Print some results
    print(f"\nBacktest Results for {strategy_name}:")
    print(f"Final Portfolio Value: {results.get('final_value', 0):.2f}")
    print(f"Return: {results.get('total_return', 0):.2%}")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.4f}")
    print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
    print(f"Number of Trades: {results.get('total_trades', 0)}")
    
    return results

def main():
    """Run tests for all strategies."""
    # Test each strategy
    strategies = ['SimpleStock', 'MACrossover', 'MultiPosition', 'AuctionMarket']
    results = {}
    
    for strategy in strategies:
        try:
            results[strategy] = test_strategy(strategy)
        except Exception as e:
            print(f"Error testing {strategy}: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("Strategy Test Summary")
    print("="*80)
    
    for strategy, result in results.items():
        print(f"{strategy}: Return = {result.get('total_return', 0):.2%}, Sharpe = {result.get('sharpe_ratio', 0):.2f}")
    
    return results

if __name__ == "__main__":
    main() 