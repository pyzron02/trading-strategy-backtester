#!/usr/bin/env python3
# simple_test.py - Simple example of using the backtesting framework

import os
import sys
from datetime import datetime

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the run_backtest function
from engine.run_backtest import run_backtest

def main():
    """
    Simple example of running a backtest with the SimpleStock strategy.
    """
    print("\n" + "="*80)
    print("SimpleStock Strategy Backtest Example")
    print("="*80 + "\n")
    
    # Define tickers to test
    tickers = ['AAPL']
    
    # Define parameters for the strategy
    parameters = {
        'sma_period': 50,
        'position_size': 10
    }
    
    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"output/simple_test_{timestamp}"
    
    print(f"Running backtest with SimpleStock strategy:")
    print(f"  Tickers: {tickers}")
    print(f"  Parameters: {parameters}")
    print(f"  Output Directory: {output_dir}")
    print("\n" + "-"*80 + "\n")
    
    # Run the backtest
    results_path = run_backtest(
        strategy_name='SimpleStock',
        tickers=tickers,
        parameters=parameters,
        output_dir=output_dir
    )
    
    print(f"\nBacktest completed. Results saved to: {results_path}")
    print("\n" + "="*80)
    print("Example Complete")
    print("="*80 + "\n")

if __name__ == '__main__':
    main() 