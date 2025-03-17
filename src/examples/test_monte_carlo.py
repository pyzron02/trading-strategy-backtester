#!/usr/bin/env python3
# test_monte_carlo.py - Simple test script for the InSampleMonteCarloTest

import os
import sys
from datetime import datetime

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the InSampleMonteCarloTest
from engine.testing.in_sample_monte_carlo import InSampleMonteCarloTest

def main():
    """
    Simple test script for the InSampleMonteCarloTest
    """
    print("\n" + "="*80)
    print("Monte Carlo Test Example")
    print("="*80 + "\n")
    
    # Define tickers to test
    tickers = ['AAPL']
    
    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"output/monte_carlo_test_{timestamp}"
    
    print(f"Running Monte Carlo test with the following parameters:")
    print(f"  Tickers: {tickers}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Number of Permutations: 2")
    print("\n" + "-"*80 + "\n")
    
    # Create and run the Monte Carlo test
    test = InSampleMonteCarloTest(
        strategy_name='SimpleStock',
        tickers=tickers,
        start_date='2020-01-01',
        end_date='2021-12-31',
        output_dir=output_dir,
        num_permutations=2,
        random_seed=42
    )
    
    # Run the test
    results = test.run_test()
    
    # Print the results
    print("\nMonte Carlo Test Results:")
    print(f"P-value (Sharpe Ratio): {results.get('p_value_sharpe', 'N/A')}")
    print(f"P-value (Total Return): {results.get('p_value_returns', 'N/A')}")
    print(f"P-value (Profit Factor): {results.get('p_value_profit_factor', 'N/A')}")
    
    print("\n" + "="*80)
    print("Test Complete")
    print("="*80 + "\n")

if __name__ == '__main__':
    main() 