#!/usr/bin/env python3
# test_simple_stock_example.py - Example script demonstrating the integrated testing framework

import os
import sys
from datetime import datetime

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Make sure tqdm is installed
try:
    import tqdm
except ImportError:
    print("Installing tqdm...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    import tqdm

# Import the StrategyTester
from engine.testing import StrategyTester

def main():
    """
    Example script demonstrating how to use the integrated testing framework
    with the SimpleStock strategy.
    """
    print("\n" + "="*80)
    print("SimpleStock Strategy Testing Example")
    print("="*80 + "\n")
    
    # Define tickers to test
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # Define date ranges
    in_sample_start = '2015-01-01'
    in_sample_end = '2019-12-31'
    out_sample_start = '2020-01-01'
    out_sample_end = '2021-12-31'
    
    # Define parameter grid for optimization
    parameter_grid = {
        'sma_period': [10, 20, 50, 100, 200],
        'position_size': [10, 20, 50, 100],
        'stop_loss': [0.0, 0.03, 0.05, 0.07],
        'take_profit': [0.0, 0.05, 0.10, 0.15],
        'trail_stop': [False, True],
        'trail_percent': [0.02, 0.03, 0.05]
    }
    
    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"output/simple_stock_example_{timestamp}"
    
    print(f"Testing SimpleStock strategy with the following parameters:")
    print(f"  Tickers: {tickers}")
    print(f"  In-Sample Period: {in_sample_start} to {in_sample_end}")
    print(f"  Out-of-Sample Period: {out_sample_start} to {out_sample_end}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Parameter Grid: {parameter_grid}")
    print("\n" + "-"*80 + "\n")
    
    # Create the strategy tester
    tester = StrategyTester(
        strategy_name='SimpleStock',
        tickers=tickers,
        in_sample_start=in_sample_start,
        in_sample_end=in_sample_end,
        out_sample_start=out_sample_start,
        out_sample_end=out_sample_end,
        output_dir=output_dir,
        num_permutations=50,  # Reduced for faster execution
        random_seed=42,
        parameter_grid=parameter_grid
    )
    
    # Run all tests
    summary = tester.run_all_tests()
    
    # Print the location of the summary report
    summary_file = os.path.join(output_dir, 'SimpleStock_summary.txt')
    print(f"\nSummary report saved to: {summary_file}")
    
    # Print the recommendation
    print(f"\nRecommendation for SimpleStock Strategy:")
    print(f"{'-'*80}")
    print(summary['recommendation'])
    
    # Print test results
    print(f"\nTest Results:")
    print(f"{'-'*80}")
    
    for test_name, test_result in summary['test_results'].items():
        print(f"{test_name.replace('_', ' ').title()}:")
        print(f"  Passed: {'Yes' if test_result['passed'] else 'No'}")
        print(f"  Interpretation: {test_result['interpretation']}")
        print()
    
    print(f"\nTests Passed: {summary['tests_passed']} / {summary['tests_passed'] + summary['tests_failed']}")
    
    # Example of running individual tests
    print("\n" + "="*80)
    print("Running Individual Tests")
    print("="*80 + "\n")
    
    # Create a new tester with a smaller parameter grid for faster execution
    small_grid = {
        'sma_period': [20, 50],
        'position_size': [10, 50]
    }
    
    quick_tester = StrategyTester(
        strategy_name='SimpleStock',
        tickers=['AAPL'],  # Just one ticker for faster execution
        in_sample_start=in_sample_start,
        in_sample_end=in_sample_end,
        out_sample_start=out_sample_start,
        out_sample_end=out_sample_end,
        output_dir=f"{output_dir}/quick_test",
        num_permutations=10,  # Very few permutations for demonstration
        random_seed=42,
        parameter_grid=small_grid
    )
    
    # Run just the in-sample excellence test
    print("\nRunning In-Sample Excellence Test Only:")
    excellence_results = quick_tester.run_in_sample_excellence()
    print(f"Best Parameters: {excellence_results['best_parameters']}")
    print(f"Best Sharpe Ratio: {excellence_results['best_sharpe']:.4f}")
    
    # Run just the walk-forward test
    print("\nRunning Walk-Forward Test Only:")
    walk_forward_results = quick_tester.run_walk_forward()
    print(f"In-Sample Sharpe: {walk_forward_results['in_sample_sharpe']:.4f}")
    print(f"Out-of-Sample Sharpe: {walk_forward_results['out_sample_sharpe']:.4f}")
    print(f"Sharpe Ratio Degradation: {walk_forward_results['sharpe_degradation']:.2%}")
    
    print("\n" + "="*80)
    print("Example Complete")
    print("="*80 + "\n")

if __name__ == '__main__':
    main() 