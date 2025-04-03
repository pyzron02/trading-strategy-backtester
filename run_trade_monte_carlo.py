#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to run Trade-Based Monte Carlo simulations for out-of-sample testing.

This script uses the TradeBasedMonteCarloTest class to:
1. Apply a trading strategy to out-of-sample data to generate trade returns
2. Resample trade returns to create Monte Carlo simulations
3. Analyze the distribution of performance metrics
4. Create visualizations of the results
"""

import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the TradeBasedMonteCarloTest class
from src.monte_carlo.trade_based_monte_carlo import TradeBasedMonteCarloTest

# Import strategy registry
from src.strategies.registry import registry


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Trade-Based Monte Carlo simulations')
    
    # Strategy and tickers
    parser.add_argument('--strategy', type=str, required=True,
                        help='Strategy name (e.g., SimpleStock, MACrossover)')
    parser.add_argument('--tickers', type=str, required=True,
                        help='Comma-separated list of ticker symbols (e.g., AAPL,MSFT)')
    
    # Data and time periods
    parser.add_argument('--input-dir', type=str, default='input',
                        help='Directory containing input data')
    parser.add_argument('--out-of-sample-start', type=str, required=True,
                        help='Start date for out-of-sample period (YYYY-MM-DD)')
    
    # Simulation parameters
    parser.add_argument('--num-simulations', type=int, default=1000,
                        help='Number of Monte Carlo simulations')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                        help='Initial capital for backtest')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-generated)')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose output')
                        
    # Parameters for the strategy (as JSON string)
    parser.add_argument('--parameters', type=str, default="{}",
                        help='JSON string of strategy parameters')
    
    return parser.parse_args()


def main():
    """Run the Trade-Based Monte Carlo simulation."""
    # Parse command line arguments
    args = parse_args()
    
    # Parse tickers
    tickers = args.tickers.split(',')
    
    # Parse parameters
    try:
        parameters = json.loads(args.parameters)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON parameters: {args.parameters}")
        return 1
    
    # Check if strategy exists
    strategy_exists = False
    if args.strategy in ["SimpleStock", "MACrossover"]:
        strategy_exists = True
    else:
        strategy_class = registry.get_strategy_class(args.strategy)
        if strategy_class:
            strategy_exists = True
    
    if not strategy_exists:
        print(f"Error: Strategy '{args.strategy}' not found")
        return 1
    
    # Create output directory if not provided
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", f"trade_monte_carlo_{args.strategy}_{timestamp}")
    else:
        output_dir = args.output_dir
    
    # Create the TradeBasedMonteCarloTest instance
    monte_carlo_test = TradeBasedMonteCarloTest(
        strategy_name=args.strategy,
        parameters=parameters,
        tickers=tickers,
        input_dir=args.input_dir,
        output_dir=output_dir,
        num_simulations=args.num_simulations,
        initial_capital=args.initial_capital,
        seed=args.seed,
        verbose=args.verbose
    )
    
    # Run the test
    results = monte_carlo_test.run_test(args.out_of_sample_start)
    
    # Check if results were generated
    if not results:
        print("Error: No results generated from Monte Carlo test")
        return 1
    
    # Print completion message
    print(f"\nTrade-Based Monte Carlo analysis completed successfully!")
    print(f"Results saved to: {monte_carlo_test.output_dir}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for metric, stats in results.items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Original: {stats['original']:.4f}")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  P-value: {stats['p_value']:.4f}")
        if stats['p_value'] < 0.05:
            if metric != 'max_drawdown' and stats['original'] > stats['mean']:
                print("  Note: Strategy significantly outperforms random simulations")
            elif metric == 'max_drawdown' and stats['original'] < stats['mean']:
                print("  Note: Strategy significantly outperforms random simulations")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 