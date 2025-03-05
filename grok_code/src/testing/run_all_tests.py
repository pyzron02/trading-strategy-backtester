#!/usr/bin/env python3
# run_all_tests.py - Run all four strategy validation tests in sequence

import os
import sys
import argparse
import subprocess
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run all four strategy validation tests in sequence.')
    
    parser.add_argument('--strategy', type=str, required=True,
                        help='Name of the strategy to test')
    
    parser.add_argument('--tickers', type=str, nargs='+',
                        help='List of ticker symbols to test')
    
    parser.add_argument('--in_sample_start', type=str, default='2015-01-01',
                        help='Start date for in-sample period (YYYY-MM-DD)')
    
    parser.add_argument('--in_sample_end', type=str, default='2019-12-31',
                        help='End date for in-sample period (YYYY-MM-DD)')
    
    parser.add_argument('--out_sample_start', type=str, default='2020-01-01',
                        help='Start date for out-of-sample period (YYYY-MM-DD)')
    
    parser.add_argument('--out_sample_end', type=str, default='2021-12-31',
                        help='End date for out-of-sample period (YYYY-MM-DD)')
    
    parser.add_argument('--output_dir', type=str, default='output/strategy_validation',
                        help='Base directory to save test results')
    
    parser.add_argument('--num_permutations', type=int, default=100,
                        help='Number of permutations for Monte Carlo tests')
    
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def run_test(script_name, args, output_subdir, additional_args=None):
    """Run a specific test script with the given arguments."""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the full output directory path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"{args.strategy}_{output_subdir}_{timestamp}")
    
    # Build the command
    cmd = [sys.executable, os.path.join(current_dir, script_name)]
    
    # Add strategy
    cmd.extend(['--strategy', args.strategy])
    
    # Add tickers if provided
    if args.tickers:
        cmd.append('--tickers')
        cmd.extend(args.tickers)
    
    # Add output directory
    cmd.extend(['--output_dir', output_dir])
    
    # Add additional arguments
    if additional_args:
        for key, value in additional_args.items():
            if value is not None:
                cmd.extend([key, str(value)])
    
    # Print the command
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Run the command
    subprocess.run(cmd)
    
    return output_dir

def main():
    """Run all four strategy validation tests in sequence."""
    args = parse_args()
    
    print(f"\n{'='*80}")
    print(f"Starting Comprehensive Strategy Validation for {args.strategy}")
    print(f"{'='*80}\n")
    
    # Create base output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Run In-Sample Excellence test
    print("\nStep 1: Running In-Sample Excellence test...")
    excellence_output = run_test(
        'in_sample_excellence.py',
        args,
        'excellence',
        {
            '--start_date': args.in_sample_start,
            '--end_date': args.in_sample_end
        }
    )
    
    # Get the path to the optimized parameters
    optimized_params_path = os.path.join(excellence_output, 'best_parameters.pkl')
    
    # 2. Run In-Sample Monte Carlo Permutation test
    print("\nStep 2: Running In-Sample Monte Carlo Permutation test...")
    run_test(
        'in_sample_monte_carlo.py',
        args,
        'in_sample_monte_carlo',
        {
            '--start_date': args.in_sample_start,
            '--end_date': args.in_sample_end,
            '--num_permutations': args.num_permutations,
            '--random_seed': args.random_seed
        }
    )
    
    # 3. Run Walk-Forward test
    print("\nStep 3: Running Walk-Forward test...")
    run_test(
        'walk_forward_test.py',
        args,
        'walk_forward',
        {
            '--in_sample_start': args.in_sample_start,
            '--in_sample_end': args.in_sample_end,
            '--out_sample_start': args.out_sample_start,
            '--out_sample_end': args.out_sample_end,
            '--load_optimized': None,
            '--optimized_params_path': optimized_params_path
        }
    )
    
    # 4. Run Walk-Forward Monte Carlo Permutation test
    print("\nStep 4: Running Walk-Forward Monte Carlo Permutation test...")
    run_test(
        'walk_forward_monte_carlo.py',
        args,
        'walk_forward_monte_carlo',
        {
            '--in_sample_start': args.in_sample_start,
            '--in_sample_end': args.in_sample_end,
            '--out_sample_start': args.out_sample_start,
            '--out_sample_end': args.out_sample_end,
            '--load_optimized': None,
            '--optimized_params_path': optimized_params_path,
            '--num_permutations': args.num_permutations,
            '--random_seed': args.random_seed
        }
    )
    
    print(f"\n{'='*80}")
    print(f"Comprehensive Strategy Validation for {args.strategy} completed!")
    print(f"Results saved to {args.output_dir}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main() 