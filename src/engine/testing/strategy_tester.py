#!/usr/bin/env python3
# strategy_tester.py - Master file for comprehensive strategy testing

import os
import sys
import argparse
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import json
from tqdm import tqdm

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Add the grandparent directory to the path so we can import strategies
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

# Import the testing modules
from engine.testing.in_sample_excellence import InSampleExcellence
from engine.testing.in_sample_monte_carlo import InSampleMonteCarloTest
from engine.testing.walk_forward_test import WalkForwardTest
from engine.testing.walk_forward_monte_carlo import WalkForwardMonteCarloTest

class StrategyTester:
    """
    Integrated testing framework for trading strategies.
    
    This class provides a unified interface for running all four testing methods:
    1. In-Sample Excellence (parameter optimization)
    2. In-Sample Monte Carlo Permutation Test
    3. Walk-Forward Test
    4. Walk-Forward Monte Carlo Permutation Test
    
    It also generates a comprehensive summary report with interpretations and recommendations.
    """
    
    def __init__(self, strategy_name, tickers=None, 
                 in_sample_start='2015-01-01', in_sample_end='2019-12-31',
                 out_sample_start='2020-01-01', out_sample_end='2021-12-31',
                 output_dir='output/strategy_testing', num_permutations=100,
                 random_seed=42, parameter_grid=None, test_types=None):
        """
        Initialize the StrategyTester with the given parameters.
        
        Args:
            strategy_name (str): Name of the strategy to test
            tickers (list): List of ticker symbols to test
            in_sample_start (str): Start date for in-sample period (YYYY-MM-DD)
            in_sample_end (str): End date for in-sample period (YYYY-MM-DD)
            out_sample_start (str): Start date for out-of-sample period (YYYY-MM-DD)
            out_sample_end (str): End date for out-of-sample period (YYYY-MM-DD)
            output_dir (str): Directory to save test results
            num_permutations (int): Number of permutations for Monte Carlo tests
            random_seed (int): Random seed for reproducibility
            parameter_grid (dict): Grid of parameters to optimize
            test_types (list): List of test types to run (default: all tests)
        """
        self.strategy_name = strategy_name
        self.tickers = tickers if tickers is not None else ['SPY']
        self.in_sample_start = in_sample_start
        self.in_sample_end = in_sample_end
        self.out_sample_start = out_sample_start
        self.out_sample_end = out_sample_end
        self.output_dir = output_dir
        self.num_permutations = num_permutations
        self.random_seed = random_seed
        self.parameter_grid = parameter_grid
        
        # Default test types
        if test_types is None:
            self.test_types = [
                'in_sample_excellence',
                'in_sample_monte_carlo',
                'walk_forward',
                'walk_forward_monte_carlo'
            ]
        else:
            self.test_types = test_types
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize results dictionary
        self.results = {
            'in_sample_excellence': None,
            'in_sample_monte_carlo': None,
            'walk_forward': None,
            'walk_forward_monte_carlo': None,
            'summary': None,
            'recommendation': None
        }
        
        # Initialize optimized parameters
        self.optimized_params = None
        
        # Set up logging
        self.log_file = os.path.join(self.output_dir, f"{self.strategy_name}_test_log.txt")
        with open(self.log_file, 'w') as f:
            f.write(f"Strategy Testing Log for {self.strategy_name}\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Parameters:\n")
            f.write(f"  Tickers: {self.tickers}\n")
            f.write(f"  In-Sample Period: {self.in_sample_start} to {self.in_sample_end}\n")
            f.write(f"  Out-of-Sample Period: {self.out_sample_start} to {self.out_sample_end}\n")
            f.write(f"  Number of Permutations: {self.num_permutations}\n")
            f.write(f"  Random Seed: {self.random_seed}\n")
            f.write(f"  Parameter Grid: {self.parameter_grid}\n")
            f.write(f"  Test Types: {self.test_types}\n\n")
    
    def run_in_sample_excellence(self):
        """
        Run the In-Sample Excellence test to optimize strategy parameters.
        
        Returns:
            dict: Results of the test
        """
        print(f"\n{'='*80}\nRunning In-Sample Excellence Test\n{'='*80}")
        
        # Create output directory for this test
        test_output_dir = os.path.join(self.output_dir, 'in_sample_excellence')
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Initialize and run the test
        test = InSampleExcellence(
            strategy_name=self.strategy_name,
            tickers=self.tickers,
            start_date=self.in_sample_start,
            end_date=self.in_sample_end,
            output_dir=test_output_dir,
            random_seed=self.random_seed,
            parameter_grid=self.parameter_grid
        )
        
        results = test.run_optimization()
        
        # Store the optimized parameters
        self.optimized_params = results['best_parameters']
        
        # Store the results
        self.results['in_sample_excellence'] = results
        
        # Log the results
        with open(self.log_file, 'a') as f:
            f.write(f"In-Sample Excellence Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Best Parameters: {results['best_parameters']}\n")
            f.write(f"Best Sharpe Ratio: {results['best_sharpe']:.4f}\n")
            f.write(f"Best Profit Factor: {results['best_profit_factor']:.4f}\n\n")
        
        return results
    
    def run_in_sample_monte_carlo(self):
        """
        Run the In-Sample Monte Carlo Permutation Test.
        
        Returns:
            dict: Results of the test
        """
        print(f"\n{'='*80}\nRunning In-Sample Monte Carlo Test\n{'='*80}")
        
        # Create output directory for this test
        test_output_dir = os.path.join(self.output_dir, 'in_sample_monte_carlo')
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Load optimized parameters if available
        if self.optimized_params is None:
            self._load_optimized_parameters()
        
        # Initialize and run the test
        test = InSampleMonteCarloTest(
            strategy_name=self.strategy_name,
            tickers=self.tickers,
            start_date=self.in_sample_start,
            end_date=self.in_sample_end,
            output_dir=test_output_dir,
            num_permutations=self.num_permutations,
            random_seed=self.random_seed,
            parameters=self.optimized_params
        )
        
        results = test.run_test()
        
        # Store the results
        self.results['in_sample_monte_carlo'] = results
        
        # Log the results
        with open(self.log_file, 'a') as f:
            f.write(f"In-Sample Monte Carlo Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"P-Value (Sharpe): {results['p_value_sharpe']:.4f}\n")
            f.write(f"P-Value (Returns): {results['p_value_returns']:.4f}\n")
            f.write(f"P-Value (Profit Factor): {results['p_value_profit_factor']:.4f}\n\n")
        
        return results
    
    def run_walk_forward(self):
        """
        Run the Walk-Forward Test.
        
        Returns:
            dict: Results of the test
        """
        print(f"\n{'='*80}\nRunning Walk-Forward Test\n{'='*80}")
        
        # Create output directory for this test
        test_output_dir = os.path.join(self.output_dir, 'walk_forward')
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Load optimized parameters if available
        if self.optimized_params is None:
            self._load_optimized_parameters()
        
        # Initialize and run the test
        test = WalkForwardTest(
            strategy_name=self.strategy_name,
            tickers=self.tickers,
            in_sample_start=self.in_sample_start,
            in_sample_end=self.in_sample_end,
            out_sample_start=self.out_sample_start,
            out_sample_end=self.out_sample_end,
            output_dir=test_output_dir,
            random_seed=self.random_seed,
            parameters=self.optimized_params
        )
        
        results = test.run_test()
        
        # Store the results
        self.results['walk_forward'] = results
        
        # Log the results
        with open(self.log_file, 'a') as f:
            f.write(f"Walk-Forward Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"In-Sample Sharpe: {results['in_sample_sharpe']:.4f}\n")
            f.write(f"Out-of-Sample Sharpe: {results['out_sample_sharpe']:.4f}\n")
            f.write(f"Sharpe Ratio Degradation: {results['sharpe_degradation']:.2%}\n\n")
        
        return results
    
    def run_walk_forward_monte_carlo(self):
        """
        Run the Walk-Forward Monte Carlo Permutation Test.
        
        Returns:
            dict: Results of the test
        """
        print(f"\n{'='*80}\nRunning Walk-Forward Monte Carlo Test\n{'='*80}")
        
        # Create output directory for this test
        test_output_dir = os.path.join(self.output_dir, 'walk_forward_monte_carlo')
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Load optimized parameters if available
        if self.optimized_params is None:
            self._load_optimized_parameters()
        
        # Initialize and run the test
        test = WalkForwardMonteCarloTest(
            strategy_name=self.strategy_name,
            tickers=self.tickers,
            in_sample_start=self.in_sample_start,
            in_sample_end=self.in_sample_end,
            out_sample_start=self.out_sample_start,
            out_sample_end=self.out_sample_end,
            output_dir=test_output_dir,
            num_permutations=self.num_permutations,
            random_seed=self.random_seed,
            parameters=self.optimized_params
        )
        
        results = test.run_test()
        
        # Store the results
        self.results['walk_forward_monte_carlo'] = results
        
        # Log the results
        with open(self.log_file, 'a') as f:
            f.write(f"Walk-Forward Monte Carlo Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"P-Value (Sharpe): {results['p_value_sharpe']:.4f}\n")
            f.write(f"P-Value (Returns): {results['p_value_returns']:.4f}\n")
            f.write(f"P-Value (Profit Factor): {results['p_value_profit_factor']:.4f}\n\n")
        
        return results
    
    def _load_optimized_parameters(self):
        """
        Load optimized parameters from the in-sample excellence test.
        If not available, use default parameters.
        """
        # Check if in-sample excellence results are available
        if self.results['in_sample_excellence'] is not None:
            self.optimized_params = self.results['in_sample_excellence']['best_parameters']
            return
        
        # Check if optimized parameters file exists
        params_file = os.path.join(self.output_dir, 'in_sample_excellence', 'best_parameters.pkl')
        if os.path.exists(params_file):
            with open(params_file, 'rb') as f:
                self.optimized_params = pickle.load(f)
            return
        
        # If no optimized parameters are available, use default parameters
        print("Warning: No optimized parameters found. Using default parameters.")
        
        # Try to import the strategy to get default parameters
        try:
            strategy_module = importlib.import_module(f"strategies.{self.strategy_name.lower()}")
            strategy_class = getattr(strategy_module, self.strategy_name)
            self.optimized_params = strategy_class.get_default_parameters()
        except (ImportError, AttributeError):
            print("Warning: Could not import strategy to get default parameters.")
            self.optimized_params = {}
    
    def _save_results(self):
        """
        Save all results to a pickle file.
        """
        results_file = os.path.join(self.output_dir, f"{self.strategy_name}_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(self.results, f)
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report of all test results.
        
        Returns:
            dict: Summary of all test results
        """
        print(f"\n{'='*80}\nGenerating Summary Report\n{'='*80}")
        
        summary = {
            'strategy_name': self.strategy_name,
            'tickers': self.tickers,
            'in_sample_period': f"{self.in_sample_start} to {self.in_sample_end}",
            'out_sample_period': f"{self.out_sample_start} to {self.out_sample_end}",
            'tests_passed': 0,
            'tests_failed': 0,
            'test_results': {}
        }
        
        # In-Sample Excellence
        if self.results['in_sample_excellence'] is not None:
            ise_results = self.results['in_sample_excellence']
            passed = ise_results['best_sharpe'] > 1.0 and ise_results['best_profit_factor'] > 1.5
            
            summary['test_results']['in_sample_excellence'] = {
                'passed': passed,
                'best_sharpe': ise_results['best_sharpe'],
                'best_profit_factor': ise_results['best_profit_factor'],
                'best_parameters': ise_results['best_parameters'],
                'interpretation': (
                    "The strategy shows strong in-sample performance with optimized parameters." 
                    if passed else 
                    "The strategy does not show strong in-sample performance even with optimized parameters."
                )
            }
            
            if passed:
                summary['tests_passed'] += 1
            else:
                summary['tests_failed'] += 1
        
        # In-Sample Monte Carlo
        if self.results['in_sample_monte_carlo'] is not None:
            ismc_results = self.results['in_sample_monte_carlo']
            passed = ismc_results['p_value_sharpe'] < 0.05 or ismc_results['p_value_returns'] < 0.05
            
            summary['test_results']['in_sample_monte_carlo'] = {
                'passed': passed,
                'p_value_sharpe': ismc_results['p_value_sharpe'],
                'p_value_returns': ismc_results['p_value_returns'],
                'p_value_profit_factor': ismc_results['p_value_profit_factor'],
                'interpretation': (
                    "The strategy's in-sample performance is statistically significant and not due to random chance." 
                    if passed else 
                    "The strategy's in-sample performance is not statistically significant and may be due to random chance."
                )
            }
            
            if passed:
                summary['tests_passed'] += 1
            else:
                summary['tests_failed'] += 1
        
        # Walk-Forward Test
        if self.results['walk_forward'] is not None:
            wf_results = self.results['walk_forward']
            passed = (
                wf_results['out_sample_sharpe'] > 0.5 and 
                wf_results['sharpe_degradation'] < 0.5 and
                wf_results['out_sample_profit_factor'] > 1.2
            )
            
            summary['test_results']['walk_forward'] = {
                'passed': passed,
                'in_sample_sharpe': wf_results['in_sample_sharpe'],
                'out_sample_sharpe': wf_results['out_sample_sharpe'],
                'sharpe_degradation': wf_results['sharpe_degradation'],
                'in_sample_profit_factor': wf_results['in_sample_profit_factor'],
                'out_sample_profit_factor': wf_results['out_sample_profit_factor'],
                'interpretation': (
                    "The strategy maintains good performance in out-of-sample data, showing robustness." 
                    if passed else 
                    "The strategy's performance degrades significantly in out-of-sample data, indicating potential overfitting."
                )
            }
            
            if passed:
                summary['tests_passed'] += 1
            else:
                summary['tests_failed'] += 1
        
        # Walk-Forward Monte Carlo
        if self.results['walk_forward_monte_carlo'] is not None:
            wfmc_results = self.results['walk_forward_monte_carlo']
            passed = wfmc_results['p_value_sharpe'] < 0.05 or wfmc_results['p_value_returns'] < 0.05
            
            summary['test_results']['walk_forward_monte_carlo'] = {
                'passed': passed,
                'p_value_sharpe': wfmc_results['p_value_sharpe'],
                'p_value_returns': wfmc_results['p_value_returns'],
                'p_value_profit_factor': wfmc_results['p_value_profit_factor'],
                'interpretation': (
                    "The strategy's out-of-sample performance is statistically significant and not due to random chance." 
                    if passed else 
                    "The strategy's out-of-sample performance is not statistically significant and may be due to random chance."
                )
            }
            
            if passed:
                summary['tests_passed'] += 1
            else:
                summary['tests_failed'] += 1
        
        # Generate overall recommendation
        recommendation = self._generate_recommendation(summary)
        summary['recommendation'] = recommendation
        
        # Store the summary
        self.results['summary'] = summary
        self.results['recommendation'] = recommendation
        
        # Save the summary to a file
        summary_file = os.path.join(self.output_dir, f"{self.strategy_name}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Strategy Testing Summary for {self.strategy_name}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"Strategy: {summary['strategy_name']}\n")
            f.write(f"Tickers: {', '.join(summary['tickers'])}\n")
            f.write(f"In-Sample Period: {summary['in_sample_period']}\n")
            f.write(f"Out-of-Sample Period: {summary['out_sample_period']}\n\n")
            
            f.write(f"Tests Passed: {summary['tests_passed']} / {summary['tests_passed'] + summary['tests_failed']}\n\n")
            
            f.write(f"Test Results:\n")
            f.write(f"{'-'*80}\n\n")
            
            for test_name, test_result in summary['test_results'].items():
                f.write(f"{test_name.replace('_', ' ').title()}:\n")
                f.write(f"  Passed: {'Yes' if test_result['passed'] else 'No'}\n")
                
                for key, value in test_result.items():
                    if key not in ['passed', 'interpretation']:
                        if isinstance(value, float):
                            f.write(f"  {key.replace('_', ' ').title()}: {value:.4f}\n")
                        else:
                            f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
                
                f.write(f"  Interpretation: {test_result['interpretation']}\n\n")
            
            f.write(f"Overall Recommendation:\n")
            f.write(f"{'-'*80}\n\n")
            f.write(f"{recommendation}\n")
        
        # Log the summary
        with open(self.log_file, 'a') as f:
            f.write(f"Summary Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tests Passed: {summary['tests_passed']} / {summary['tests_passed'] + summary['tests_failed']}\n")
            f.write(f"Recommendation: {recommendation.split('.')[0]}\n\n")
        
        return summary
    
    def _generate_recommendation(self, summary):
        """
        Generate an overall recommendation based on the test results.
        
        Args:
            summary (dict): Summary of all test results
            
        Returns:
            str: Overall recommendation
        """
        tests_passed = summary['tests_passed']
        total_tests = summary['tests_passed'] + summary['tests_failed']
        
        if tests_passed == total_tests:
            return (
                "STRONG RECOMMENDATION: This strategy passes all tests and shows excellent performance "
                "both in-sample and out-of-sample. The strategy demonstrates statistical significance "
                "and robustness. It is highly recommended for live trading, subject to proper risk management."
            )
        
        elif tests_passed == 3:
            return (
                "POSITIVE RECOMMENDATION: This strategy passes most tests and shows good performance. "
                "While not perfect, it demonstrates sufficient robustness and statistical significance "
                "to be considered for live trading with careful monitoring."
            )
        
        elif tests_passed == 2:
            return (
                "NEUTRAL RECOMMENDATION: This strategy passes half of the tests. It shows some promise "
                "but also has significant weaknesses. Consider further refinement before live trading, "
                "or use with reduced position sizes and strict risk controls."
            )
        
        elif tests_passed == 1:
            return (
                "NEGATIVE RECOMMENDATION: This strategy passes only one test and fails most validation checks. "
                "It likely suffers from overfitting or lacks statistical significance. Not recommended for "
                "live trading without substantial improvements."
            )
        
        else:  # tests_passed == 0
            return (
                "STRONG NEGATIVE RECOMMENDATION: This strategy fails all tests. It shows poor performance, "
                "lacks statistical significance, and does not generalize to out-of-sample data. "
                "Do not use for live trading. Consider a complete redesign of the strategy."
            )
    
    def run_all_tests(self):
        """
        Run all selected tests and generate a summary report.
        
        Returns:
            dict: Summary of all test results
        """
        print(f"\n{'='*80}")
        print(f"Starting Comprehensive Testing for {self.strategy_name}")
        print(f"{'='*80}\n")
        
        # Run In-Sample Excellence if selected
        if 'in_sample_excellence' in self.test_types:
            self.run_in_sample_excellence()
        
        # Run In-Sample Monte Carlo if selected
        if 'in_sample_monte_carlo' in self.test_types:
            self.run_in_sample_monte_carlo()
        
        # Run Walk-Forward Test if selected
        if 'walk_forward' in self.test_types:
            self.run_walk_forward()
        
        # Run Walk-Forward Monte Carlo if selected
        if 'walk_forward_monte_carlo' in self.test_types:
            self.run_walk_forward_monte_carlo()
        
        # Generate summary report
        summary = self.generate_summary_report()
        
        # Save all results
        self._save_results()
        
        print(f"\n{'='*80}")
        print(f"Testing Complete for {self.strategy_name}")
        print(f"{'='*80}\n")
        
        return summary

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Comprehensive Strategy Testing Framework')
    
    parser.add_argument('--strategy', type=str, required=True,
                        help='Name of the strategy to test')
    
    parser.add_argument('--tickers', type=str, nargs='+', default=['SPY'],
                        help='List of ticker symbols to test')
    
    parser.add_argument('--in_sample_start', type=str, default='2015-01-01',
                        help='Start date for in-sample period (YYYY-MM-DD)')
    
    parser.add_argument('--in_sample_end', type=str, default='2019-12-31',
                        help='End date for in-sample period (YYYY-MM-DD)')
    
    parser.add_argument('--out_sample_start', type=str, default='2020-01-01',
                        help='Start date for out-of-sample period (YYYY-MM-DD)')
    
    parser.add_argument('--out_sample_end', type=str, default='2021-12-31',
                        help='End date for out-of-sample period (YYYY-MM-DD)')
    
    parser.add_argument('--output_dir', type=str, default='output/strategy_testing',
                        help='Directory to save test results')
    
    parser.add_argument('--num_permutations', type=int, default=100,
                        help='Number of permutations for Monte Carlo tests')
    
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--param_grid', type=str, default=None,
                        help='JSON string of parameter grid for optimization')
    
    parser.add_argument('--test_types', type=str, nargs='+', 
                        choices=['in_sample_excellence', 'in_sample_monte_carlo', 
                                'walk_forward', 'walk_forward_monte_carlo', 'all'],
                        default=['all'],
                        help='List of test types to run')
    
    return parser.parse_args()

def main():
    """
    Main function to run the strategy tester from the command line.
    """
    args = parse_args()
    
    # Process test types
    if 'all' in args.test_types:
        test_types = [
            'in_sample_excellence',
            'in_sample_monte_carlo',
            'walk_forward',
            'walk_forward_monte_carlo'
        ]
    else:
        test_types = args.test_types
    
    # Process parameter grid
    if args.param_grid:
        parameter_grid = json.loads(args.param_grid)
    else:
        parameter_grid = None
    
    # Create and run the strategy tester
    tester = StrategyTester(
        strategy_name=args.strategy,
        tickers=args.tickers,
        in_sample_start=args.in_sample_start,
        in_sample_end=args.in_sample_end,
        out_sample_start=args.out_sample_start,
        out_sample_end=args.out_sample_end,
        output_dir=args.output_dir,
        num_permutations=args.num_permutations,
        random_seed=args.random_seed,
        parameter_grid=parameter_grid,
        test_types=test_types
    )
    
    # Run all tests
    summary = tester.run_all_tests()
    
    # Print recommendation
    print(f"\nRecommendation for {args.strategy}:")
    print(f"{'-'*80}")
    print(summary['recommendation'])

if __name__ == '__main__':
    main() 