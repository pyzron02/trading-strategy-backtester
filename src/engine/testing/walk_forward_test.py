#!/usr/bin/env python3
# walk_forward_test.py - Evaluate strategy performance on out-of-sample data

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import pickle
from tqdm import tqdm

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from engine.run_backtest import run_backtest

class WalkForwardTest:
    """
    Evaluate strategy performance on out-of-sample data to assess real-world applicability.
    
    This test trains the strategy on in-sample data and then evaluates its performance
    on out-of-sample data that was not used during optimization. This helps validate
    that the strategy generalizes well to new, unseen market conditions.
    """
    
    def __init__(self, strategy_name, in_sample_start='2015-01-01', in_sample_end='2019-12-31',
                 out_sample_start='2020-01-01', out_sample_end='2021-12-31', tickers=None,
                 output_dir='output/walk_forward_test', parameters=None, load_optimized=False,
                 optimized_params_path=None):
        """
        Initialize the walk-forward test.
        
        Args:
            strategy_name (str): Name of the strategy to test.
            in_sample_start (str): Start date for the in-sample period in 'YYYY-MM-DD' format.
            in_sample_end (str): End date for the in-sample period in 'YYYY-MM-DD' format.
            out_sample_start (str): Start date for the out-of-sample period in 'YYYY-MM-DD' format.
            out_sample_end (str): End date for the out-of-sample period in 'YYYY-MM-DD' format.
            tickers (list): List of stock ticker symbols. If None, will use all tickers in stock_data.csv.
            output_dir (str): Directory to save test results.
            parameters (dict): Strategy parameters to use. If None, will use default parameters.
            load_optimized (bool): Whether to load optimized parameters from a previous run.
            optimized_params_path (str): Path to the optimized parameters file.
        """
        self.strategy_name = strategy_name
        self.tickers = tickers
        self.in_sample_start = datetime.strptime(in_sample_start, '%Y-%m-%d')
        self.in_sample_end = datetime.strptime(in_sample_end, '%Y-%m-%d')
        self.out_sample_start = datetime.strptime(out_sample_start, '%Y-%m-%d')
        self.out_sample_end = datetime.strptime(out_sample_end, '%Y-%m-%d')
        self.parameters = parameters
        self.load_optimized = load_optimized
        self.optimized_params_path = optimized_params_path
        
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        self.output_dir = os.path.join(project_root, output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load optimized parameters if specified
        if self.load_optimized and self.optimized_params_path:
            self._load_optimized_parameters()
    
    def _load_optimized_parameters(self):
        """Load optimized parameters from a previous run."""
        try:
            with open(self.optimized_params_path, 'rb') as f:
                optimized_results = pickle.load(f)
                
            # Get the best parameters based on Sharpe ratio
            best_params = optimized_results.get('best_params', {})
            if best_params:
                self.parameters = best_params
                print(f"Loaded optimized parameters: {self.parameters}")
            else:
                print("No optimized parameters found. Using default parameters.")
        except Exception as e:
            print(f"Error loading optimized parameters: {e}")
            print("Using default parameters instead.")
    
    def run_in_sample_backtest(self):
        """Run backtest on in-sample data."""
        print(f"\nRunning in-sample backtest ({self.in_sample_start.strftime('%Y-%m-%d')} to {self.in_sample_end.strftime('%Y-%m-%d')})...")
        
        # Create in-sample output directory
        in_sample_dir = os.path.join(self.output_dir, 'in_sample')
        os.makedirs(in_sample_dir, exist_ok=True)
        
        # Run backtest with in-sample date range
        in_sample_results = run_backtest(
            output_dir=in_sample_dir,
            strategy_name=self.strategy_name,
            tickers=self.tickers,
            parameters=self.parameters,
            start_date=self.in_sample_start.strftime('%Y-%m-%d'),
            end_date=self.in_sample_end.strftime('%Y-%m-%d')
        )
        
        return in_sample_results
    
    def run_out_sample_backtest(self):
        """Run backtest on out-of-sample data."""
        print(f"\nRunning out-of-sample backtest ({self.out_sample_start.strftime('%Y-%m-%d')} to {self.out_sample_end.strftime('%Y-%m-%d')})...")
        
        # Create out-of-sample output directory
        out_sample_dir = os.path.join(self.output_dir, 'out_sample')
        os.makedirs(out_sample_dir, exist_ok=True)
        
        # Run backtest with out-of-sample date range
        out_sample_results = run_backtest(
            output_dir=out_sample_dir,
            strategy_name=self.strategy_name,
            tickers=self.tickers,
            parameters=self.parameters,
            start_date=self.out_sample_start.strftime('%Y-%m-%d'),
            end_date=self.out_sample_end.strftime('%Y-%m-%d')
        )
        
        return out_sample_results
    
    def compare_performance(self, in_sample_results, out_sample_results):
        """Compare in-sample and out-of-sample performance."""
        print("\nComparing in-sample and out-of-sample performance...")
        
        # Define metrics to compare
        metrics = ['total_return', 'benchmark_return', 'alpha']
        
        # Create comparison dataframe
        comparison = pd.DataFrame(index=metrics, columns=['In-Sample', 'Out-of-Sample', 'Difference', 'Degradation %'])
        
        for metric in metrics:
            in_val = in_sample_results.get(metric, 0)
            out_val = out_sample_results.get(metric, 0)
            diff = out_val - in_val
            
            # Calculate degradation percentage (avoid division by zero)
            if in_val != 0:
                degradation = (diff / in_val) * 100
            else:
                degradation = 0
                
            comparison.loc[metric] = [in_val, out_val, diff, degradation]
        
        # Save comparison to CSV
        comparison_path = os.path.join(self.output_dir, 'performance_comparison.csv')
        comparison.to_csv(comparison_path)
        print(f"Performance comparison saved to {comparison_path}")
        
        return comparison
    
    def plot_equity_curves(self, in_sample_results, out_sample_results):
        """Plot in-sample and out-of-sample equity curves."""
        print("\nPlotting equity curves...")
        
        # Extract equity curves
        in_sample_equity = in_sample_results.get('equity_curve', [])
        out_sample_equity = out_sample_results.get('equity_curve', [])
        
        if not in_sample_equity or not out_sample_equity:
            print("Error: Equity curve data is missing.")
            return
        
        # Convert to DataFrames
        in_sample_df = pd.DataFrame(in_sample_equity)
        out_sample_df = pd.DataFrame(out_sample_equity)
        
        # Set Date as index
        in_sample_df['Date'] = pd.to_datetime(in_sample_df['Date'])
        out_sample_df['Date'] = pd.to_datetime(out_sample_df['Date'])
        in_sample_df.set_index('Date', inplace=True)
        out_sample_df.set_index('Date', inplace=True)
        
        # Normalize equity curves to start at 100
        in_sample_equity_norm = 100 * (in_sample_df['Value'] / in_sample_df['Value'].iloc[0])
        out_sample_equity_norm = 100 * (out_sample_df['Value'] / out_sample_df['Value'].iloc[0])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot equity curves
        plt.plot(in_sample_equity_norm.index.to_numpy(), in_sample_equity_norm.to_numpy(), label='In-Sample', color='blue')
        plt.plot(out_sample_equity_norm.index.to_numpy(), out_sample_equity_norm.to_numpy(), label='Out-of-Sample', color='red')
        
        # Add vertical line to separate in-sample and out-of-sample periods
        plt.axvline(x=self.in_sample_end, color='black', linestyle='--')
        
        # Add labels and title
        plt.title(f"{self.strategy_name} Equity Curve", fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Normalized Equity (Starting at 100)', fontsize=12)
        plt.legend()
        plt.grid(True)
        
        # Add text annotations
        plt.figtext(0.15, 0.02, f"In-Sample Return: {in_sample_results.get('total_return', 0):.2f}%", fontsize=10)
        plt.figtext(0.65, 0.02, f"Out-of-Sample Return: {out_sample_results.get('total_return', 0):.2f}%", fontsize=10)
        
        # Save plot
        plot_file = os.path.join(self.output_dir, f"{self.strategy_name}_equity_curves.png")
        plt.savefig(plot_file)
        plt.close()
        
        print(f"Equity curves saved to {plot_file}")
    
    def plot_drawdowns(self, in_sample_results, out_sample_results):
        """Plot in-sample and out-of-sample drawdowns."""
        print("\nPlotting drawdowns...")
        
        # Extract drawdown data
        in_sample_dd = in_sample_results.get('drawdowns', pd.DataFrame())
        out_sample_dd = out_sample_results.get('drawdowns', pd.DataFrame())
        
        if in_sample_dd.empty or out_sample_dd.empty:
            print("Error: Drawdown data is missing.")
            return
        
        # Ensure datetime index
        in_sample_dd.index = pd.to_datetime(in_sample_dd.index)
        out_sample_dd.index = pd.to_datetime(out_sample_dd.index)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.plot(in_sample_dd * 100, label='In-Sample', color='blue')
        plt.plot(out_sample_dd * 100, label='Out-of-Sample', color='red')
        
        # Add vertical line separating in-sample and out-of-sample periods
        plt.axvline(x=self.in_sample_end, color='black', linestyle='--', 
                    label=f'Train/Test Split ({self.in_sample_end.strftime("%Y-%m-%d")})')
        
        plt.title(f'{self.strategy_name} Walk-Forward Test: Drawdowns')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Invert y-axis since drawdowns are negative
        plt.gca().invert_yaxis()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'drawdowns_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Drawdowns plot saved to {plot_path}")
    
    def plot_monthly_returns(self, in_sample_results, out_sample_results):
        """Plot in-sample and out-of-sample monthly returns."""
        print("\nPlotting monthly returns...")
        
        # Extract monthly returns
        in_sample_monthly = in_sample_results.get('monthly_returns', pd.DataFrame())
        out_sample_monthly = out_sample_results.get('monthly_returns', pd.DataFrame())
        
        if in_sample_monthly.empty or out_sample_monthly.empty:
            print("Error: Monthly returns data is missing.")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
        
        # Plot in-sample monthly returns
        in_sample_monthly.plot(kind='bar', ax=ax1, color='blue', alpha=0.7)
        ax1.set_title('In-Sample Monthly Returns')
        ax1.set_ylabel('Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # Plot out-of-sample monthly returns
        out_sample_monthly.plot(kind='bar', ax=ax2, color='red', alpha=0.7)
        ax2.set_title('Out-of-Sample Monthly Returns')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'monthly_returns_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Monthly returns plot saved to {plot_path}")
    
    def run_test(self):
        """Run the complete walk-forward test."""
        print(f"\n{'='*80}\nRunning Walk-Forward Test for {self.strategy_name}\n{'='*80}")
        
        # Run in-sample backtest
        in_sample_results = self.run_in_sample_backtest()
        
        # Run out-of-sample backtest
        out_sample_results = self.run_out_sample_backtest()
        
        # Compare performance
        comparison = self.compare_performance(in_sample_results, out_sample_results)
        
        # Plot results
        self.plot_equity_curves(in_sample_results, out_sample_results)
        self.plot_drawdowns(in_sample_results, out_sample_results)
        self.plot_monthly_returns(in_sample_results, out_sample_results)
        
        # Save results
        results = {
            'in_sample_results': in_sample_results,
            'out_sample_results': out_sample_results,
            'comparison': comparison,
            'parameters': self.parameters,
            'in_sample_period': {
                'start': self.in_sample_start.strftime('%Y-%m-%d'),
                'end': self.in_sample_end.strftime('%Y-%m-%d')
            },
            'out_sample_period': {
                'start': self.out_sample_start.strftime('%Y-%m-%d'),
                'end': self.out_sample_end.strftime('%Y-%m-%d')
            }
        }
        
        results_path = os.path.join(self.output_dir, 'walk_forward_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nWalk-Forward Test completed. Results saved to {self.output_dir}")
        
        return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run a walk-forward test for a trading strategy.')
    
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
    
    parser.add_argument('--output_dir', type=str, default='output/walk_forward_test',
                        help='Directory to save test results')
    
    parser.add_argument('--load_optimized', action='store_true',
                        help='Load optimized parameters from a previous run')
    
    parser.add_argument('--optimized_params_path', type=str,
                        help='Path to the optimized parameters file')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Create and run the walk-forward test
    test = WalkForwardTest(
        strategy_name=args.strategy,
        tickers=args.tickers,
        in_sample_start=args.in_sample_start,
        in_sample_end=args.in_sample_end,
        out_sample_start=args.out_sample_start,
        out_sample_end=args.out_sample_end,
        output_dir=args.output_dir,
        load_optimized=args.load_optimized,
        optimized_params_path=args.optimized_params_path
    )
    
    results = test.run_test() 