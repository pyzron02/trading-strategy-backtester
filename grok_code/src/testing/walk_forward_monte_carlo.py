#!/usr/bin/env python3
# walk_forward_monte_carlo.py - Validate out-of-sample performance against permuted data

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
import random

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from engine.run_backtest import run_backtest
from testing.walk_forward_test import WalkForwardTest

class WalkForwardMonteCarloTest:
    """
    Validate out-of-sample performance by comparing it to permuted data.
    
    This test extends the walk-forward test by running the strategy on permuted
    out-of-sample data to determine if the out-of-sample performance is statistically
    significant or could have occurred by chance.
    """
    
    def __init__(self, strategy_name, in_sample_start='2015-01-01', in_sample_end='2019-12-31',
                 out_sample_start='2020-01-01', out_sample_end='2021-12-31', tickers=None,
                 output_dir='output/walk_forward_monte_carlo', parameters=None, 
                 load_optimized=False, optimized_params_path=None, num_permutations=100,
                 random_seed=42):
        """
        Initialize the walk-forward Monte Carlo permutation test.
        
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
            num_permutations (int): Number of permutations to run.
            random_seed (int): Random seed for reproducibility.
        """
        self.strategy_name = strategy_name
        self.tickers = tickers
        self.in_sample_start = in_sample_start
        self.in_sample_end = in_sample_end
        self.out_sample_start = out_sample_start
        self.out_sample_end = out_sample_end
        self.parameters = parameters
        self.load_optimized = load_optimized
        self.optimized_params_path = optimized_params_path
        self.num_permutations = num_permutations
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        self.output_dir = os.path.join(project_root, output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a subdirectory for permutation results
        self.permutations_dir = os.path.join(self.output_dir, 'permutations')
        os.makedirs(self.permutations_dir, exist_ok=True)
        
        # Initialize the walk-forward test
        self.walk_forward_test = WalkForwardTest(
            strategy_name=self.strategy_name,
            in_sample_start=self.in_sample_start,
            in_sample_end=self.in_sample_end,
            out_sample_start=self.out_sample_start,
            out_sample_end=self.out_sample_end,
            tickers=self.tickers,
            output_dir=os.path.join(self.output_dir, 'original'),
            parameters=self.parameters,
            load_optimized=self.load_optimized,
            optimized_params_path=self.optimized_params_path
        )
    
    def _load_stock_data(self):
        """Load stock data from CSV file."""
        print("\nLoading stock data...")
        
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        data_path = os.path.join(project_root, 'input', 'stock_data.csv')
        
        # Load data
        try:
            data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
            print(f"Loaded stock data with {len(data)} rows and {len(data.columns)} columns.")
            return data
        except Exception as e:
            print(f"Error loading stock data: {e}")
            return None
    
    def _create_permuted_data(self, data, permutation_id):
        """Create permuted data by shuffling returns within the out-of-sample period."""
        print(f"\nCreating permuted data for permutation {permutation_id}...")
        
        # Make a copy of the original data
        permuted_data = data.copy()
        
        # Filter for out-of-sample period
        out_sample_mask = (permuted_data.index >= self.out_sample_start) & (permuted_data.index <= self.out_sample_end)
        out_sample_data = permuted_data[out_sample_mask]
        
        # Get unique tickers
        tickers = []
        for col in out_sample_data.columns:
            ticker = col.split('_')[0]
            if ticker not in tickers and ticker != 'SP500':
                tickers.append(ticker)
        
        # Shuffle returns for each ticker
        for ticker in tickers:
            # Get price columns for this ticker
            close_col = f"{ticker}_Close"
            
            if close_col in out_sample_data.columns:
                # Calculate returns
                returns = out_sample_data[close_col].pct_change().dropna()
                
                # Shuffle returns
                shuffled_returns = returns.sample(frac=1, random_state=self.random_seed + permutation_id).values
                
                # Reconstruct prices from shuffled returns
                initial_price = out_sample_data[close_col].iloc[0]
                shuffled_prices = [initial_price]
                
                for ret in shuffled_returns:
                    next_price = shuffled_prices[-1] * (1 + ret)
                    shuffled_prices.append(next_price)
                
                # Replace prices in the permuted data
                permuted_data.loc[out_sample_mask, close_col] = shuffled_prices[:len(out_sample_data)]
        
        # Save permuted data to CSV
        permuted_data_path = os.path.join(self.permutations_dir, f'permuted_data_{permutation_id}.csv')
        permuted_data.to_csv(permuted_data_path)
        print(f"Permuted data saved to {permuted_data_path}")
        
        return permuted_data
    
    def _run_permutation_test(self, permutation_id, permuted_data):
        """Run a backtest on permuted data."""
        print(f"\nRunning permutation test {permutation_id}...")
        
        # Create output directory for this permutation
        permutation_dir = os.path.join(self.permutations_dir, f'permutation_{permutation_id}')
        os.makedirs(permutation_dir, exist_ok=True)
        
        # Save permuted data to a temporary CSV file
        temp_data_path = os.path.join(permutation_dir, 'permuted_stock_data.csv')
        permuted_data.to_csv(temp_data_path)
        
        # Run backtest on permuted data
        try:
            # Use the original in-sample parameters
            results = run_backtest(
                output_dir=permutation_dir,
                strategy_name=self.strategy_name,
                tickers=self.tickers,
                start_date=self.out_sample_start,
                end_date=self.out_sample_end,
                parameters=self.parameters,
                data_path=temp_data_path  # Use permuted data
            )
            
            # Extract performance metrics
            metrics = results.get('performance_metrics', {})
            
            # Save results
            results_path = os.path.join(permutation_dir, 'results.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"Permutation {permutation_id} completed. Results saved to {permutation_dir}")
            
            return metrics
        except Exception as e:
            print(f"Error running permutation {permutation_id}: {e}")
            return {}
    
    def _analyze_permutation_results(self, original_results, permutation_metrics):
        """Analyze permutation results and calculate p-values."""
        print("\nAnalyzing permutation results...")
        
        # Extract original out-of-sample metrics
        original_metrics = original_results.get('out_sample_results', {}).get('performance_metrics', {})
        
        # Metrics to analyze
        metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 
                   'win_rate', 'profit_factor', 'avg_win_loss_ratio']
        
        # Create dataframe to store permutation results
        permutation_df = pd.DataFrame(permutation_metrics)
        
        # Calculate p-values
        p_values = {}
        for metric in metrics:
            original_value = original_metrics.get(metric, 0)
            permutation_values = permutation_df[metric].dropna().values
            
            # For metrics where higher is better
            if metric in ['total_return', 'annualized_return', 'sharpe_ratio', 'win_rate', 
                          'profit_factor', 'avg_win_loss_ratio']:
                p_value = np.mean(permutation_values >= original_value)
            # For metrics where lower is better
            else:
                p_value = np.mean(permutation_values <= original_value)
            
            p_values[metric] = p_value
        
        # Create summary dataframe
        summary = pd.DataFrame({
            'Original': pd.Series(original_metrics),
            'Permutation Mean': permutation_df.mean(),
            'Permutation Std': permutation_df.std(),
            'Permutation Min': permutation_df.min(),
            'Permutation Max': permutation_df.max(),
            'p-value': pd.Series(p_values)
        })
        
        # Save summary to CSV
        summary_path = os.path.join(self.output_dir, 'permutation_summary.csv')
        summary.to_csv(summary_path)
        print(f"Permutation summary saved to {summary_path}")
        
        return summary
    
    def _plot_permutation_distributions(self, original_results, permutation_metrics):
        """Plot distributions of permutation results with original results highlighted."""
        print("\nPlotting permutation distributions...")
        
        # Extract original out-of-sample metrics
        original_metrics = original_results.get('out_sample_results', {}).get('performance_metrics', {})
        
        # Metrics to plot
        metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 
                   'win_rate', 'profit_factor', 'avg_win_loss_ratio']
        
        # Create dataframe to store permutation results
        permutation_df = pd.DataFrame(permutation_metrics)
        
        # Create plots
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            # Plot permutation distribution
            sns.histplot(permutation_df[metric].dropna(), kde=True, color='gray', alpha=0.7)
            
            # Plot original value
            original_value = original_metrics.get(metric, 0)
            plt.axvline(x=original_value, color='red', linestyle='--', 
                        label=f'Original Value: {original_value:.4f}')
            
            # Calculate p-value
            if metric in ['total_return', 'annualized_return', 'sharpe_ratio', 'win_rate', 
                          'profit_factor', 'avg_win_loss_ratio']:
                p_value = np.mean(permutation_df[metric].dropna().values >= original_value)
            else:
                p_value = np.mean(permutation_df[metric].dropna().values <= original_value)
            
            # Add p-value to plot
            plt.text(0.05, 0.95, f'p-value: {p_value:.4f}', transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.title(f'Permutation Distribution of {metric.replace("_", " ").title()}')
            plt.xlabel(metric.replace('_', ' ').title())
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = os.path.join(self.output_dir, f'permutation_distribution_{metric}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Permutation distribution plot for {metric} saved to {plot_path}")
    
    def run_test(self):
        """Run the complete walk-forward Monte Carlo permutation test."""
        print(f"\n{'='*80}\nRunning Walk-Forward Monte Carlo Test for {self.strategy_name}\n{'='*80}")
        
        # Run the original walk-forward test
        print("\nRunning original walk-forward test...")
        original_results = self.walk_forward_test.run_test()
        
        # Load stock data
        stock_data = self._load_stock_data()
        if stock_data is None:
            print("Error: Could not load stock data. Aborting test.")
            return None
        
        # Run permutation tests
        permutation_metrics = []
        for i in tqdm(range(self.num_permutations), desc="Running permutations"):
            # Create permuted data
            permuted_data = self._create_permuted_data(stock_data, i)
            
            # Run permutation test
            metrics = self._run_permutation_test(i, permuted_data)
            
            # Store metrics
            permutation_metrics.append(metrics)
        
        # Analyze permutation results
        summary = self._analyze_permutation_results(original_results, permutation_metrics)
        
        # Plot permutation distributions
        self._plot_permutation_distributions(original_results, permutation_metrics)
        
        # Save all results
        results = {
            'original_results': original_results,
            'permutation_metrics': permutation_metrics,
            'summary': summary,
            'parameters': self.parameters,
            'num_permutations': self.num_permutations,
            'random_seed': self.random_seed,
            'in_sample_period': {
                'start': self.in_sample_start,
                'end': self.in_sample_end
            },
            'out_sample_period': {
                'start': self.out_sample_start,
                'end': self.out_sample_end
            }
        }
        
        results_path = os.path.join(self.output_dir, 'walk_forward_monte_carlo_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nWalk-Forward Monte Carlo Test completed. Results saved to {self.output_dir}")
        
        return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run a walk-forward Monte Carlo permutation test for a trading strategy.')
    
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
    
    parser.add_argument('--output_dir', type=str, default='output/walk_forward_monte_carlo',
                        help='Directory to save test results')
    
    parser.add_argument('--load_optimized', action='store_true',
                        help='Load optimized parameters from a previous run')
    
    parser.add_argument('--optimized_params_path', type=str,
                        help='Path to the optimized parameters file')
    
    parser.add_argument('--num_permutations', type=int, default=100,
                        help='Number of permutations to run')
    
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Create and run the walk-forward Monte Carlo test
    test = WalkForwardMonteCarloTest(
        strategy_name=args.strategy,
        tickers=args.tickers,
        in_sample_start=args.in_sample_start,
        in_sample_end=args.in_sample_end,
        out_sample_start=args.out_sample_start,
        out_sample_end=args.out_sample_end,
        output_dir=args.output_dir,
        load_optimized=args.load_optimized,
        optimized_params_path=args.optimized_params_path,
        num_permutations=args.num_permutations,
        random_seed=args.random_seed
    )
    
    results = test.run_test() 