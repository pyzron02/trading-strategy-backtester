#!/usr/bin/env python3
# in_sample_monte_carlo.py - Compare strategy performance on real vs permuted data

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

class InSampleMonteCarloTest:
    """
    Compare strategy performance on real data to permuted data to check for data mining bias.
    
    This test runs the strategy on the original data and then on multiple permutations of the
    data where the returns are shuffled randomly. If the strategy performs significantly better
    on the original data than on the permuted data, it suggests the strategy is capturing real
    patterns rather than just fitting to noise.
    """
    
    def __init__(self, strategy_name, tickers=None, start_date='2015-01-01', end_date='2019-12-31',
                 output_dir='output/in_sample_monte_carlo', parameters=None, num_permutations=100,
                 random_seed=42):
        """
        Initialize the in-sample Monte Carlo permutation test.
        
        Args:
            strategy_name (str): Name of the strategy to test.
            tickers (list): List of stock ticker symbols. If None, will use all tickers in stock_data.csv.
            start_date (str): Start date for the in-sample period in 'YYYY-MM-DD' format.
            end_date (str): End date for the in-sample period in 'YYYY-MM-DD' format.
            output_dir (str): Directory to save test results.
            parameters (dict): Strategy parameters to use. If None, will use default parameters.
            num_permutations (int): Number of permutations to run.
            random_seed (int): Random seed for reproducibility.
        """
        self.strategy_name = strategy_name
        self.tickers = tickers if tickers is not None else ['SPY']
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.parameters = parameters
        self.num_permutations = num_permutations
        self.random_seed = random_seed
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # Results storage
        self.original_results = None
        self.permutation_results = []
        
        # Load stock data
        self.stock_data = self._load_stock_data()
    
    def _load_stock_data(self):
        """
        Load stock data from CSV file.
        
        Returns:
            pd.DataFrame: Stock data.
        """
        # Try different possible locations for the stock data file
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'input', 'stock_data.csv'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'input', 'stock_data.csv'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'input', 'stock_data.csv'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input', 'stock_data.csv'),
            os.path.join('input', 'stock_data.csv')
        ]
        
        # Check if any of the paths exist
        stock_csv = None
        for path in possible_paths:
            if os.path.exists(path):
                stock_csv = path
                break
        
        # If no existing file is found, create a dummy data file
        if stock_csv is None:
            print("Stock data file not found. Creating a dummy data file...")
            stock_csv = self._create_dummy_data()
        
        # Load the data
        print(f"Loading stock data from {stock_csv}")
        data = pd.read_csv(stock_csv)
        
        return data
    
    def _create_dummy_data(self):
        """
        Create a dummy stock data file for testing.
        
        Returns:
            str: Path to the created file.
        """
        # Create the input directory if it doesn't exist
        input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
        os.makedirs(input_dir, exist_ok=True)
        
        # Path to the dummy data file
        dummy_file = os.path.join(input_dir, 'stock_data.csv')
        
        # Create a date range
        start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.end_date, '%Y-%m-%d')
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Create a DataFrame with dummy data
        data = pd.DataFrame({
            'Date': date_range.strftime('%Y-%m-%d')
        })
        
        # Add data for each ticker
        for ticker in self.tickers:
            # Generate random price data
            np.random.seed(self.random_seed)
            price = 100.0
            prices = []
            for _ in range(len(date_range)):
                price = price * (1 + np.random.normal(0.0005, 0.02))
                prices.append(price)
            
            # Add columns for this ticker
            data[f'{ticker}_Open'] = prices
            data[f'{ticker}_High'] = [p * (1 + np.random.uniform(0, 0.02)) for p in prices]
            data[f'{ticker}_Low'] = [p * (1 - np.random.uniform(0, 0.02)) for p in prices]
            data[f'{ticker}_Close'] = [p * (1 + np.random.normal(0, 0.005)) for p in prices]
            data[f'{ticker}_Volume'] = [int(np.random.uniform(1000000, 10000000)) for _ in prices]
        
        # Add SP500 data
        data['SP500_Open'] = [p * 0.1 for p in prices]
        data['SP500_High'] = [p * 0.1 * (1 + np.random.uniform(0, 0.01)) for p in prices]
        data['SP500_Low'] = [p * 0.1 * (1 - np.random.uniform(0, 0.01)) for p in prices]
        data['SP500_Close'] = [p * 0.1 * (1 + np.random.normal(0, 0.003)) for p in prices]
        data['SP500_Volume'] = [int(np.random.uniform(10000000, 100000000)) for _ in prices]
        
        # Save the data to CSV
        data.to_csv(dummy_file, index=False)
        print(f"Dummy stock data created at {dummy_file}")
        
        return dummy_file
    
    def _create_permuted_data(self, permutation_id):
        """
        Create permuted data for Monte Carlo test.
        
        Args:
            permutation_id (int): ID of the permutation.
            
        Returns:
            str: Path to the permuted data directory.
        """
        # Create a directory for this permutation
        permuted_dir = os.path.join(self.output_dir, f'permuted_data_{permutation_id}')
        os.makedirs(permuted_dir, exist_ok=True)
        
        # Load the original data if not already loaded
        if not hasattr(self, 'stock_data'):
            self.stock_data = self._load_stock_data()
        
        # Create a copy of the data
        permuted_data = self.stock_data.copy()
        
        # Detect tickers in the data
        tickers = self._detect_tickers(permuted_data)
        
        # Permute the data for each ticker
        for ticker in tickers:
            # Get the columns for this ticker
            ticker_cols = [col for col in permuted_data.columns if col.startswith(f'{ticker}_')]
            
            # Skip if no columns found
            if not ticker_cols:
                continue
            
            # Get the indices to permute (excluding the first few rows to maintain some structure)
            n_rows = len(permuted_data)
            permute_indices = np.random.permutation(np.arange(5, n_rows))
            
            # Permute the data for this ticker
            for col in ticker_cols:
                # Extract the values to permute
                values = permuted_data[col].values.copy()
                to_permute = values[5:]
                
                # Permute the values
                np.random.shuffle(to_permute)
                
                # Put the permuted values back
                values[5:] = to_permute
                
                # Update the DataFrame
                permuted_data[col] = values
        
        # Save the permuted data
        permuted_file = os.path.join(permuted_dir, f'permuted_data_{permutation_id}.csv')
        permuted_data.to_csv(permuted_file, index=False)
        
        return permuted_dir
    
    def _detect_tickers(self, data):
        """
        Detect tickers in the data.
        
        Args:
            data (pd.DataFrame): Stock data.
            
        Returns:
            list: List of detected tickers.
        """
        # Extract unique ticker symbols from column names (format: TICKER_Field)
        all_tickers = set()
        ticker_pattern = r'([A-Z]+)_(?:Open|High|Low|Close|Volume)'
        
        for col in data.columns:
            import re
            match = re.match(ticker_pattern, col)
            if match:
                ticker = match.group(1)
                if ticker != 'SP500':  # Exclude SP500
                    all_tickers.add(ticker)
        
        return sorted(list(all_tickers))
    
    def _calculate_metrics(self, results_path):
        """
        Calculate performance metrics from backtest results.
        
        Args:
            results_path (str): Path to the backtest results directory.
            
        Returns:
            dict: Dictionary of performance metrics.
        """
        # Load equity curve
        equity_curve_path = os.path.join(results_path, 'equity_curve.csv')
        if not os.path.exists(equity_curve_path):
            return {}
        
        equity_curve = pd.read_csv(equity_curve_path)
        equity_curve['Date'] = pd.to_datetime(equity_curve['Date'])
        equity_curve.set_index('Date', inplace=True)
        
        # Load trade log
        trade_log_path = os.path.join(results_path, 'trade_log.csv')
        if not os.path.exists(trade_log_path):
            return {}
        
        trade_log = pd.read_csv(trade_log_path)
        
        # Calculate metrics
        metrics = {}
        
        # Calculate returns
        equity_curve['Return'] = equity_curve['Value'].pct_change()
        
        # Total return
        initial_value = equity_curve['Value'].iloc[0]
        final_value = equity_curve['Value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        metrics['total_return'] = total_return
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        metrics['annualized_return'] = annualized_return
        
        # Volatility
        daily_volatility = equity_curve['Return'].std()
        annualized_volatility = daily_volatility * (252 ** 0.5)
        metrics['volatility'] = annualized_volatility
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        metrics['sharpe_ratio'] = sharpe_ratio
        
        # Maximum drawdown
        equity_curve['Peak'] = equity_curve['Value'].cummax()
        equity_curve['Drawdown'] = (equity_curve['Value'] - equity_curve['Peak']) / equity_curve['Peak']
        max_drawdown = abs(equity_curve['Drawdown'].min())
        metrics['max_drawdown'] = max_drawdown
        
        # Calculate trade metrics
        if not trade_log.empty:
            # Filter to closed trades
            closed_trades = trade_log[trade_log['type'] == 'close']
            
            # Win rate
            winning_trades = closed_trades[closed_trades['pnl'] > 0]
            win_rate = len(winning_trades) / len(closed_trades) if len(closed_trades) > 0 else 0
            metrics['win_rate'] = win_rate
            
            # Profit factor
            gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
            losing_trades = closed_trades[closed_trades['pnl'] <= 0]
            gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            metrics['profit_factor'] = profit_factor
            
            # Average win/loss ratio
            avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
            avg_loss = abs(losing_trades['pnl'].mean()) if not losing_trades.empty else 0
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
            metrics['win_loss_ratio'] = win_loss_ratio
            
            # Number of trades
            metrics['num_trades'] = len(closed_trades)
        
        return metrics
    
    def run_test(self, num_permutations=None):
        """
        Run the Monte Carlo permutation test.
        
        Args:
            num_permutations (int): Number of permutations to run. If None, use the value from __init__.
            
        Returns:
            dict: Results of the test including p-values for various metrics.
        """
        if num_permutations is None:
            num_permutations = self.num_permutations
            
        print(f"Running Monte Carlo permutation test with {num_permutations} permutations")
        
        # Step 1: Run the strategy on the original data
        print("Running strategy on original data...")
        original_results_dir = self._run_original_backtest()
        
        # Step 2: Calculate performance metrics for the original data
        original_metrics = self._calculate_metrics(original_results_dir)
        print(f"Original metrics: {original_metrics}")
        
        # Step 3: Run the strategy on permuted data
        print(f"Running strategy on {num_permutations} permutations of the data...")
        permutation_metrics = []
        
        for i in tqdm(range(num_permutations), desc="Running permutations"):
            # Generate permuted data
            permuted_data_dir = self._create_permuted_data(i)
            
            # Run backtest on permuted data
            permutation_results_dir = self._run_permutation_test(i, permuted_data_dir)
            
            # Calculate metrics for this permutation
            metrics = self._calculate_metrics(permutation_results_dir)
            permutation_metrics.append(metrics)
        
        # Step 4: Calculate p-values
        p_values = self._calculate_p_values(original_metrics, permutation_metrics)
        
        # Step 5: Plot the results
        self._plot_results(original_metrics, permutation_metrics)
        
        # Save results
        results = {
            'original_metrics': original_metrics,
            'permutation_metrics': permutation_metrics,
            'p_values': p_values
        }
        
        # Extract key metrics for easier access
        results.update({
            'p_value_sharpe': p_values.get('sharpe_ratio', 1.0),
            'p_value_returns': p_values.get('total_return', 1.0),
            'p_value_profit_factor': p_values.get('profit_factor', 1.0)
        })
        
        # Save results to file
        results_file = os.path.join(self.output_dir, 'monte_carlo_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Print summary
        print("\nMonte Carlo Test Results:")
        print(f"P-value (Sharpe Ratio): {p_values.get('sharpe_ratio', 1.0):.4f}")
        print(f"P-value (Total Return): {p_values.get('total_return', 1.0):.4f}")
        print(f"P-value (Profit Factor): {p_values.get('profit_factor', 1.0):.4f}")
        
        # Interpret results
        alpha = 0.05  # Significance level
        if min(p_values.values()) < alpha:
            print("\nInterpretation: The strategy's performance is statistically significant.")
            print("This suggests the strategy is capturing real patterns in the data rather than just fitting to noise.")
        else:
            print("\nInterpretation: The strategy's performance is NOT statistically significant.")
            print("This suggests the strategy may be fitting to noise rather than capturing real patterns in the data.")
        
        return results
    
    def _calculate_p_values(self, original_metrics, permutation_metrics):
        """
        Calculate p-values for each metric by comparing original results to permutation results.
        
        Args:
            original_metrics (dict): Metrics for the original data.
            permutation_metrics (list): List of metrics for each permutation.
            
        Returns:
            dict: Dictionary of p-values for each metric.
        """
        p_values = {}
        
        # For each metric, calculate p-value
        for metric in original_metrics:
            # Get original value
            original_value = original_metrics[metric]
            
            # Get permutation values
            permutation_values = [result[metric] for result in permutation_metrics]
            
            # Calculate p-value (proportion of permutations with better performance)
            if metric in ['total_return', 'annualized_return', 'sharpe_ratio', 'win_rate', 'profit_factor', 'win_loss_ratio']:
                # Higher is better
                p_value = sum(1 for val in permutation_values if val >= original_value) / len(permutation_values)
            elif metric in ['volatility', 'max_drawdown']:
                # Lower is better
                p_value = sum(1 for val in permutation_values if val <= original_value) / len(permutation_values)
            else:
                # Default: higher is better
                p_value = sum(1 for val in permutation_values if val >= original_value) / len(permutation_values)
            
            p_values[metric] = p_value
        
        return p_values
    
    def _plot_results(self, original_metrics, permutation_metrics):
        """
        Plot the results of the Monte Carlo permutation test.
        
        Args:
            original_metrics (dict): Metrics for the original data.
            permutation_metrics (list): List of metrics for each permutation.
        """
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # For each metric, create a histogram of permutation values with original value marked
        for metric in original_metrics:
            # Get original value
            original_value = original_metrics[metric]
            
            # Get permutation values
            permutation_values = [result[metric] for result in permutation_metrics]
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot histogram of permutation values
            sns.histplot(permutation_values, kde=True)
            
            # Plot vertical line for original value
            plt.axvline(original_value, color='red', linestyle='--', linewidth=2, 
                        label=f'Original Value: {original_value:.4f}')
            
            # Calculate p-value
            if metric in ['total_return', 'annualized_return', 'sharpe_ratio', 'win_rate', 'profit_factor', 'win_loss_ratio']:
                # Higher is better
                p_value = sum(1 for val in permutation_values if val >= original_value) / len(permutation_values)
                better_text = "better than"
            elif metric in ['volatility', 'max_drawdown']:
                # Lower is better
                p_value = sum(1 for val in permutation_values if val <= original_value) / len(permutation_values)
                better_text = "better than"
            else:
                # Default: higher is better
                p_value = sum(1 for val in permutation_values if val >= original_value) / len(permutation_values)
                better_text = "better than"
            
            # Add p-value to plot
            plt.title(f'Distribution of {metric} in Permutation Tests\n'
                      f'p-value: {p_value:.4f} ({p_value*100:.1f}% of permutations {better_text} original)')
            
            plt.xlabel(metric)
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save figure
            plt.savefig(os.path.join(plots_dir, f"{metric}_distribution.png"))
            plt.close()
        
        # Create a summary plot of p-values
        metrics = list(original_metrics.keys())
        p_values = [self._calculate_p_values(original_metrics, permutation_metrics)[metric] for metric in metrics]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metrics, p_values)
        
        # Color bars based on significance
        for i, bar in enumerate(bars):
            if p_values[i] <= 0.05:
                bar.set_color('green')
            elif p_values[i] <= 0.1:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        plt.axhline(0.05, color='black', linestyle='--', alpha=0.7, label='p=0.05')
        plt.title('p-values for Each Metric')
        plt.xlabel('Metric')
        plt.ylabel('p-value')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(plots_dir, "p_values_summary.png"))
        plt.close()

    def _run_original_backtest(self):
        """
        Run the strategy on the original data.
        
        Returns:
            str: Path to the results directory.
        """
        # Create a directory for the original backtest
        original_dir = os.path.join(self.output_dir, 'original')
        os.makedirs(original_dir, exist_ok=True)
        
        # Run the backtest
        results_dir = run_backtest(
            strategy_name=self.strategy_name,
            tickers=self.tickers,
            parameters=self.parameters,
            output_dir=original_dir
        )
        
        return results_dir
    
    def _run_permutation_test(self, permutation_id, permuted_data_dir):
        """
        Run the strategy on permuted data.
        
        Args:
            permutation_id (int): ID of the permutation.
            permuted_data_dir (str): Directory containing the permuted data.
            
        Returns:
            str: Path to the results directory.
        """
        # Create a directory for this permutation
        permutation_dir = os.path.join(self.output_dir, f'permutation_{permutation_id}')
        os.makedirs(permutation_dir, exist_ok=True)
        
        # Run the backtest
        results_dir = run_backtest(
            strategy_name=self.strategy_name,
            tickers=self.tickers,
            parameters=self.parameters,
            output_dir=permutation_dir
        )
        
        return results_dir

def main():
    parser = argparse.ArgumentParser(description="Run Monte Carlo permutation test to check for data mining bias.")
    parser.add_argument('--strategy_name', type=str, default='SimpleStock',
                        help="Name of the strategy to test (e.g., SimpleStock, MultiPosition, AuctionMarket)")
    parser.add_argument('--tickers', type=str, default=None,
                        help="Comma-separated list of ticker symbols (e.g., MSFT,AAPL)")
    parser.add_argument('--start_date', type=str, default='2015-01-01',
                        help="Start date for in-sample period (YYYY-MM-DD)")
    parser.add_argument('--end_date', type=str, default='2019-12-31',
                        help="End date for in-sample period (YYYY-MM-DD)")
    parser.add_argument('--num_permutations', type=int, default=100,
                        help="Number of permutations to run")
    parser.add_argument('--parameters_file', type=str, default=None,
                        help="Path to pickle file with strategy parameters")
    
    args = parser.parse_args()
    
    # Parse tickers if provided
    tickers = None
    if args.tickers:
        tickers = [ticker.strip() for ticker in args.tickers.split(',')]
    
    # Load parameters if provided
    parameters = None
    if args.parameters_file and os.path.exists(args.parameters_file):
        with open(args.parameters_file, 'rb') as f:
            best_result = pickle.load(f)
            parameters = best_result.get('parameters', None)
    
    # Run test
    tester = InSampleMonteCarloTest(
        strategy_name=args.strategy_name,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        parameters=parameters
    )
    
    tester.run_test(num_permutations=args.num_permutations)

if __name__ == "__main__":
    main() 