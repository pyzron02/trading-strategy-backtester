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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re

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
                 optimized_params_path=None, plot=False):
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
            plot (bool): Whether to generate plots during backtests. Default is False.
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
        self.plot = plot
        
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
        
        # Calculate warmup period based on strategy
        warmup_period = 60  # Default warmup period
        if self.strategy_name == 'MACrossover':
            if self.parameters and 'slow_period' in self.parameters:
                warmup_period = self.parameters['slow_period'] * 2
            else:
                warmup_period = 60  # Default - twice the default slow period (30)
        
        # Run backtest with in-sample date range
        try:
            in_sample_results = run_backtest(
                output_dir=in_sample_dir,
                strategy_name=self.strategy_name,
                tickers=self.tickers,
                parameters=self.parameters,
                start_date=self.in_sample_start.strftime('%Y-%m-%d'),
                end_date=self.in_sample_end.strftime('%Y-%m-%d'),
                plot=self.plot
            )
            if in_sample_results is None:
                print(f"Warning: In-sample backtest returned None. Using empty results.")
                in_sample_results = {"status": "error", "message": "Backtest failed to return results"}
        except Exception as e:
            print(f"Error in in-sample backtest: {e}")
            in_sample_results = {"status": "error", "message": f"Exception: {str(e)}"}
        
        return in_sample_results
    
    def run_out_sample_backtest(self):
        """Run backtest on out-of-sample data."""
        print(f"\nRunning out-of-sample backtest ({self.out_sample_start.strftime('%Y-%m-%d')} to {self.out_sample_end.strftime('%Y-%m-%d')})...")
        
        # Create out-of-sample output directory
        out_sample_dir = os.path.join(self.output_dir, 'out_sample')
        os.makedirs(out_sample_dir, exist_ok=True)
        
        # Calculate warmup period based on strategy
        warmup_period = 60  # Default warmup period
        if self.strategy_name == 'MACrossover':
            if self.parameters and 'slow_period' in self.parameters:
                warmup_period = self.parameters['slow_period'] * 2
            else:
                warmup_period = 60  # Default - twice the default slow period (30)
        
        # Run backtest with out-of-sample date range
        try:
            out_sample_results = run_backtest(
                output_dir=out_sample_dir,
                strategy_name=self.strategy_name,
                tickers=self.tickers,
                parameters=self.parameters,
                start_date=self.out_sample_start.strftime('%Y-%m-%d'),
                end_date=self.out_sample_end.strftime('%Y-%m-%d'),
                plot=self.plot
            )
            if out_sample_results is None:
                print(f"Warning: Out-of-sample backtest returned None. Using empty results.")
                out_sample_results = {"status": "error", "message": "Backtest failed to return results"}
        except Exception as e:
            print(f"Error in out-of-sample backtest: {e}")
            out_sample_results = {"status": "error", "message": f"Exception: {str(e)}"}
        
        return out_sample_results
    
    def compare_performance(self, in_sample_results, out_sample_results):
        """Compare in-sample and out-of-sample performance."""
        print("\nComparing in-sample and out-of-sample performance...")
        
        # Define metrics to compare
        metrics = ['total_return', 'benchmark_return', 'alpha']
        
        # Create comparison dataframe
        comparison = pd.DataFrame(index=metrics, columns=['In-Sample', 'Out-of-Sample', 'Difference', 'Degradation %'])
        
        # Extract metrics from results.txt files instead of using result objects
        in_sample_metrics = self._extract_metrics_from_file(os.path.join(self.output_dir, 'in_sample', 'results.txt'))
        out_sample_metrics = self._extract_metrics_from_file(os.path.join(self.output_dir, 'out_sample', 'results.txt'))
        
        for metric in metrics:
            # Get values from parsed metrics, default to 0 if not found
            in_val = in_sample_metrics.get(metric, 0)
            out_val = out_sample_metrics.get(metric, 0)
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
        
    def _extract_metrics_from_file(self, results_file_path):
        """Extract performance metrics from a results.txt file.
        
        Args:
            results_file_path (str): Path to the results.txt file
            
        Returns:
            dict: Dictionary containing extracted metrics
        """
        metrics = {}
        
        if not os.path.exists(results_file_path):
            print(f"Warning: Results file not found at {results_file_path}")
            return metrics
            
        try:
            with open(results_file_path, 'r') as f:
                content = f.read()
            
            print(f"Extracting metrics from: {results_file_path}")
                
            # Extract total return from the file (format: "Total Return: X.XX%")
            total_return_match = re.search(r'Total Return:\s*([-\d.]+)%', content)
            if total_return_match:
                metrics['total_return'] = float(total_return_match.group(1)) / 100  # Convert percentage to decimal
                print(f"  Found total_return: {metrics['total_return']}")
                
            # Extract benchmark return (format: "Benchmark Return: X.XX%")
            benchmark_match = re.search(r'Benchmark Return:\s*([-\d.]+)%', content)
            if benchmark_match:
                metrics['benchmark_return'] = float(benchmark_match.group(1)) / 100  # Convert percentage to decimal
                print(f"  Found benchmark_return: {metrics['benchmark_return']}")
                
            # Extract alpha (format: "Alpha: X.XX%")
            alpha_match = re.search(r'Alpha:\s*([-\d.]+)%', content)
            if alpha_match:
                metrics['alpha'] = float(alpha_match.group(1)) / 100  # Convert percentage to decimal
                print(f"  Found alpha: {metrics['alpha']}")
                
            # Add other metrics as needed
            sharpe_match = re.search(r'Sharpe Ratio:\s*([-\d.]+)', content)
            if sharpe_match:
                metrics['sharpe_ratio'] = float(sharpe_match.group(1))
                print(f"  Found sharpe_ratio: {metrics['sharpe_ratio']}")
                
            max_dd_match = re.search(r'Maximum Drawdown:\s*([-\d.]+)%', content)
            if max_dd_match:
                metrics['max_drawdown'] = float(max_dd_match.group(1)) / 100  # Convert percentage to decimal
                print(f"  Found max_drawdown: {metrics['max_drawdown']}")
                
            profit_factor_match = re.search(r'Profit Factor:\s*([-\d.]+)', content)
            if profit_factor_match:
                metrics['profit_factor'] = float(profit_factor_match.group(1))
                print(f"  Found profit_factor: {metrics['profit_factor']}")
            
            print(f"Extracted {len(metrics)} metrics: {list(metrics.keys())}")
                
        except Exception as e:
            print(f"Error extracting metrics from {results_file_path}: {e}")
            import traceback
            traceback.print_exc()
            
        return metrics
    
    def plot_equity_curves(self, in_sample_results, out_sample_results):
        """Plot in-sample and out-of-sample equity curves."""
        print("\nPlotting equity curves...")
        
        # Extract equity curves
        in_sample_equity = in_sample_results.get('equity_curve', [])
        out_sample_equity = out_sample_results.get('equity_curve', [])
        
        # Check if equity curves are empty using appropriate pandas methods
        # or convert to appropriate containers if they're plain Python lists
        if (isinstance(in_sample_equity, pd.DataFrame) and in_sample_equity.empty) or \
           (isinstance(out_sample_equity, pd.DataFrame) and out_sample_equity.empty) or \
           (not isinstance(in_sample_equity, pd.DataFrame) and not in_sample_equity) or \
           (not isinstance(out_sample_equity, pd.DataFrame) and not out_sample_equity):
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
        
        # Create interactive plot with Plotly
        fig = go.Figure()
        
        # Add in-sample equity curve
        fig.add_trace(go.Scatter(
            x=in_sample_equity_norm.index,
            y=in_sample_equity_norm.values,
            mode='lines',
            name='In-Sample',
            line=dict(color='blue')
        ))
        
        # Add out-of-sample equity curve
        fig.add_trace(go.Scatter(
            x=out_sample_equity_norm.index,
            y=out_sample_equity_norm.values,
            mode='lines',
            name='Out-of-Sample',
            line=dict(color='red')
        ))
        
        # Add vertical line to separate in-sample and out-of-sample periods
        # Use shape instead of add_vline to avoid type errors
        split_date = self.in_sample_end.strftime('%Y-%m-%d')
        fig.add_shape(
            type="line",
            x0=split_date, x1=split_date,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="black", width=2, dash="dash")
        )
        
        # Add annotation for the split
        fig.add_annotation(
            x=split_date,
            y=1,
            yref="paper",
            text="Train/Test Split",
            showarrow=False,
            xanchor="right"
        )
        
        # Update layout with labels and title
        fig.update_layout(
            title=f"{self.strategy_name} Equity Curve",
            xaxis_title="Date",
            yaxis_title="Normalized Equity (Starting at 100)",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            annotations=[
                dict(
                    x=0.15,
                    y=0.05,
                    xref="paper",
                    yref="paper",
                    text=f"In-Sample Return: {in_sample_results.get('total_return', 0):.2f}%",
                    showarrow=False
                ),
                dict(
                    x=0.85,
                    y=0.05,
                    xref="paper",
                    yref="paper",
                    text=f"Out-of-Sample Return: {out_sample_results.get('total_return', 0):.2f}%",
                    showarrow=False
                )
            ],
            hovermode="x unified",
            template="plotly_white",  # Clean white background with grid
        )
        
        # Add grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        
        # Save plot as HTML file
        plot_file = os.path.join(self.output_dir, f"{self.strategy_name}_equity_curves.html")
        fig.write_html(plot_file, include_plotlyjs='cdn')
        
        print(f"Interactive equity curves saved to {plot_file}")
        return plot_file
    
    def plot_drawdowns(self, in_sample_results, out_sample_results):
        """Plot in-sample and out-of-sample drawdowns."""
        print("\nPlotting drawdowns comparison...")
        
        # Extract drawdowns
        in_sample_drawdowns = in_sample_results.get('drawdowns', None)
        out_sample_drawdowns = out_sample_results.get('drawdowns', None)
        
        # Check if data exists and is suitable
        if (in_sample_drawdowns is None or out_sample_drawdowns is None or
            (isinstance(in_sample_drawdowns, pd.DataFrame) and in_sample_drawdowns.empty) or
            (isinstance(out_sample_drawdowns, pd.DataFrame) and out_sample_drawdowns.empty)):
            print("Error: Drawdown data is missing or empty.")
            return None
        
        # Ensure we have pandas DataFrames
        if not isinstance(in_sample_drawdowns, pd.DataFrame):
            try:
                in_sample_drawdowns = pd.DataFrame(in_sample_drawdowns)
            except Exception as e:
                print(f"Error converting in-sample drawdowns to DataFrame: {e}")
                return None
        
        if not isinstance(out_sample_drawdowns, pd.DataFrame):
            try:
                out_sample_drawdowns = pd.DataFrame(out_sample_drawdowns)
            except Exception as e:
                print(f"Error converting out-of-sample drawdowns to DataFrame: {e}")
                return None
        
        # Create interactive plot
        fig = make_subplots(rows=2, cols=1,
                          shared_xaxes=False,
                          vertical_spacing=0.1,
                          subplot_titles=('In-Sample Drawdowns', 'Out-of-Sample Drawdowns'))
        
        # Format dates
        if not isinstance(in_sample_drawdowns.index, pd.DatetimeIndex):
            try:
                in_sample_drawdowns.index = pd.to_datetime(in_sample_drawdowns.index)
            except Exception as e:
                print(f"Error converting in-sample dates: {e}")
        
        if not isinstance(out_sample_drawdowns.index, pd.DatetimeIndex):
            try:
                out_sample_drawdowns.index = pd.to_datetime(out_sample_drawdowns.index)
            except Exception as e:
                print(f"Error converting out-of-sample dates: {e}")
            
            # Add in-sample drawdowns
        fig.add_trace(
            go.Scatter(
                x=in_sample_drawdowns.index,
                y=in_sample_drawdowns.iloc[:, 0].values * -1,  # Convert to negative values for visualization
                mode='lines',
                name='In-Sample Drawdowns',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
            
            # Add out-of-sample drawdowns
        fig.add_trace(
            go.Scatter(
                x=out_sample_drawdowns.index,
                y=out_sample_drawdowns.iloc[:, 0].values * -1,  # Convert to negative values for visualization
                mode='lines',
                name='Out-of-Sample Drawdowns',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"{self.strategy_name} Drawdowns Comparison",
            height=800,
            width=1000,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update y-axes
        fig.update_yaxes(title_text='Drawdown (%)', row=1, col=1)
        fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
        
        # Update x-axes
        fig.update_xaxes(title_text='Date', row=1, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=1)
        
        # Save plot
        plot_file = os.path.join(self.output_dir, f"drawdowns_comparison.html")
        fig.write_html(plot_file)
        print(f"Interactive drawdowns comparison saved to {plot_file}")
        
        return plot_file
    
    def plot_monthly_returns(self, in_sample_results, out_sample_results):
        """Plot in-sample and out-of-sample monthly returns."""
        print("\nPlotting monthly returns comparison...")
        
        # Extract monthly returns
        in_sample_monthly = in_sample_results.get('monthly_returns', None)
        out_sample_monthly = out_sample_results.get('monthly_returns', None)
        
        # Check if data exists
        if (in_sample_monthly is None or out_sample_monthly is None or
            (isinstance(in_sample_monthly, pd.DataFrame) and in_sample_monthly.empty) or
            (isinstance(out_sample_monthly, pd.DataFrame) and out_sample_monthly.empty)):
            print("Error: Monthly returns data is missing or empty.")
            return None
        
        # Ensure we have pandas DataFrames
        if not isinstance(in_sample_monthly, pd.DataFrame):
            try:
                in_sample_monthly = pd.DataFrame(in_sample_monthly)
            except Exception as e:
                print(f"Error converting in-sample monthly returns to DataFrame: {e}")
                return None
        
        if not isinstance(out_sample_monthly, pd.DataFrame):
            try:
                out_sample_monthly = pd.DataFrame(out_sample_monthly)
            except Exception as e:
                print(f"Error converting out-of-sample monthly returns to DataFrame: {e}")
                return None
        
        # Create interactive plot
        fig = make_subplots(rows=2, cols=1,
                          shared_xaxes=False,
                          vertical_spacing=0.1,
                          subplot_titles=('In-Sample Monthly Returns', 'Out-of-Sample Monthly Returns'))
        
        # Format data
        if not isinstance(in_sample_monthly.index, pd.DatetimeIndex):
            try:
                in_sample_monthly.index = pd.to_datetime(in_sample_monthly.index)
            except Exception as e:
                print(f"Error converting in-sample dates: {e}")
        
        if not isinstance(out_sample_monthly.index, pd.DatetimeIndex):
            try:
                out_sample_monthly.index = pd.to_datetime(out_sample_monthly.index)
            except Exception as e:
                print(f"Error converting out-of-sample dates: {e}")
        
        # Add in-sample monthly returns
        fig.add_trace(
            go.Bar(
                x=in_sample_monthly.index,
                y=in_sample_monthly.iloc[:, 0].values * 100,  # Convert to percentage
                name='In-Sample Monthly Returns',
                marker_color=in_sample_monthly.iloc[:, 0].values.astype(float) > 0,
                marker=dict(
                    color=['green' if x > 0 else 'red' for x in in_sample_monthly.iloc[:, 0].values]
                )
            ),
            row=1, col=1
        )
        
        # Add out-of-sample monthly returns
        fig.add_trace(
            go.Bar(
                x=out_sample_monthly.index,
                y=out_sample_monthly.iloc[:, 0].values * 100,  # Convert to percentage
                name='Out-of-Sample Monthly Returns',
                marker_color=out_sample_monthly.iloc[:, 0].values.astype(float) > 0,
                marker=dict(
                    color=['green' if x > 0 else 'red' for x in out_sample_monthly.iloc[:, 0].values]
                )
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"{self.strategy_name} Monthly Returns Comparison",
            height=800,
            width=1000,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update y-axes
        fig.update_yaxes(title_text='Return (%)', row=1, col=1)
        fig.update_yaxes(title_text='Return (%)', row=2, col=1)
        
        # Update x-axes
        fig.update_xaxes(title_text='Month', row=1, col=1)
        fig.update_xaxes(title_text='Month', row=2, col=1)
        
        # Save plot
        plot_file = os.path.join(self.output_dir, f"monthly_returns_comparison.html")
        fig.write_html(plot_file)
        print(f"Interactive monthly returns comparison saved to {plot_file}")
        
        return plot_file
    
    def run_test(self):
        """Run the walk forward test."""
        print(f"Running walk forward test for {self.strategy_name}...")
        
        # Create summary file
        with open(os.path.join(self.output_dir, 'walkforward_summary.txt'), 'w') as f:
            f.write(f"Walk Forward Analysis: {self.strategy_name}\n")
            f.write(f"===================================\n\n")
            f.write(f"In-Sample Period: {self.in_sample_start.strftime('%Y-%m-%d')} to {self.in_sample_end.strftime('%Y-%m-%d')}\n")
            f.write(f"Out-of-Sample Period: {self.out_sample_start.strftime('%Y-%m-%d')} to {self.out_sample_end.strftime('%Y-%m-%d')}\n")
            f.write(f"Tickers: {self.tickers}\n")
            f.write(f"Parameters: {self.parameters or 'Default parameters'}\n\n")
        
        # Run in-sample backtest
        in_sample_results = self.run_in_sample_backtest()
        
        # Run out-of-sample backtest
        out_sample_results = self.run_out_sample_backtest()
        
        # Compare performance - ensures result files exist first
        comparison = self.compare_performance(in_sample_results, out_sample_results)
        
        # Visualize results
        try:
            # Plot equity curves
            equity_curves_file = self.plot_equity_curves(in_sample_results, out_sample_results)
            
            # Plot drawdowns comparison
            drawdowns_file = self.plot_drawdowns(in_sample_results, out_sample_results)
            
            # Plot monthly returns comparison
            monthly_returns_file = self.plot_monthly_returns(in_sample_results, out_sample_results)
            
            # Update summary file with visualization paths
            with open(os.path.join(self.output_dir, 'walkforward_summary.txt'), 'a') as f:
                f.write("\nVisualizations:\n")
                if equity_curves_file:
                    f.write(f"Equity Curves: {os.path.basename(equity_curves_file)}\n")
                if drawdowns_file:
                    f.write(f"Drawdowns Comparison: {os.path.basename(drawdowns_file)}\n")
                if monthly_returns_file:
                    f.write(f"Monthly Returns Comparison: {os.path.basename(monthly_returns_file)}\n")
        
        except Exception as e:
            print(f"Error visualizing results: {e}")
            import traceback
            traceback.print_exc()
        
        # Extract key metrics from both results files for summary
        in_sample_metrics = self._extract_metrics_from_file(os.path.join(self.output_dir, 'in_sample', 'results.txt'))
        out_sample_metrics = self._extract_metrics_from_file(os.path.join(self.output_dir, 'out_sample', 'results.txt'))
        
        # Append performance metrics to summary file
        with open(os.path.join(self.output_dir, 'walkforward_summary.txt'), 'a') as f:
            f.write("\nPerformance Metrics:\n")
            f.write("-----------------\n")
            
            metrics_to_include = [
                ('total_return', 'Total Return'),
                ('benchmark_return', 'Benchmark Return'),
                ('alpha', 'Alpha'),
                ('sharpe_ratio', 'Sharpe Ratio'),
                ('max_drawdown', 'Maximum Drawdown'),
                ('profit_factor', 'Profit Factor')
            ]
            
            # Calculate maximum length for nice formatting
            max_len = max(len(name) for _, name in metrics_to_include) + 2
            
            for key, name in metrics_to_include:
                in_value = in_sample_metrics.get(key, 'N/A')
                out_value = out_sample_metrics.get(key, 'N/A')
                
                # Format percentages correctly
                if key in ['total_return', 'benchmark_return', 'alpha', 'max_drawdown'] and isinstance(in_value, float):
                    in_value = f"{in_value*100:.2f}%"
                    out_value = f"{out_value*100:.2f}%"
                elif isinstance(in_value, float):
                    in_value = f"{in_value:.4f}"
                    out_value = f"{out_value:.4f}"
                
                f.write(f"{name + ':':<{max_len}} In-Sample: {in_value:<10} Out-of-Sample: {out_value:<10}")
                
                # Add degradation info for numeric metrics
                if key in ['total_return', 'benchmark_return', 'alpha', 'sharpe_ratio']:
                    if isinstance(in_sample_metrics.get(key), (int, float)) and isinstance(out_sample_metrics.get(key), (int, float)):
                        in_val = in_sample_metrics.get(key, 0)
                        out_val = out_sample_metrics.get(key, 0)
                        if in_val != 0:
                            degradation = ((out_val - in_val) / in_val) * 100
                            f.write(f" Degradation: {degradation:.2f}%")
                f.write("\n")
            
            f.write("\n=====WORKFLOW STATUS=====\n")
            f.write("Walk Forward Analysis completed successfully.\n")
        
        print(f"\nWalk Forward Test completed. Results saved to {self.output_dir}")
        
        # Return final paths
        return {
            'in_sample_results': in_sample_results,
            'out_sample_results': out_sample_results,
            'comparison': comparison,
            'summary_file': os.path.join(self.output_dir, 'walkforward_summary.txt')
        }

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