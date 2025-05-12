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
import plotly.io as pio
import re

# Set Plotly template for consistent styling across all visualizations
pio.templates.default = "plotly_white"

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
                 optimized_params_path=None, plot=False, workflow_type=None, enhanced_visuals=True):
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
            workflow_type (str): Type of workflow (simple, optimization, monte_carlo, complete).
            enhanced_visuals (bool): Whether to use enhanced visualizations. Default is True.
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
        self.workflow_type = workflow_type
        self.enhanced_visuals = enhanced_visuals
        
        # Standard visualization settings
        self.plot_width = 1200
        self.plot_height = 800
        self.colors = {
            'in_sample': 'blue',
            'out_sample': 'red',
            'split_line': 'black',
            'positive': 'green',
            'negative': 'red',
            'background': 'rgba(255, 255, 255, 0.7)',
            'fill': 'rgba(0, 100, 220, 0.2)'
        }
        
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
            # Only plot if explicitly requested by user
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
            # Only plot if explicitly requested by user
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
            line=dict(color=self.colors['in_sample'], width=2),
            hovertemplate='Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Add out-of-sample equity curve
        fig.add_trace(go.Scatter(
            x=out_sample_equity_norm.index,
            y=out_sample_equity_norm.values,
            mode='lines',
            name='Out-of-Sample',
            line=dict(color=self.colors['out_sample'], width=2),
            hovertemplate='Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Add vertical line to separate in-sample and out-of-sample periods
        split_date = self.in_sample_end.strftime('%Y-%m-%d')
        fig.add_shape(
            type="line",
            x0=split_date, x1=split_date,
            y0=0, y1=1,
            yref="paper",
            line=dict(color=self.colors['split_line'], width=2, dash="dash")
        )
        
        # Add annotation for the split
        fig.add_annotation(
            x=split_date,
            y=1,
            yref="paper",
            text="Train/Test Split",
            showarrow=False,
            xanchor="right",
            font=dict(size=12)
        )
        
        # Determine title based on workflow type
        if self.workflow_type == 'complete':
            title_text = f"{self.strategy_name} Walk-Forward Testing: Optimized Strategy Performance"
        else:
            title_text = f"{self.strategy_name} Walk-Forward Testing: Strategy Performance"
        
        # Format performance values as percentages
        in_sample_return = in_sample_results.get('metrics', {}).get('total_return_pct', 
                                                in_sample_results.get('total_return', 0) * 100)
        out_sample_return = out_sample_results.get('metrics', {}).get('total_return_pct',
                                                 out_sample_results.get('total_return', 0) * 100)
        
        # Update layout with labels and title
        fig.update_layout(
            title=dict(
                text=title_text, 
                font=dict(size=24)
            ),
            xaxis_title="Date",
            yaxis_title="Normalized Equity (Starting at 100)",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1
            ),
            annotations=[
                dict(
                    x=0.15,
                    y=0.05,
                    xref="paper",
                    yref="paper",
                    text=f"In-Sample Return: {in_sample_return:.2f}%",
                    showarrow=False,
                    font=dict(
                        size=14,
                        color=self.colors['in_sample']
                    ),
                    bgcolor=self.colors['background'],
                    bordercolor=self.colors['in_sample'],
                    borderwidth=1,
                    borderpad=4
                ),
                dict(
                    x=0.85,
                    y=0.05,
                    xref="paper",
                    yref="paper",
                    text=f"Out-of-Sample Return: {out_sample_return:.2f}%",
                    showarrow=False,
                    font=dict(
                        size=14,
                        color=self.colors['out_sample']
                    ),
                    bgcolor=self.colors['background'],
                    bordercolor=self.colors['out_sample'],
                    borderwidth=1,
                    borderpad=4
                )
            ],
            hovermode="x unified",
            width=self.plot_width,  
            height=self.plot_height,
            margin=dict(l=50, r=50, t=100, b=100)  # Add margins
        )
        
        # Add grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        
        try:
            # Save plot as HTML file
            plot_file = os.path.join(self.output_dir, f"{self.strategy_name}_equity_curves.html")
            config = {
                'responsive': True,
                'displayModeBar': True,
                'modeBarButtons': [['toImage', 'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']]
            }
            try:
                fig.write_html(plot_file, include_plotlyjs='cdn', config=config, full_html=True)
                print(f"Interactive equity curves saved to {plot_file}")
            except Exception as e:
                print(f"Error saving HTML plot: {e}")
                # Fallback to PNG if HTML fails
                try:
                    png_file = os.path.join(self.output_dir, f"{self.strategy_name}_equity_curves.png")
                    fig.write_image(png_file)
                    print(f"Fallback: Equity curves saved as PNG to {png_file}")
                    plot_file = png_file
                except Exception as png_err:
                    print(f"Error saving PNG plot: {png_err}")
                    plot_file = None
            return plot_file
        except Exception as e:
            print(f"Error saving equity curves plot: {e}")
            return None
    
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
        
        # Create interactive plot with custom styling
        fig = make_subplots(
            rows=2, 
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.1,
            subplot_titles=(
                'In-Sample Period Drawdowns', 
                'Out-of-Sample Period Drawdowns'
            )
        )
        
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
            
        # Calculate max drawdown values for annotations
        try:
            in_sample_max_dd = in_sample_drawdowns.iloc[:, 0].min() * 100
            out_sample_max_dd = out_sample_drawdowns.iloc[:, 0].min() * 100
            
            # Find the dates of max drawdowns
            in_sample_max_dd_date = in_sample_drawdowns.iloc[:, 0].idxmin()
            out_sample_max_dd_date = out_sample_drawdowns.iloc[:, 0].idxmin()
        except Exception as e:
            print(f"Error calculating max drawdown values: {e}")
            in_sample_max_dd = 0
            out_sample_max_dd = 0
            in_sample_max_dd_date = None
            out_sample_max_dd_date = None
        
        # Add in-sample drawdowns
        fig.add_trace(
            go.Scatter(
                x=in_sample_drawdowns.index,
                y=in_sample_drawdowns.iloc[:, 0].values * -100,  # Convert to percentage
                mode='lines',
                name='In-Sample Drawdowns',
                line=dict(color=self.colors['in_sample'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba(0, 0, 255, 0.2)",  # Light blue with transparency
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add out-of-sample drawdowns
        fig.add_trace(
            go.Scatter(
                x=out_sample_drawdowns.index,
                y=out_sample_drawdowns.iloc[:, 0].values * -100,  # Convert to percentage
                mode='lines',
                name='Out-of-Sample Drawdowns',
                line=dict(color=self.colors['out_sample'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba(255, 0, 0, 0.2)",  # Light red with transparency
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add max drawdown annotations
        if in_sample_max_dd_date is not None:
            fig.add_annotation(
                x=in_sample_max_dd_date,
                y=in_sample_max_dd * -1,  # Negate for visualization
                text=f"Max DD: {in_sample_max_dd:.2f}%",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=self.colors['in_sample'],
                font=dict(size=12, color=self.colors['in_sample']),
                bgcolor=self.colors['background'],
                bordercolor=self.colors['in_sample'],
                borderwidth=1,
                borderpad=4,
                row=1,
                col=1
            )
            
        if out_sample_max_dd_date is not None:
            fig.add_annotation(
                x=out_sample_max_dd_date,
                y=out_sample_max_dd * -1,  # Negate for visualization
                text=f"Max DD: {out_sample_max_dd:.2f}%",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=self.colors['out_sample'],
                font=dict(size=12, color=self.colors['out_sample']),
                bgcolor=self.colors['background'],
                bordercolor=self.colors['out_sample'],
                borderwidth=1,
                borderpad=4,
                row=2,
                col=1
            )
        
        # Determine title based on workflow type
        if self.workflow_type == 'complete':
            title_text = f"{self.strategy_name} Walk-Forward Testing: Drawdown Analysis (Optimized Strategy)"
        else:
            title_text = f"{self.strategy_name} Walk-Forward Testing: Drawdown Analysis"
        
        # Update layout with improved styling
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=24)
            ),
            height=self.plot_height,
            width=self.plot_width,
            showlegend=True,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1
            ),
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Update y-axes with percentage formatting
        fig.update_yaxes(
            title_text='Drawdown (%)', 
            row=1, 
            col=1,
            gridwidth=1,
            gridcolor='LightGrey',
            zerolinecolor='black',
            zerolinewidth=1.5
        )
        
        fig.update_yaxes(
            title_text='Drawdown (%)', 
            row=2, 
            col=1,
            gridwidth=1,
            gridcolor='LightGrey',
            zerolinecolor='black',
            zerolinewidth=1.5
        )
        
        # Update x-axes with better date formatting
        fig.update_xaxes(
            title_text='Date', 
            row=1, 
            col=1,
            gridwidth=1,
            gridcolor='LightGrey',
            tickangle=45
        )
        
        fig.update_xaxes(
            title_text='Date', 
            row=2, 
            col=1,
            gridwidth=1,
            gridcolor='LightGrey',
            tickangle=45
        )
        
        try:
            # Save plot
            plot_file = os.path.join(self.output_dir, f"{self.strategy_name}_drawdowns_comparison.html")
            config = {
                'responsive': True,
                'displayModeBar': True,
                'modeBarButtons': [['toImage', 'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']]
            }
            try:
                fig.write_html(plot_file, include_plotlyjs='cdn', config=config, full_html=True)
                print(f"Interactive drawdowns comparison saved to {plot_file}")
            except Exception as e:
                print(f"Error saving HTML plot: {e}")
                # Fallback to PNG if HTML fails
                try:
                    png_file = os.path.join(self.output_dir, f"{self.strategy_name}_drawdowns_comparison.png")
                    fig.write_image(png_file)
                    print(f"Fallback: Drawdowns comparison saved as PNG to {png_file}")
                    plot_file = png_file
                except Exception as png_err:
                    print(f"Error saving PNG plot: {png_err}")
                    plot_file = None
            return plot_file
        except Exception as e:
            print(f"Error saving drawdowns plot: {e}")
            return None
    
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
        
        # Create interactive plot with custom styling
        fig = make_subplots(
            rows=2, 
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.15,
            subplot_titles=(
                'In-Sample Period Monthly Returns', 
                'Out-of-Sample Period Monthly Returns'
            )
        )
        
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
        
        # Calculate statistics for annotations
        try:
            in_sample_avg_return = in_sample_monthly.iloc[:, 0].mean() * 100
            out_sample_avg_return = out_sample_monthly.iloc[:, 0].mean() * 100
            
            in_sample_positive_months = sum(in_sample_monthly.iloc[:, 0] > 0)
            in_sample_total_months = len(in_sample_monthly)
            in_sample_hit_rate = in_sample_positive_months / in_sample_total_months if in_sample_total_months > 0 else 0
            
            out_sample_positive_months = sum(out_sample_monthly.iloc[:, 0] > 0)
            out_sample_total_months = len(out_sample_monthly)
            out_sample_hit_rate = out_sample_positive_months / out_sample_total_months if out_sample_total_months > 0 else 0
        except Exception as e:
            print(f"Error calculating monthly return statistics: {e}")
            in_sample_avg_return = 0
            out_sample_avg_return = 0
            in_sample_hit_rate = 0
            out_sample_hit_rate = 0
        
        # Add in-sample monthly returns with improved styling
        fig.add_trace(
            go.Bar(
                x=in_sample_monthly.index,
                y=in_sample_monthly.iloc[:, 0].values * 100,  # Convert to percentage
                name='In-Sample Returns',
                marker=dict(
                    color=[self.colors['positive'] if x > 0 else self.colors['negative'] 
                           for x in in_sample_monthly.iloc[:, 0].values],
                    line=dict(
                        color='rgba(0, 0, 0, 0.3)',
                        width=0.5
                    )
                ),
                hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add average line for in-sample
        fig.add_trace(
            go.Scatter(
                x=[in_sample_monthly.index.min(), in_sample_monthly.index.max()],
                y=[in_sample_avg_return, in_sample_avg_return],
                mode='lines',
                name='In-Sample Average',
                line=dict(color=self.colors['in_sample'], width=2, dash='dash'),
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Add out-of-sample monthly returns with improved styling
        fig.add_trace(
            go.Bar(
                x=out_sample_monthly.index,
                y=out_sample_monthly.iloc[:, 0].values * 100,  # Convert to percentage
                name='Out-of-Sample Returns',
                marker=dict(
                    color=[self.colors['positive'] if x > 0 else self.colors['negative'] 
                           for x in out_sample_monthly.iloc[:, 0].values],
                    line=dict(
                        color='rgba(0, 0, 0, 0.3)',
                        width=0.5
                    )
                ),
                hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add average line for out-of-sample
        fig.add_trace(
            go.Scatter(
                x=[out_sample_monthly.index.min(), out_sample_monthly.index.max()],
                y=[out_sample_avg_return, out_sample_avg_return],
                mode='lines',
                name='Out-of-Sample Average',
                line=dict(color=self.colors['out_sample'], width=2, dash='dash'),
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        # Add zero reference lines
        fig.add_shape(
            type="line",
            x0=in_sample_monthly.index.min(),
            x1=in_sample_monthly.index.max(),
            y0=0,
            y1=0,
            line=dict(
                color="black",
                width=1.5,
            ),
            row=1,
            col=1
        )
        
        fig.add_shape(
            type="line",
            x0=out_sample_monthly.index.min(),
            x1=out_sample_monthly.index.max(),
            y0=0,
            y1=0,
            line=dict(
                color="black",
                width=1.5,
            ),
            row=2,
            col=1
        )
        
        # Add annotation with statistics to each subplot
        fig.add_annotation(
            x=0.01,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Avg Monthly Return: {in_sample_avg_return:.2f}%<br>Win Rate: {in_sample_hit_rate:.1%}",
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor=self.colors['background'],
            bordercolor=self.colors['in_sample'],
            borderwidth=1,
            borderpad=4,
            row=1,
            col=1
        )
        
        fig.add_annotation(
            x=0.01,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Avg Monthly Return: {out_sample_avg_return:.2f}%<br>Win Rate: {out_sample_hit_rate:.1%}",
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor=self.colors['background'],
            bordercolor=self.colors['out_sample'],
            borderwidth=1,
            borderpad=4,
            row=2,
            col=1
        )
        
        # Determine title based on workflow type
        if self.workflow_type == 'complete':
            title_text = f"{self.strategy_name} Walk-Forward Testing: Monthly Returns (Optimized Strategy)"
        else:
            title_text = f"{self.strategy_name} Walk-Forward Testing: Monthly Returns"
        
        # Update layout with improved styling
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=24)
            ),
            height=self.plot_height,
            width=self.plot_width,
            showlegend=True,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1
            ),
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Update y-axes with percentage formatting
        fig.update_yaxes(
            title_text='Return (%)', 
            row=1, 
            col=1,
            gridwidth=1,
            gridcolor='LightGrey',
            zerolinecolor='black',
            zerolinewidth=1.5
        )
        
        fig.update_yaxes(
            title_text='Return (%)', 
            row=2, 
            col=1,
            gridwidth=1,
            gridcolor='LightGrey',
            zerolinecolor='black',
            zerolinewidth=1.5
        )
        
        # Update x-axes with better date formatting
        fig.update_xaxes(
            title_text='Month', 
            row=1, 
            col=1,
            gridwidth=1,
            gridcolor='LightGrey',
            tickformat='%b %Y',
            tickangle=45
        )
        
        fig.update_xaxes(
            title_text='Month', 
            row=2, 
            col=1,
            gridwidth=1,
            gridcolor='LightGrey',
            tickformat='%b %Y',
            tickangle=45
        )
        
        try:
            # Save plot
            plot_file = os.path.join(self.output_dir, f"{self.strategy_name}_monthly_returns_comparison.html")
            config = {
                'responsive': True,
                'displayModeBar': True,
                'modeBarButtons': [['toImage', 'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']]
            }
            try:
                fig.write_html(plot_file, include_plotlyjs='cdn', config=config, full_html=True)
                print(f"Interactive monthly returns comparison saved to {plot_file}")
            except Exception as e:
                print(f"Error saving HTML plot: {e}")
                # Fallback to PNG if HTML fails
                try:
                    png_file = os.path.join(self.output_dir, f"{self.strategy_name}_monthly_returns_comparison.png")
                    fig.write_image(png_file)
                    print(f"Fallback: Monthly returns comparison saved as PNG to {png_file}")
                    plot_file = png_file
                except Exception as png_err:
                    print(f"Error saving PNG plot: {png_err}")
                    plot_file = None
            return plot_file
        except Exception as e:
            print(f"Error saving monthly returns plot: {e}")
            return None
    
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
        
        # Only generate visualizations if plotting is enabled
        if self.plot or self.enhanced_visuals:
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
                        # Copy to strategy-prefixed filename for better frontend detection
                        strategy_prefixed_equity = os.path.join(self.output_dir, f"{self.strategy_name}_equity_curves.html")
                        try:
                            import shutil
                            shutil.copy2(equity_curves_file, strategy_prefixed_equity)
                            f.write(f"Prefixed Equity Curves: {os.path.basename(strategy_prefixed_equity)}\n")
                        except Exception as copy_err:
                            print(f"Unable to create prefixed equity curve file: {copy_err}")
                            
                    if drawdowns_file:
                        f.write(f"Drawdowns Comparison: {os.path.basename(drawdowns_file)}\n")
                        # Copy to strategy-prefixed filename for better frontend detection
                        strategy_prefixed_drawdowns = os.path.join(self.output_dir, f"{self.strategy_name}_drawdowns.html")
                        try:
                            import shutil
                            shutil.copy2(drawdowns_file, strategy_prefixed_drawdowns)
                            f.write(f"Prefixed Drawdowns: {os.path.basename(strategy_prefixed_drawdowns)}\n")
                        except Exception as copy_err:
                            print(f"Unable to create prefixed drawdowns file: {copy_err}")
                        
                    if monthly_returns_file:
                        f.write(f"Monthly Returns Comparison: {os.path.basename(monthly_returns_file)}\n")
                        # Copy to strategy-prefixed filename for better frontend detection
                        strategy_prefixed_monthly = os.path.join(self.output_dir, f"{self.strategy_name}_monthly_returns.html")
                        try:
                            import shutil
                            shutil.copy2(monthly_returns_file, strategy_prefixed_monthly)
                            f.write(f"Prefixed Monthly Returns: {os.path.basename(strategy_prefixed_monthly)}\n")
                        except Exception as copy_err:
                            print(f"Unable to create prefixed monthly returns file: {copy_err}")
                        
                    # Add a note about the enhanced visualization style
                    f.write("\nVisualization Notes:\n")
                    f.write("- Interactive plots support zooming, panning, and exporting as images\n")
                    f.write("- All visualizations use consistent styling for better comparison\n")
                    f.write("- Hover over chart elements to see detailed information\n")
                    if self.workflow_type:
                        f.write(f"- Visualizations optimized for '{self.workflow_type}' workflow type\n")
            
            except Exception as e:
                print(f"Error visualizing results: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Skipping visualization generation as plotting is disabled.")
            # Add a note about disabled visualizations to the summary file
            with open(os.path.join(self.output_dir, 'walkforward_summary.txt'), 'a') as f:
                f.write("\nVisualizations:\n")
                f.write("Visualizations were not generated because plotting is disabled.\n")
                f.write("To generate visualizations, run with the --plot flag.\n")
        
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
                    # Make sure out_value is also a float before formatting
                    if isinstance(out_value, float):
                        out_value = f"{out_value*100:.2f}%"
                elif isinstance(in_value, float):
                    in_value = f"{in_value:.4f}"
                    # Make sure out_value is also a float before formatting
                    if isinstance(out_value, float):
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
    
    parser.add_argument('--workflow_type', type=str, choices=['simple', 'optimization', 'monte_carlo', 'complete'],
                        help='Type of workflow being used (affects visualization styling)')
    
    parser.add_argument('--enhanced_visuals', action='store_true', default=True,
                        help='Use enhanced visualization styling and features')
    
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Generate plots during backtests')
    
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
        optimized_params_path=args.optimized_params_path,
        plot=args.plot,
        workflow_type=args.workflow_type,
        enhanced_visuals=args.enhanced_visuals
    )
    
    results = test.run_test()
    
    print(f"\nWalk-forward test completed. Visualizations available in {args.output_dir}") 