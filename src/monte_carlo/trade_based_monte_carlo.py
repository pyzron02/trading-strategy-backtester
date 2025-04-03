#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trade-Based Monte Carlo Simulation for Out-of-Sample Testing

This module implements Monte Carlo simulation by resampling trade returns
instead of permuting market data. This approach tests the robustness of
trading strategies by analyzing the distribution of possible outcomes
when trade returns are resampled.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import backtrader as bt
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union

# Import strategies and utility functions
from src.strategies.registry import registry
from src.strategies.simple_stock import SimpleStock
from src.strategies.ma_crossover import MACrossover

# Import statistics and visualization tools
from scipy import stats


class TradeBasedMonteCarloTest:
    """
    Implements Monte Carlo simulation by resampling trade returns.
    
    This class runs a strategy on out-of-sample data to generate trade returns,
    then conducts Monte Carlo simulations by resampling these returns to assess
    the strategy's robustness.
    """
    
    def __init__(
        self,
        strategy_name: str,
        parameters: Dict[str, Any],
        tickers: List[str],
        input_dir: str = "input",
        output_dir: str = None,
        num_simulations: int = 1000,
        initial_capital: float = 100000.0,
        seed: int = None,
        verbose: bool = False
    ):
        """
        Initialize the Trade-Based Monte Carlo Test.
        
        Args:
            strategy_name (str): Name of the strategy to test
            parameters (dict): Strategy parameters (optimized)
            tickers (list): List of ticker symbols
            input_dir (str): Directory containing input data
            output_dir (str): Directory to save output
            num_simulations (int): Number of Monte Carlo simulations
            initial_capital (float): Initial capital for the backtest
            seed (int): Random seed for reproducibility
            verbose (bool): Whether to print verbose output
        """
        self.strategy_name = strategy_name
        self.parameters = parameters
        self.tickers = tickers
        self.input_dir = input_dir
        
        # Create output directory if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join("output", f"trade_monte_carlo_{timestamp}")
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.num_simulations = num_simulations
        self.initial_capital = initial_capital
        self.verbose = verbose
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Store original trade returns and performance metrics
        self.original_trades = None
        self.original_metrics = None
        
        # Store Monte Carlo simulation results
        self.simulated_metrics = []
        
        # Placeholders for analysis results
        self.analysis_results = {}
        
        # Print initialization information
        if self.verbose:
            print(f"Initialized Trade-Based Monte Carlo Test for {strategy_name} strategy")
            print(f"Parameters: {parameters}")
            print(f"Tickers: {tickers}")
            print(f"Number of simulations: {num_simulations}")
    
    def load_stock_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load stock data for the specified tickers.
        
        Returns:
            dict: Dictionary mapping ticker symbols to DataFrames
        """
        ticker_data = {}
        
        for ticker in self.tickers:
            try:
                # Try different potential file paths
                potential_paths = [
                    os.path.join(self.input_dir, f"{ticker}.csv"),
                    os.path.join(self.input_dir, f"{ticker}_data.csv"),
                    os.path.join(self.input_dir, "stock_data", f"{ticker}.csv"),
                    os.path.join("data", f"{ticker}.csv")
                ]
                
                file_path = None
                for path in potential_paths:
                    if os.path.exists(path):
                        file_path = path
                        break
                
                if file_path is None:
                    print(f"Error: Could not find data file for ticker {ticker}")
                    continue
                
                # Load the data
                if self.verbose:
                    print(f"Loading data for {ticker} from {file_path}")
                    
                df = pd.read_csv(file_path)
                
                # Ensure the DataFrame has the required columns
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    print(f"Error: Missing columns {missing_columns} for ticker {ticker}")
                    continue
                
                # Ensure Date is in datetime format
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Sort by date
                df = df.sort_values('Date')
                
                # Add to ticker_data dictionary
                ticker_data[ticker] = df
                
                if self.verbose:
                    print(f"Loaded {len(df)} rows for {ticker}")
                
            except Exception as e:
                print(f"Error loading data for ticker {ticker}: {e}")
        
        return ticker_data
    
    def run_out_of_sample_backtest(self, ticker_data: Dict[str, pd.DataFrame], out_of_sample_start: str) -> Tuple[List[Dict], Dict]:
        """
        Run the strategy on out-of-sample data using optimized parameters.
        
        Args:
            ticker_data (dict): Dictionary mapping ticker symbols to DataFrames
            out_of_sample_start (str): Start date for out-of-sample period (YYYY-MM-DD)
            
        Returns:
            tuple: (List of trade dictionaries, Performance metrics dictionary)
        """
        # Convert out_of_sample_start to datetime
        out_of_sample_date = pd.to_datetime(out_of_sample_start)
        
        # Filter data for out-of-sample period
        oos_ticker_data = {}
        for ticker, df in ticker_data.items():
            oos_df = df[df['Date'] >= out_of_sample_date].copy()
            if len(oos_df) > 0:
                oos_ticker_data[ticker] = oos_df
        
        if not oos_ticker_data:
            raise ValueError(f"No out-of-sample data available after {out_of_sample_start}")
        
        if self.verbose:
            print(f"Running out-of-sample backtest from {out_of_sample_start}")
            print(f"Using parameters: {self.parameters}")
        
        # Create Cerebro instance
        cerebro = bt.Cerebro()
        
        # Set initial cash
        cerebro.broker.setcash(self.initial_capital)
        
        # Set commission
        cerebro.broker.setcommission(commission=0.001)
        
        # Add data feeds
        for ticker, df in oos_ticker_data.items():
            # Create a copy to avoid modifying the original
            df_copy = df.copy()
            
            # Ensure the Date column is a datetime
            if 'Date' in df_copy.columns:
                df_copy['Date'] = pd.to_datetime(df_copy['Date'])
                # Set Date as index for backtrader
                df_copy = df_copy.set_index('Date')
            
            # Create a data feed
            data = bt.feeds.PandasData(
                dataname=df_copy,
                datetime=None,  # Date is in the index
                open='Open',
                high='High',
                low='Low',
                close='Close',
                volume='Volume',
                openinterest=-1  # Not available
            )
            
            # Add the data feed with a name
            cerebro.adddata(data, name=ticker)
        
        # Add observers and analyzers
        cerebro.addobserver(bt.observers.Trades)
        cerebro.addobserver(bt.observers.Value)
        
        # Add analyzers for performance metrics
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        
        # Add custom trade log analyzer
        class TradeLog(bt.Analyzer):
            def __init__(self):
                self.log = []
            
            def notify_trade(self, trade):
                if trade.isclosed:
                    self.log.append({
                        'date': trade.dtclose.strftime('%Y-%m-%d'),
                        'ticker': trade.data._name,
                        'type': 'SELL' if trade.size < 0 else 'BUY',
                        'price': trade.price,
                        'size': abs(trade.size),
                        'value': trade.value,
                        'pnl': trade.pnl,
                        'commission': trade.commission
                    })
        
        cerebro.addanalyzer(TradeLog, _name='trade_log')
        
        # Add the strategy
        if self.strategy_name == "SimpleStock":
            cerebro.addstrategy(SimpleStock, **self.parameters)
        elif self.strategy_name == "MACrossover":
            cerebro.addstrategy(MACrossover, **self.parameters)
        else:
            # Try to get the strategy class from the registry
            strategy_class = registry.get_strategy_class(self.strategy_name)
            if strategy_class:
                cerebro.addstrategy(strategy_class, **self.parameters)
            else:
                raise ValueError(f"Strategy {self.strategy_name} not found")
        
        # Run the backtest
        results = cerebro.run()
        
        if not results:
            raise ValueError("No results from backtest")
        
        strategy = results[0]
        
        # Get trade log
        trade_log = strategy.analyzers.trade_log.log
        
        # Save trade log to CSV
        trade_log_path = os.path.join(self.output_dir, 'original_trade_log.csv')
        trade_log_df = pd.DataFrame(trade_log)
        if not trade_log_df.empty:
            trade_log_df.to_csv(trade_log_path, index=False)
            if self.verbose:
                print(f"Saved {len(trade_log)} trades to {trade_log_path}")
        
        # Calculate performance metrics
        final_value = strategy.broker.getvalue()
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Get metrics from analyzers
        sharpe_ratio = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
        max_drawdown = strategy.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0.0)
        
        # Get trade analysis
        trade_analysis = strategy.analyzers.trade_analyzer.get_analysis()
        
        # Calculate win rate and profit factor
        total_trades = trade_analysis.get('total', {}).get('total', 0)
        won_trades = trade_analysis.get('won', {}).get('total', 0)
        win_rate = won_trades / total_trades if total_trades > 0 else 0.0
        
        gross_won = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0.0)
        gross_lost = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0.0))
        profit_factor = gross_won / gross_lost if gross_lost > 0 else 0.0
        
        # Compile metrics
        metrics = {
            'initial_value': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades
        }
        
        # Save original metrics to JSON
        metrics_path = os.path.join(self.output_dir, 'original_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        if self.verbose:
            print(f"Out-of-sample backtest completed")
            print(f"Generated {total_trades} trades")
            print(f"Performance metrics: {metrics}")
        
        return trade_log, metrics
    
    def run_monte_carlo_simulations(self, trade_log: List[Dict]) -> List[Dict]:
        """
        Run Monte Carlo simulations by resampling trade returns.
        
        Args:
            trade_log (list): List of trade dictionaries from the original backtest
            
        Returns:
            list: List of performance metrics dictionaries for each simulation
        """
        if not trade_log:
            raise ValueError("No trades to resample")
        
        # Extract trade returns (PnL values)
        trade_returns = [trade['pnl'] for trade in trade_log]
        
        if self.verbose:
            print(f"Running {self.num_simulations} Monte Carlo simulations")
            print(f"Resampling from {len(trade_returns)} trade returns")
        
        # Run simulations with progress bar
        simulated_metrics = []
        for i in tqdm(range(self.num_simulations), desc="Monte Carlo Sims", disable=not self.verbose):
            # Resample trade returns with replacement
            resampled_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            
            # Calculate performance metrics for resampled returns
            metrics = self._calculate_performance_metrics(resampled_returns)
            metrics['simulation_id'] = i
            
            simulated_metrics.append(metrics)
        
        # Save simulated metrics to CSV
        sim_metrics_df = pd.DataFrame(simulated_metrics)
        sim_metrics_path = os.path.join(self.output_dir, 'simulation_metrics.csv')
        sim_metrics_df.to_csv(sim_metrics_path, index=False)
        
        if self.verbose:
            print(f"Completed {self.num_simulations} Monte Carlo simulations")
            print(f"Saved simulation metrics to {sim_metrics_path}")
        
        return simulated_metrics
    
    def _calculate_performance_metrics(self, trade_returns: List[float]) -> Dict:
        """
        Calculate performance metrics for a sequence of trade returns.
        
        Args:
            trade_returns (list): List of trade PnL values
            
        Returns:
            dict: Performance metrics
        """
        # Calculate cumulative return
        cumulative_return = sum(trade_returns) / self.initial_capital
        final_value = self.initial_capital * (1 + cumulative_return)
        
        # Create equity curve
        equity_curve = np.cumsum(trade_returns)
        equity_curve = np.insert(equity_curve, 0, 0)  # Start with 0
        equity_curve += self.initial_capital  # Add initial capital
        
        # Calculate daily returns (assuming each trade is a period)
        # This is a simplification since trades don't occur at regular intervals
        period_returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Calculate Sharpe ratio (simplified, using trade-by-trade returns)
        sharpe_ratio = 0.0
        if len(period_returns) > 1 and np.std(period_returns) > 0:
            sharpe_ratio = np.mean(period_returns) / np.std(period_returns) * np.sqrt(252)
        
        # Calculate maximum drawdown
        peak = equity_curve[0]
        max_drawdown = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate win rate and profit factor
        wins = sum(1 for r in trade_returns if r > 0)
        losses = sum(1 for r in trade_returns if r < 0)
        
        win_rate = wins / len(trade_returns) if len(trade_returns) > 0 else 0.0
        
        gross_profit = sum(r for r in trade_returns if r > 0)
        gross_loss = abs(sum(r for r in trade_returns if r < 0))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        return {
            'total_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trade_returns),
            'final_value': final_value
        }
    
    def analyze_results(self, original_metrics: Dict, simulated_metrics: List[Dict]) -> Dict:
        """
        Analyze the distribution of Monte Carlo simulation results.
        
        Args:
            original_metrics (dict): Performance metrics from the original backtest
            simulated_metrics (list): List of performance metrics from simulations
            
        Returns:
            dict: Analysis results
        """
        if self.verbose:
            print("Analyzing Monte Carlo simulation results")
        
        # Convert to DataFrame for easier analysis
        sim_df = pd.DataFrame(simulated_metrics)
        
        # Calculate statistics for each metric
        metrics_to_analyze = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        
        analysis_results = {}
        for metric in metrics_to_analyze:
            # Get values for this metric
            sim_values = sim_df[metric].values
            original_value = original_metrics[metric]
            
            # Calculate statistics
            stats_dict = {
                'mean': np.mean(sim_values),
                'median': np.median(sim_values),
                'std': np.std(sim_values),
                'min': np.min(sim_values),
                'max': np.max(sim_values),
                'p5': np.percentile(sim_values, 5),
                'p25': np.percentile(sim_values, 25),
                'p75': np.percentile(sim_values, 75),
                'p95': np.percentile(sim_values, 95),
                'original': original_value
            }
            
            # Calculate p-value (two-tailed)
            # For metrics where higher is better (all except max_drawdown)
            if metric != 'max_drawdown':
                p_value = np.mean(sim_values >= original_value)
            else:
                p_value = np.mean(sim_values <= original_value)
            
            stats_dict['p_value'] = p_value
            
            # Add to results
            analysis_results[metric] = stats_dict
        
        # Save analysis results to JSON
        analysis_path = os.path.join(self.output_dir, 'analysis_results.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=4)
        
        if self.verbose:
            print(f"Saved analysis results to {analysis_path}")
            
            # Print summary
            print("\nAnalysis Summary:")
            for metric, stats in analysis_results.items():
                print(f"\n{metric.replace('_', ' ').title()}:")
                print(f"  Original: {stats['original']:.4f}")
                print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Range: [{stats['p5']:.4f}, {stats['p95']:.4f}] (90% confidence)")
                print(f"  P-value: {stats['p_value']:.4f}")
        
        return analysis_results
    
    def create_visualizations(self, original_metrics: Dict, simulated_metrics: List[Dict]) -> None:
        """
        Create visualizations of the Monte Carlo simulation results.
        
        Args:
            original_metrics (dict): Performance metrics from the original backtest
            simulated_metrics (list): List of performance metrics from simulations
        """
        if self.verbose:
            print("Creating visualizations")
        
        # Create visualizations directory
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Convert to DataFrame for easier plotting
        sim_df = pd.DataFrame(simulated_metrics)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        metrics_to_plot = {
            'total_return': 'Total Return',
            'sharpe_ratio': 'Sharpe Ratio',
            'max_drawdown': 'Maximum Drawdown',
            'win_rate': 'Win Rate',
            'profit_factor': 'Profit Factor'
        }
        
        # Create distribution plots for each metric
        for metric, title in metrics_to_plot.items():
            plt.figure(figsize=(10, 6))
            
            # Get values and original value
            sim_values = sim_df[metric].values
            original_value = original_metrics[metric]
            
            # Create KDE plot
            sns.histplot(sim_values, kde=True, stat='density', alpha=0.6)
            
            # Add vertical lines for original value, mean, and percentiles
            plt.axvline(original_value, color='red', linestyle='--', linewidth=2, 
                        label=f'Original: {original_value:.4f}')
            
            mean_value = np.mean(sim_values)
            plt.axvline(mean_value, color='green', linestyle='-', linewidth=2, 
                        label=f'Mean: {mean_value:.4f}')
            
            p5 = np.percentile(sim_values, 5)
            p95 = np.percentile(sim_values, 95)
            plt.axvline(p5, color='orange', linestyle=':', linewidth=2, 
                        label=f'5th Percentile: {p5:.4f}')
            plt.axvline(p95, color='orange', linestyle=':', linewidth=2, 
                        label=f'95th Percentile: {p95:.4f}')
            
            # Calculate p-value
            if metric != 'max_drawdown':
                p_value = np.mean(sim_values >= original_value)
            else:
                p_value = np.mean(sim_values <= original_value)
            
            # Add title and labels
            plt.title(f'{title} Distribution - Monte Carlo Simulation\n'
                     f'P-value: {p_value:.4f}', fontsize=14)
            plt.xlabel(title, fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(viz_dir, f'{metric}_distribution.png')
            plt.savefig(plot_path)
            plt.close()
        
        # Create a cumulative return comparison plot
        plt.figure(figsize=(12, 8))
        
        # Generate equity curves for 100 random simulations (to avoid cluttering)
        random_indices = np.random.choice(len(simulated_metrics), size=min(100, len(simulated_metrics)), replace=False)
        
        # Plot each equity curve
        for idx in random_indices:
            metrics = simulated_metrics[idx]
            return_pct = metrics['total_return']
            final_value = self.initial_capital * (1 + return_pct)
            
            # Simple linear equity curve (this is a simplification)
            x = np.linspace(0, 1, 100)
            y = self.initial_capital * (1 + x * return_pct)
            
            plt.plot(x, y, color='gray', alpha=0.1)
        
        # Plot the original equity curve
        original_return = original_metrics['total_return']
        x = np.linspace(0, 1, 100)
        y = self.initial_capital * (1 + x * original_return)
        plt.plot(x, y, color='red', linewidth=3, label='Original')
        
        # Plot the mean equity curve
        mean_return = np.mean([m['total_return'] for m in simulated_metrics])
        y = self.initial_capital * (1 + x * mean_return)
        plt.plot(x, y, color='blue', linewidth=3, label='Mean Simulation')
        
        # Add title and labels
        plt.title('Equity Curves - Original vs. Monte Carlo Simulations', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(viz_dir, 'equity_curves_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        
        if self.verbose:
            print(f"Saved {len(metrics_to_plot)} distribution plots and equity curve comparison to {viz_dir}")
    
    def run_test(self, out_of_sample_start: str) -> Dict:
        """
        Run the complete out-of-sample Monte Carlo test.
        
        Args:
            out_of_sample_start (str): Start date for out-of-sample period (YYYY-MM-DD)
            
        Returns:
            dict: Analysis results
        """
        try:
            # Step 1: Load stock data
            ticker_data = self.load_stock_data()
            if not ticker_data:
                raise ValueError("No stock data loaded")
            
            # Step 2: Run out-of-sample backtest
            trade_log, original_metrics = self.run_out_of_sample_backtest(ticker_data, out_of_sample_start)
            if not trade_log:
                raise ValueError("No trades generated in out-of-sample backtest")
            
            # Save for later reference
            self.original_trades = trade_log
            self.original_metrics = original_metrics
            
            # Step 3: Run Monte Carlo simulations
            simulated_metrics = self.run_monte_carlo_simulations(trade_log)
            self.simulated_metrics = simulated_metrics
            
            # Step 4: Analyze results
            analysis_results = self.analyze_results(original_metrics, simulated_metrics)
            self.analysis_results = analysis_results
            
            # Step 5: Create visualizations
            self.create_visualizations(original_metrics, simulated_metrics)
            
            # Print summary of test results
            if self.verbose:
                print("\nTrade-Based Monte Carlo Test Summary:")
                print(f"Strategy: {self.strategy_name}")
                print(f"Out-of-sample period from: {out_of_sample_start}")
                print(f"Original trades: {len(trade_log)}")
                print(f"Monte Carlo simulations: {self.num_simulations}")
                print(f"Results saved to: {self.output_dir}")
            
            return analysis_results
            
        except Exception as e:
            print(f"Error running trade-based Monte Carlo test: {e}")
            import traceback
            traceback.print_exc()
            return {} 