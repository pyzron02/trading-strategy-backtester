#!/usr/bin/env python3
# Shebang line allowing script to be executed directly
# run_backtest.py - Run a backtest for a strategy

import os
import argparse
import backtrader as bt
import pandas as pd
from datetime import datetime, date
import pickle
import re
import numpy as np
import json
import importlib

from multiprocessing import Pool, cpu_count
from typing import List, Optional, Dict, Any
# Importing necessary libraries:
# - os/sys: For file and path operations
# - argparse: For parsing command-line arguments
# - backtrader (bt): Main backtesting framework
# - pandas (pd): For data manipulation
# - datetime: For date handling
# - pickle: For serialization of Python objects
# - re: For regular expressions
# - numpy (np): For numerical operations
# - json: For JSON serialization
# - multiprocessing: For parallel processing
# - typing: For type annotations

from strategies import registry
# Importing the strategy registry module which contains available trading strategies

from engine.data_feeds import ValidatingCSVData, TradeLogger
from engine.serialization import CustomJSONEncoder
from engine.results_serializer import write_summary_file, save_backtest_results

def get_strategy_class(strategy_name):
    """
    Get the strategy class for a given strategy name.
    
    Args:
        strategy_name (str): Name of the strategy
        
    Returns:
        class: The strategy class
    """
    try:
        if strategy_name == 'SimpleStock':
            from strategies.simple_stock_strategy import SimpleStockStrategy
            return SimpleStockStrategy
        elif strategy_name == 'MultiPosition':
            from strategies.multi_position_strategy import MultiPositionStrategy
            return MultiPositionStrategy
        elif strategy_name == 'AuctionMarket':
            from strategies.auction_market_strategy import AuctionMarketStrategy
            return AuctionMarketStrategy
        elif strategy_name == 'MACrossover':
            from strategies.ma_crossover import MACrossover
            return MACrossover
        elif strategy_name == 'LimitOrder':
            from strategies.limit_order_strategy import LimitOrderStrategy
            return LimitOrderStrategy
        else:
            # Try to import dynamically
            try:
                # Try from the strategies directory
                module_name = f"strategies.{strategy_name.lower()}_strategy"
                module = __import__(module_name, fromlist=[strategy_name])
                class_name = f"{strategy_name}Strategy"
                if hasattr(module, class_name):
                    return getattr(module, class_name)
                
                # Try with exact module name
                module_name = f"strategies.{strategy_name.lower()}"
                module = __import__(module_name, fromlist=[strategy_name])
                if hasattr(module, strategy_name):
                    return getattr(module, strategy_name)
                
                # Try with exact class name
                if hasattr(module, strategy_name):
                    return getattr(module, strategy_name)
                
                # Try the registry
                from strategies import registry
                return registry.get_strategy_class(strategy_name)
            except Exception as e:
                print(f"Error importing strategy {strategy_name}: {e}")
                return None
    except Exception as e:
        print(f"Error loading strategy class {strategy_name}: {e}")
        return None

def run_backtest(
    strategy_name: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_dir: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    stock_csv: Optional[str] = None,
    plot: bool = True,
    initial_capital: float = 100000.0,
    commission: float = 0.001,
    commission_type: str = "percentage",
    data_dir: str = "input",
    verbose: bool = False,
    slippage: float = 0.0,
    enhanced_plots: bool = False,
    optimize_sharpe: bool = False,
    live_mode: bool = False,
    additional_data: Optional[Dict[str, Any]] = None,
    highlight_trades: bool = True,
    force_download: bool = False
) -> Dict[str, Any]:
    """
    Run a backtest for the given strategy and parameters.
    
    Args:
        strategy_name: Name of the trading strategy to use
        tickers: List of ticker symbols to trade
        start_date: Start date for the backtest (YYYY-MM-DD)
        end_date: End date for the backtest (YYYY-MM-DD)
        output_dir: Directory to save backtest results
        parameters: Dictionary of strategy parameters
        stock_csv: Path to CSV file with stock data (optional)
        plot: Whether to generate plots
        force_download: Force downloading data from yfinance instead of using CSV files
        initial_capital: Initial capital for the backtest
        commission: Commission rate for trades
        data_dir: Directory containing input data
        verbose: Enable verbose output
        slippage: Slippage per trade as a decimal (e.g., 0.01 for 1%)
        enhanced_plots: Whether to generate enhanced plots
        optimize_sharpe: Whether to optimize for Sharpe ratio
        live_mode: Whether to run in live mode
        additional_data: Additional data for the strategy
        
    Returns:
        Dict containing backtest results
    """
    if verbose:
        print(f"Running backtest for {strategy_name} with {tickers}")
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Ensure parameters is a dict
    if parameters is None:
        parameters = {}
    
    # Set default parameters if not provided
    if strategy_name is None:
        strategy_name = "SimpleStock"  # Default strategy
        
    if tickers is None:
        tickers = ["SPY"]  # Default ticker
    
    # Set default warmup period based on strategy if not provided
    if parameters and 'warmup_period' in parameters:
        warmup_period = parameters['warmup_period']
    else:
        # Default warmup period
        warmup_period = 252  # One year of trading days
    
    # Find the stock CSV file if not provided (unless force_download is True)
    if stock_csv is None and not force_download:
        # Look in common locations
        input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), data_dir)
        
        # Always prioritize stock_data.csv from data_setup.py
        stock_data_csv = os.path.join(input_dir, "stock_data.csv")
        if os.path.exists(stock_data_csv):
            stock_csv = stock_data_csv
            if verbose:
                print(f"Using stock_data.csv from data_setup.py at: {stock_csv}")
        else:
            # Build possible file paths as fallback
            ticker_str = "_".join(tickers) if isinstance(tickers, list) else tickers
            possible_paths = [
                os.path.join(input_dir, f"{ticker_str}.csv"),
                os.path.join(input_dir, "stocks", f"{ticker_str}.csv"),
                os.path.join(input_dir, "data", f"{ticker_str}.csv"),
            ]
            
            # Check if any of the possible paths exist
            for path in possible_paths:
                if os.path.exists(path):
                    stock_csv = path
                    if verbose:
                        print(f"Using fallback data file: {stock_csv}")
                    break
            else:
                # If no path exists, try to download the data
                try:
                    # First, try to use data_setup.py to fetch the data properly
                    from data_preprocessing.data_setup import fetch_stock_data
                    try:
                        print(f"Attempting to fetch data using data_setup.py for {tickers}")
                        fetch_stock_data(tickers, start_date, end_date)
                        if os.path.exists(stock_data_csv):
                            stock_csv = stock_data_csv
                            print(f"Successfully fetched and prepared data using data_setup.py at: {stock_csv}")
                        else:
                            raise FileNotFoundError("stock_data.csv not created by data_setup.py")
                    except Exception as e:
                        print(f"Error using data_setup.py: {e}. Falling back to yfinance direct download.")
                        # Fallback to direct yfinance download
                        import yfinance as yf
                        # Convert tickers to a comma-separated string if it's a list
                        ticker_str = ",".join(tickers) if isinstance(tickers, list) else tickers
                        # Download data
                        data = yf.download(ticker_str, start=start_date, end=end_date)
                        # Save to CSV
                        stock_csv = os.path.join(input_dir, "stock_data.csv")
                        data.to_csv(stock_csv)
                        print(f"Downloaded data for {ticker_str} and saved to {stock_csv}")
                except Exception as e:
                    error_message = f"Error downloading data: {e}"
                    print(error_message)
                    if output_dir:
                        with open(os.path.join(output_dir, 'error.log'), 'w') as f:
                            f.write(f"{error_message}\n")
                    return {"status": "error", "message": error_message}
    
    # If force_download is True, download data regardless of CSV existence
    if force_download and stock_csv is None:
        input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), data_dir)
        try:
            # First, try to use data_setup.py to fetch the data properly
            from data_preprocessing.data_setup import fetch_stock_data
            try:
                print(f"Force downloading data using data_setup.py for {tickers}")
                fetch_stock_data(tickers, start_date, end_date, force_refresh=True)
                stock_csv = os.path.join(input_dir, "stock_data.csv")
                if os.path.exists(stock_csv):
                    print(f"Successfully fetched and prepared data using data_setup.py at: {stock_csv}")
                else:
                    raise FileNotFoundError("stock_data.csv not created by data_setup.py")
            except Exception as e:
                print(f"Error using data_setup.py: {e}. Falling back to yfinance direct download.")
                # Fallback to direct yfinance download
                import yfinance as yf
                # Convert tickers to a comma-separated string if it's a list
                ticker_str = ",".join(tickers) if isinstance(tickers, list) else tickers
                # Download data
                data = yf.download(ticker_str, start=start_date, end=end_date)
                # Save to CSV
                stock_csv = os.path.join(input_dir, "stock_data.csv")
                data.to_csv(stock_csv)
                print(f"Downloaded data for {ticker_str} and saved to {stock_csv}")
        except Exception as e:
            error_message = f"Error downloading data: {e}"
            print(error_message)
            if output_dir:
                with open(os.path.join(output_dir, 'error.log'), 'w') as f:
                    f.write(f"{error_message}\n")
            return {"status": "error", "message": error_message}
    
    # Read the CSV file to extract column names
    try:
        df = pd.read_csv(stock_csv)
        columns = list(df.columns)
        
        # Map column names to their indices for easier reference
        col_map = {col: i for i, col in enumerate(columns)}
        
        # Debug output of available columns
        print(f"Available columns in CSV: {columns}")
        
        # Get ticker names if not provided
        if tickers is None or len(tickers) == 0:
            tickers = [col.split('_')[0] for col in columns if '_Close' in col]
            # Exclude SP500 from tickers, as it's usually used as a benchmark
            if 'SP500' in tickers:
                tickers.remove('SP500')
        
        # Convert start and end dates to datetime
        from_date = pd.to_datetime(start_date)
        to_date = pd.to_datetime(end_date)
        
        # Initialize Cerebro
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(initial_capital)  # Set initial cash
        if commission_type == "fixed":
            cerebro.broker.setcommission(
                commission=commission, commtype=bt.CommInfoBase.COMM_FIXED
            )
        else:
            cerebro.broker.setcommission(commission)
        
        # Check if there are any missing tickers and download them if needed
        missing_tickers = []
        valid_tickers = []
        for ticker in tickers:
            close_col = f"{ticker}_Close"
            if close_col in columns:
                valid_tickers.append(ticker)
            else:
                missing_tickers.append(ticker)
                
        if missing_tickers:
            print(f"Missing data for tickers: {missing_tickers}. Attempting to download...")
            try:
                # Import the data_setup module to fetch missing tickers
                from data_preprocessing.data_setup import fetch_stock_data
                # Force refresh to ensure we get all the missing tickers
                fetch_stock_data(tickers, start_date, end_date, force_refresh=True)
                print(f"Successfully downloaded data for missing tickers: {missing_tickers}")
                
                # Re-read the CSV file to get updated columns
                df = pd.read_csv(stock_csv)
                columns = list(df.columns)
                col_map = {col: i for i, col in enumerate(columns)}
                
                # Validate tickers again
                valid_tickers = []
                still_missing = []
                for ticker in tickers:
                    close_col = f"{ticker}_Close"
                    if close_col in columns:
                        valid_tickers.append(ticker)
                    else:
                        still_missing.append(ticker)
                
                if still_missing:
                    print(f"Warning: Still unable to get data for tickers: {still_missing}")
            except Exception as e:
                print(f"Error downloading ticker data: {e}")
        
        if not valid_tickers:
            error_message = f"No valid tickers found in CSV file: {stock_csv}"
            print(error_message)
            with open(os.path.join(output_dir, 'error.log'), 'w') as f:
                f.write(f"{error_message}\n")
            return None
        
        tickers = valid_tickers
        
        # Add data feeds for each ticker
        skipped_tickers = []
        for ticker in tickers:
            # Check if required columns exist for this ticker
            required_columns = [f"{ticker}_Open", f"{ticker}_High", f"{ticker}_Low", f"{ticker}_Close"]
            if not all(col in columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in columns]
                print(f"Warning: Skipping {ticker} due to missing columns: {missing_cols}")
                skipped_tickers.append(ticker)
                continue
            
            try:
                # Create a validating data feed from the CSV that filters out invalid prices
                data = ValidatingCSVData(
                    dataname=stock_csv,
                    fromdate=from_date,
                    todate=to_date,
                    nullvalue=0.0,
                    dtformat='%Y-%m-%d',
                    datetime=0,  # Column 0 is the date
                    open=col_map.get(f"{ticker}_Open"),
                    high=col_map.get(f"{ticker}_High"),
                    low=col_map.get(f"{ticker}_Low"),
                    close=col_map.get(f"{ticker}_Close"),
                    volume=col_map.get(f"{ticker}_Volume", -1),  # -1 means not used
                    openinterest=-1,  # Not used
                    name=ticker
                )
                cerebro.adddata(data)
                print(f"Added data feed for {ticker}")
            except Exception as e:
                print(f"Error adding data feed for {ticker}: {e}")
                skipped_tickers.append(ticker)
        
        # Remove skipped tickers from the list
        for ticker in skipped_tickers:
            if ticker in tickers:
                tickers.remove(ticker)
        
        if not tickers:
            error_message = "All tickers were skipped due to data issues"
            print(error_message)
            with open(os.path.join(output_dir, 'error.log'), 'w') as f:
                f.write(f"{error_message}\n")
            return None
    
    except Exception as e:
        error_message = f"Error processing CSV file: {e}"
        print(error_message)
        with open(os.path.join(output_dir, 'error.log'), 'w') as f:
            f.write(f"{error_message}\n")
        return None
    
    # Get the strategy class
    try:
        StrategyClass = get_strategy_class(strategy_name)
        if StrategyClass is None:
            error_message = f"Strategy class not found for {strategy_name}"
            print(error_message)
            with open(os.path.join(output_dir, 'error.log'), 'w') as f:
                f.write(f"{error_message}\n")
            return None
    except Exception as e:
        error_message = f"Error loading strategy class: {e}"
        print(error_message)
        with open(os.path.join(output_dir, 'error.log'), 'w') as f:
            f.write(f"{error_message}\n")
        return None
    
    # Print parameter debug information
    print(f"Parameters before adding strategy: {parameters}")
    
    # Normalize parameters to handle empty/None cases
    if parameters is None:
        parameters = {}
    
    # Add the strategy to Cerebro
    cerebro.addstrategy(StrategyClass, **parameters)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    cerebro.addanalyzer(TradeLogger, _name='tradelogger')
    
    
    # Run the backtest
    print(f"Starting backtest with {len(tickers)} tickers: {tickers}")
    results = cerebro.run()
    
    # There should be only one strategy instance
    if not results or len(results) == 0:
        error_message = "Backtest failed to run"
        print(error_message)
        with open(os.path.join(output_dir, 'error.log'), 'w') as f:
            f.write(f"{error_message}\n")
        return None
    
    strategy = results[0]
    
    # Extract comprehensive metrics
    metrics = {}
    
    # Initial and final values
    initial_value = cerebro.broker.startingcash
    final_value = cerebro.broker.getvalue()
    
    # Returns, drawdown, and Sharpe ratio
    total_return = (final_value / initial_value) - 1
    
    # Get max drawdown from backtrader's analyzer
    # Backtrader returns drawdown as a percentage value (e.g., 38.93 for 38.93%)
    raw_drawdown = strategy.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
    
    # Convert to decimal form for consistent storage
    max_drawdown = raw_drawdown / 100 if raw_drawdown else 0
    max_drawdown_money = strategy.analyzers.drawdown.get_analysis().get('max', {}).get('moneydown', 0)
    
    # Get sharpe ratio - properly handle the case where it's not available
    try:
        sharpe_ratio = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        if sharpe_ratio is None:
            sharpe_ratio = 0
    except (AttributeError, TypeError, KeyError):
        sharpe_ratio = 0
    
    # Add key performance metrics to the metrics dictionary
    metrics.update({
        'initial_value': initial_value,
        'final_value': final_value,
        'total_return': total_return,
        'total_return_pct': total_return * 100,  # Percentage format
        'max_drawdown': max_drawdown,  # Store as decimal (e.g., 0.3893 for 38.93%)
        'max_drawdown_pct': raw_drawdown,  # Store the original percentage (e.g., 38.93%)
        'max_drawdown_money': max_drawdown_money,
        'sharpe_ratio': sharpe_ratio
    })
    
    # Get annual returns if available
    try:
        returns_analysis = strategy.analyzers.returns.get_analysis()
        annual_return = returns_analysis.get('ravg', 0)
        metrics['annual_return'] = annual_return
        metrics['annual_return_pct'] = annual_return * 100  # Percentage format
    except (AttributeError, TypeError):
        metrics['annual_return'] = 0
        metrics['annual_return_pct'] = 0
    
    # Get SP500 return for the same period as benchmark
    benchmark_return = 0
    try:
        # First try SPY which is often used as a benchmark
        benchmark_col = None
        for possible_col in ['SPY_Close', 'SP500_Close', 'S&P500_Close', 'SPX_Close']:
            if possible_col in columns:
                benchmark_col = possible_col
                break
        
        if benchmark_col:
            benchmark_df = df[['Date', benchmark_col]].copy()
            benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])
            
            # Filter by date range
            if from_date and to_date:
                benchmark_df = benchmark_df[(benchmark_df['Date'] >= from_date.strftime('%Y-%m-%d')) & 
                                          (benchmark_df['Date'] <= to_date.strftime('%Y-%m-%d'))]
            
            # Calculate return
            if len(benchmark_df) >= 2:
                first_price = benchmark_df[benchmark_col].iloc[0]
                last_price = benchmark_df[benchmark_col].iloc[-1]
                benchmark_return = (last_price / first_price) - 1
    except Exception as e:
        print(f"Error calculating benchmark return: {e}")
    
    # Calculate alpha
    alpha = total_return - benchmark_return
    
    metrics.update({
        'benchmark_return': benchmark_return,
        'benchmark_return_pct': benchmark_return * 100,  # Percentage format
        'alpha': alpha,
        'alpha_pct': alpha * 100  # Percentage format
    })
    
    # Trade statistics
    trade_stats = strategy.analyzers.trades.get_analysis()
    
    # Detailed trade analysis
    total_trades = trade_stats.get('total', {}).get('total', 0)
    
    # Initialize trade statistics variables
    won = 0
    lost = 0
    win_rate = 0
    gross_won = 0
    gross_lost = 0
    avg_win = 0
    avg_loss = 0
    profit_factor = 0
    avg_trade_pnl = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    avg_trade_length = 0
    
    # Also extract trades from the strategy's custom trade tracking if available
    if hasattr(strategy, 'trades') and strategy.trades:
        # Use the strategy's custom trade log if it has more information
        custom_trades = [t for t in strategy.trades if t['type'] == 'close']
        if len(custom_trades) >= total_trades:
            # Calculate from custom trades (more accurate since position tracking is improved)
            total_trades = len(custom_trades)
            won = len([t for t in custom_trades if t.get('pnl', 0) > 0])
            lost = len([t for t in custom_trades if t.get('pnl', 0) <= 0])
            win_rate = won / total_trades if total_trades > 0 else 0
            
            # Get PnL information
            gross_won = sum([t.get('pnl', 0) for t in custom_trades if t.get('pnl', 0) > 0])
            gross_lost = sum([t.get('pnl', 0) for t in custom_trades if t.get('pnl', 0) <= 0])
            
            # Calculate average win/loss
            avg_win = gross_won / won if won > 0 else 0
            avg_loss = gross_lost / lost if lost > 0 else 0
            
            # Calculate profit factor with safety check for division by zero
            profit_factor = abs(gross_won / gross_lost) if gross_lost != 0 and gross_lost < 0 else float('inf')
            
            # Average trade PnL
            avg_trade_pnl = sum([t.get('pnl', 0) for t in custom_trades]) / total_trades if total_trades > 0 else 0
            
            # Get consecutive wins/losses sequences
            win_streaks, loss_streaks = [], []
            current_win_streak, current_loss_streak = 0, 0
            
            for trade in custom_trades:
                if trade.get('pnl', 0) > 0:
                    # Win
                    current_win_streak += 1
                    if current_loss_streak > 0:
                        loss_streaks.append(current_loss_streak)
                        current_loss_streak = 0
                else:
                    # Loss
                    current_loss_streak += 1
                    if current_win_streak > 0:
                        win_streaks.append(current_win_streak)
                        current_win_streak = 0
            
            # Add the last streak
            if current_win_streak > 0:
                win_streaks.append(current_win_streak)
            if current_loss_streak > 0:
                loss_streaks.append(current_loss_streak)
            
            max_consecutive_wins = max(win_streaks) if win_streaks else 0
            max_consecutive_losses = max(loss_streaks) if loss_streaks else 0
            
            # Trade duration not available in custom log typically
            avg_trade_length = trade_stats.get('len', {}).get('average', 0)
        else:
            # Fallback to built-in analyzer if custom trades don't have enough data
            if total_trades > 0:
                # Get win/loss counts
                won = trade_stats.get('won', {}).get('total', 0)
                lost = trade_stats.get('lost', {}).get('total', 0)
                win_rate = won / total_trades if total_trades > 0 else 0
                
                # Get gross PnL for winning trades (won -> pnl -> total)
                try:
                    gross_won = float(trade_stats.get('won', {}).get('pnl', {}).get('total', 0))
                except (TypeError, ValueError, AttributeError):
                    gross_won = 0
                    
                # Get gross PnL for losing trades (lost -> pnl -> total)
                try:
                    # Lost trades PnL is negative, so we take the absolute value
                    gross_lost = abs(float(trade_stats.get('lost', {}).get('pnl', {}).get('total', 0)))
                except (TypeError, ValueError, AttributeError):
                    gross_lost = 0
                
                # Get average win/loss values
                try:
                    avg_win = float(trade_stats.get('won', {}).get('pnl', {}).get('average', 0))
                except (TypeError, ValueError, AttributeError):
                    avg_win = 0
                    
                try:
                    avg_loss = float(trade_stats.get('lost', {}).get('pnl', {}).get('average', 0))
                except (TypeError, ValueError, AttributeError):
                    avg_loss = 0
                
                # Calculate profit factor with safety check for division by zero
                profit_factor = gross_won / gross_lost if gross_lost > 0 else float('inf')
                
                # Get average trade PnL (pnl -> net -> average)
                try:
                    avg_trade_pnl = float(trade_stats.get('pnl', {}).get('net', {}).get('average', 0))
                except (TypeError, ValueError, AttributeError):
                    # Fallback to gross average if net is not available
                    try:
                        avg_trade_pnl = float(trade_stats.get('pnl', {}).get('gross', {}).get('average', 0))
                    except (TypeError, ValueError, AttributeError):
                        avg_trade_pnl = 0
                
                # Get max consecutive wins and losses
                try:
                    max_consecutive_wins = trade_stats.get('streak', {}).get('won', {}).get('longest', 0)
                except (TypeError, AttributeError):
                    max_consecutive_wins = 0
                    
                try:
                    max_consecutive_losses = trade_stats.get('streak', {}).get('lost', {}).get('longest', 0)
                except (TypeError, AttributeError):
                    max_consecutive_losses = 0
                
                # Get average trade duration
                try:
                    avg_trade_length = trade_stats.get('len', {}).get('average', 0)
                except (TypeError, AttributeError):
                    avg_trade_length = 0
    else:
        # Strategy doesn't have custom trades, use built-in analyzer
        if total_trades > 0:
            # Get win/loss counts
            won = trade_stats.get('won', {}).get('total', 0)
            lost = trade_stats.get('lost', {}).get('total', 0)
            win_rate = won / total_trades if total_trades > 0 else 0
            
            # Get gross PnL for winning trades (won -> pnl -> total)
            try:
                gross_won = float(trade_stats.get('won', {}).get('pnl', {}).get('total', 0))
            except (TypeError, ValueError, AttributeError):
                gross_won = 0
                
            # Get gross PnL for losing trades (lost -> pnl -> total)
            try:
                # Lost trades PnL is negative, so we take the absolute value
                gross_lost = abs(float(trade_stats.get('lost', {}).get('pnl', {}).get('total', 0)))
            except (TypeError, ValueError, AttributeError):
                gross_lost = 0
            
            # Get average win/loss values
            try:
                avg_win = float(trade_stats.get('won', {}).get('pnl', {}).get('average', 0))
            except (TypeError, ValueError, AttributeError):
                avg_win = 0
                
            try:
                avg_loss = float(trade_stats.get('lost', {}).get('pnl', {}).get('average', 0))
            except (TypeError, ValueError, AttributeError):
                avg_loss = 0
            
            # Calculate profit factor with safety check for division by zero
            profit_factor = gross_won / gross_lost if gross_lost > 0 else float('inf')
            
            # Get average trade PnL (pnl -> net -> average)
            try:
                avg_trade_pnl = float(trade_stats.get('pnl', {}).get('net', {}).get('average', 0))
            except (TypeError, ValueError, AttributeError):
                # Fallback to gross average if net is not available
                try:
                    avg_trade_pnl = float(trade_stats.get('pnl', {}).get('gross', {}).get('average', 0))
                except (TypeError, ValueError, AttributeError):
                    avg_trade_pnl = 0
            
            # Get max consecutive wins and losses
            try:
                max_consecutive_wins = trade_stats.get('streak', {}).get('won', {}).get('longest', 0)
            except (TypeError, AttributeError):
                max_consecutive_wins = 0
                
            try:
                max_consecutive_losses = trade_stats.get('streak', {}).get('lost', {}).get('longest', 0)
            except (TypeError, AttributeError):
                max_consecutive_losses = 0
            
            # Get average trade duration
            try:
                avg_trade_length = trade_stats.get('len', {}).get('average', 0)
            except (TypeError, AttributeError):
                avg_trade_length = 0
    
    # Update metrics with trade statistics
    if total_trades > 0:
        metrics.update({
            'total_trades': total_trades,
            'winning_trades': won,
            'losing_trades': lost,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,  # Percentage format
            'profit_factor': profit_factor,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'gross_profit': gross_won,
            'gross_loss': gross_lost if isinstance(gross_lost, float) and gross_lost < 0 else -gross_lost,  # Make it negative for clarity
            'net_profit': gross_won + (gross_lost if isinstance(gross_lost, float) and gross_lost < 0 else -gross_lost),
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_trade_length': avg_trade_length
        })
    
    # Equity curve (if available)
    equity_curve = None
    drawdowns = None
    monthly_returns = None
    
    if hasattr(strategy, 'equity_curve') and strategy.equity_curve:
        # Convert to pandas DataFrame
        equity_curve = pd.DataFrame(strategy.equity_curve)
        
        # Save to CSV
        equity_curve_file = os.path.join(output_dir, 'equity_curve.csv')
        equity_curve.to_csv(equity_curve_file, index=False)
        print(f"Saved equity curve to {equity_curve_file}")
        
        # Calculate drawdowns if we have an equity curve
        try:
            if 'Value' in equity_curve.columns and 'Date' in equity_curve.columns:
                # Set Date as index
                eq_curve = equity_curve.copy()
                eq_curve['Date'] = pd.to_datetime(eq_curve['Date'])
                eq_curve.set_index('Date', inplace=True)
                
                # Calculate rolling maximum
                if not eq_curve.empty:
                    rolling_max = eq_curve['Value'].cummax()
                    # Calculate drawdown series in dollar amounts (will be negative values)
                    drawdowns_dollars = eq_curve['Value'] - rolling_max
                    
                    # Also calculate percentage drawdowns for backwards compatibility
                    drawdowns_pct = (eq_curve['Value'] / rolling_max) - 1
                    
                    # Save dollar drawdowns to CSV (primary output)
                    if drawdowns_dollars is not None and not drawdowns_dollars.empty:
                        drawdowns_file = os.path.join(output_dir, 'drawdowns.csv')
                        drawdowns_dollars.to_csv(drawdowns_file, header=['Drawdown'])
                        print(f"Saved dollar drawdowns to {drawdowns_file}")
                        
                        # Also save percentage drawdowns as separate file for reference
                        drawdowns_pct_file = os.path.join(output_dir, 'drawdowns_pct.csv')
                        drawdowns_pct.to_csv(drawdowns_pct_file, header=['Drawdown_Pct'])
                        print(f"Saved percentage drawdowns to {drawdowns_pct_file}")
                    
                    # Calculate monthly returns
                    # Resample to month end and calculate percent change
                    monthly_returns = eq_curve['Value'].resample('M').last().pct_change().dropna()
                    
                    # Save monthly returns to CSV
                    if monthly_returns is not None and not monthly_returns.empty:
                        monthly_returns_file = os.path.join(output_dir, 'monthly_returns.csv')
                        monthly_returns.to_csv(monthly_returns_file, header=['Return'])
                        print(f"Saved monthly returns to {monthly_returns_file}")
        except Exception as e:
            print(f"Error calculating drawdowns or monthly returns: {e}")
    
    # Trade log (if available)
    trade_log = None
    
    # Check for PairsTrading strategy with trades_history
    if hasattr(strategy, 'trades_history') and strategy.trades_history:
        # Convert PairsTrading trades to standard format
        pairs_trades = []
        for trade in strategy.trades_history:
            # Extract entry data for both assets
            asset1 = trade['asset1']
            asset2 = trade['asset2']
            entry_date = trade['entry_date']
            exit_date = trade.get('exit_date', entry_date)
            
            # Create individual trade entries for each asset
            # Asset 1 trades
            pairs_trades.append({
                'date': entry_date,
                'action': 'buy' if asset1['size'] > 0 else 'sell',
                'type': 'open',
                'ticker': asset1['name'],
                'price': asset1['price'],
                'size': abs(asset1['size']),
                'commission': 0,  # Not tracked in PairsTrading
                'pnl': 0
            })
            
            pairs_trades.append({
                'date': exit_date,
                'action': 'sell' if asset1['size'] > 0 else 'buy',
                'type': 'close',
                'ticker': asset1['name'],
                'price': asset1.get('exit_price', asset1['price']),
                'size': abs(asset1['size']),
                'commission': 0,
                'pnl': trade.get('pnl', 0) / 2  # Split PnL between assets
            })
            
            # Asset 2 trades
            pairs_trades.append({
                'date': entry_date,
                'action': 'buy' if asset2['size'] > 0 else 'sell',
                'type': 'open',
                'ticker': asset2['name'],
                'price': asset2['price'],
                'size': abs(asset2['size']),
                'commission': 0,
                'pnl': 0
            })
            
            pairs_trades.append({
                'date': exit_date,
                'action': 'sell' if asset2['size'] > 0 else 'buy',
                'type': 'close',
                'ticker': asset2['name'],
                'price': asset2.get('exit_price', asset2['price']),
                'size': abs(asset2['size']),
                'commission': 0,
                'pnl': trade.get('pnl', 0) / 2  # Split PnL between assets
            })
        
        if pairs_trades:
            trade_log = pd.DataFrame(pairs_trades)
            trade_log_file = os.path.join(output_dir, 'trade_log.csv')
            trade_log.to_csv(trade_log_file, index=False)
            print(f"Saved PairsTrading trade log to {trade_log_file}")
    
    # First check the strategy's custom trades attribute
    elif hasattr(strategy, 'trades') and strategy.trades:
        # Convert to pandas DataFrame
        trade_log = pd.DataFrame(strategy.trades)
        
        # Save to CSV
        trade_log_file = os.path.join(output_dir, 'trade_log.csv')
        trade_log.to_csv(trade_log_file, index=False)
        print(f"Saved trade log to {trade_log_file}")
    
    # Also check the TradeLogger analyzer for trades (as a backup)
    elif hasattr(strategy.analyzers, 'tradelogger'):
        logger_trades = strategy.analyzers.tradelogger.get_analysis()
        if logger_trades:
            # Convert to pandas DataFrame
            trade_log = pd.DataFrame(logger_trades)
            
            # Save to CSV
            trade_log_file = os.path.join(output_dir, 'trade_log.csv')
            trade_log.to_csv(trade_log_file, index=False)
            print(f"Saved trade log from TradeLogger analyzer to {trade_log_file}")
    
    # Generate plots only if explicitly requested via the --plot flag
    if plot:
        # Collect data for all tickers from data feeds
        ticker_data = {}
        for data in cerebro.datas:
            ticker = data._name
            # Convert array.array to list for plotly compatibility
            ticker_data[ticker] = {
                'dates': [bt.num2date(x).strftime('%Y-%m-%d') for x in data.datetime.get(size=len(data))],
                'open': list(data.open.get(size=len(data))),
                'high': list(data.high.get(size=len(data))),
                'low': list(data.low.get(size=len(data))),
                'close': list(data.close.get(size=len(data))),
                'volume': list(data.volume.get(size=len(data))) if hasattr(data, 'volume') and data.volume is not None else None
            }

        from visualization.backtest_charts import generate_backtest_visualizations
        generate_backtest_visualizations(
            output_dir=output_dir,
            strategy_name=strategy_name,
            tickers=tickers,
            equity_curve=equity_curve,
            drawdowns=drawdowns,
            trade_log=trade_log,
            ticker_data=ticker_data,
            metrics=metrics,
            cerebro=cerebro,
            strategy=strategy,
            highlight_trades=highlight_trades,
            enhanced_plots=enhanced_plots
        )
    
    # Create result dictionary
    result = {
        "status": "success",
        "strategy_name": strategy_name,
        "start_date": start_date,  # Include start_date explicitly
        "end_date": end_date,      # Include end_date explicitly
        "parameters": parameters,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "trade_log": trade_log,
        "drawdowns": drawdowns,
        "monthly_returns": monthly_returns
    }
    
    # If we have no trades in trade_log, check if the TradeLogger has any
    if (trade_log is None or trade_log.empty) and hasattr(strategy.analyzers, 'tradelogger'):
        logger_trades = strategy.analyzers.tradelogger.get_analysis()
        if logger_trades:
            result["trade_log"] = pd.DataFrame(logger_trades)
    
    # Include parameters in metrics so write_summary_file can access them
    metrics['parameters'] = parameters

    # Write human-readable performance summary
    write_summary_file(
        output_dir, metrics, initial_value, final_value,
        strategy_name, tickers, start_date, end_date
    )

    # Serialize results to JSON and pickle
    save_backtest_results(output_dir, result, metrics)
    
    print(f"Backtest complete. Results saved to {output_dir}")
    return result

def run_parallel_backtests(backtest_configs, num_workers=None):
    """
    Run multiple backtest configurations in parallel using multiprocessing.
    
    Args:
        backtest_configs (list): List of dictionaries, each containing parameters for run_backtest
        num_workers (int): Number of parallel workers (CPU cores) to use. 
                          If None, uses all available cores minus one.
    
    Returns:
        list: List of backtest results in the same order as the input configurations
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Use all cores except one by default
    
    print(f"Running {len(backtest_configs)} backtests in parallel using {num_workers} CPU cores")
    
    # Create a pool of workers
    with Pool(processes=num_workers) as pool:
        # Map each backtest configuration to the run_backtest function
        results = pool.map(_run_backtest_wrapper, backtest_configs)
    
    return results

def _run_backtest_wrapper(config):
    """
    Wrapper function for run_backtest to be used with multiprocessing.
    
    Args:
        config (dict): Dictionary containing parameters for run_backtest
    
    Returns:
        dict: Results from the backtest
    """
    try:
        # Extract parameters from config dictionary
        return run_backtest(**config)
    except Exception as e:
        print(f"Error in backtest: {str(e)}")
        # Return error information instead of raising exception to avoid crashing the pool
        return {
            'error': str(e),
            'config': config,
            'success': False
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a backtest with a specified strategy on multiple tickers.")
    parser.add_argument('--strategy_name', type=str, default='SimpleStock',
                        help="Name of the strategy to run (e.g., SimpleStock, MultiPosition, AuctionMarket)")
    parser.add_argument('--output_dir', type=str, default='output',
                        help="Directory to save backtest results")
    parser.add_argument('--tickers', type=str, default=None,
                        help="Comma-separated list of stock tickers (e.g., MSFT,AAPL,GOOG). If not provided, will use all tickers in the CSV except SP500.")
    parser.add_argument('--start_date', type=str, default='2015-01-01',
                        help="Start date for the backtest in format 'YYYY-MM-DD'")
    parser.add_argument('--end_date', type=str, default='2019-12-31',
                        help="End date for the backtest in format 'YYYY-MM-DD'")
    parser.add_argument('--stock_csv', type=str, default=None,
                        help="Path to CSV file with stock data")
    parser.add_argument('--window', action='store_true',
                        help="Whether to display backtrader's live plotting window")
    parser.add_argument('--plot', action='store_true',
                        help="Generate and save plots of backtest results (disabled by default)")
    parser.add_argument('--warmup_period', type=int, default=None,
                        help="Number of days to add before start_date as warmup for indicators")
    args = parser.parse_args()
    
    tickers = args.tickers.split(',') if args.tickers else None
    run_backtest(output_dir=args.output_dir, strategy_name=args.strategy_name, tickers=tickers, 
                 start_date=args.start_date, end_date=args.end_date, stock_csv=args.stock_csv, 
                 plot=args.plot, verbose=True)
    # Main block that runs when the script is executed directly
    # Sets up command-line argument parsing for strategy name, output directory, tickers, and date range
    # Parses a comma-separated list of tickers if provided
    # Calls the run_backtest function with the parsed arguments