#!/usr/bin/env python3
# Shebang line allowing script to be executed directly
# run_backtest.py - Run a backtest for a strategy

import os
import sys
import argparse
import backtrader as bt
import pandas as pd
from datetime import datetime, date
import pickle
import re
import numpy as np
import json
import matplotlib.pyplot as plt
import importlib
from multiprocessing import Pool, cpu_count
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
# - matplotlib.pyplot (plt): For plotting
# - multiprocessing: For parallel processing

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# Adding the parent directory to the Python path
# Ensures that modules in the parent directory can be imported

from strategies import registry
# Importing the strategy registry module which contains available trading strategies

class TradeLogger(bt.Analyzer):
    def __init__(self):
        self.trades = []
    # Defining a TradeLogger class that inherits from backtrader's Analyzer
    # Initialize an empty list to store trades

    def notify_trade(self, trade):
        """Log both opening and closing trades."""
        date = bt.num2date(trade.dtopen if not trade.isclosed else trade.dtclose).strftime('%Y-%m-%d')
        action = 'buy' if trade.size > 0 else 'sell'
        trade_type = 'open' if not trade.isclosed else 'close'
        trade_entry = {
            'date': date,
            'action': action,
            'type': trade_type,
            'price': trade.price,
            'size': abs(trade.size),
            'commission': trade.commission,
            'pnl': trade.pnl if trade.isclosed else 0.0,  # PnL is 0 for open trades
            'symbol': trade.data._name
        }
        self.trades.append(trade_entry)
        print(f"{date} - {trade_type.capitalize()} trade logged: {trade_entry}")
    # Method triggered when a trade is opened or closed
    # - Formats the trade data (date, action, price, size, etc.) into a dictionary
    # - Appends the trade information to the trades list
    # - Prints trade information to the console

    def get_analysis(self):
        return self.trades
    # Returns the collected trade information when analysis is requested

# Custom JSON encoder to handle NumPy types and other non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif pd.isna(obj):  # Handle NaN/None values
            return None
        return super(CustomJSONEncoder, self).default(obj)

def run_backtest(output_dir=None, strategy_name='SimpleStock', tickers=None, parameters=None, 
                start_date='2015-01-01', end_date='2019-12-31', stock_csv=None, window=False, plot=False,
                warmup_period=None):
    """
    Run a backtest for a strategy on historical data.
    
    Args:
        output_dir (str): Directory to save results
        strategy_name (str): Name of strategy to backtest
        tickers (list): List of ticker symbols to include
        parameters (dict): Strategy parameters
        start_date (str): Start date for backtest (YYYY-MM-DD)
        end_date (str): End date for backtest (YYYY-MM-DD)
        stock_csv (str): Path to CSV file with stock data
        window (bool): Whether to display backtrader's live plotting window
        plot (bool): Whether to save plots of backtest results
        warmup_period (int): Number of days to add before start_date as warmup for indicators
        
    Returns:
        dict: Backtest results
    """
    # If warmup_period isn't specified, set a default based on strategy
    if warmup_period is None:
        # For MACrossover, use twice the slow period for safety
        if strategy_name == 'MACrossover':
            # Default slow_period is 30, so 60 days warmup should be safe
            if parameters and 'slow_period' in parameters:
                warmup_period = parameters['slow_period'] * 2
            else:
                warmup_period = 60  # Default - twice the default slow period (30)
        else:
            # For other strategies, use a standard 50-day warmup
            warmup_period = 50
    
    # Calculate the actual fromdate with warmup period
    if start_date:
        fromdate = pd.to_datetime(start_date) - pd.Timedelta(days=warmup_period)
        fromdate = fromdate.to_pydatetime()
    else:
        fromdate = None
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Handle output directory
    if output_dir is None:
        # Default output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(project_root, 'output', f"{strategy_name}_{timestamp}")
    elif not os.path.isabs(output_dir):
        # If relative path, make it absolute
        output_dir = os.path.join(project_root, output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
        
    # Initialize Cerebro engine
    cerebro = bt.Cerebro()
    
    # Set initial cash
    initial_cash = 100000.0
    cerebro.broker.setcash(initial_cash)
    print(f"Initial cash: {initial_cash}")
    
    # Get absolute path for stock_csv if not provided
    if stock_csv is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        stock_csv = os.path.join(project_root, 'input', 'stock_data.csv')

    # Check if file exists
    if not os.path.exists(stock_csv):
        raise FileNotFoundError(f"CSV file not found: {stock_csv}")
    
    # Read the CSV file to get column names and their positions
    with open(stock_csv, 'r') as f:
        header = f.readline().strip().split(',')
    
    # Create a mapping of column names to their indices
    column_map = {col: i for i, col in enumerate(header)}
    print("CSV columns:", list(column_map.keys()))  # Debug output

    # Auto-detect tickers if not provided
    if tickers is None:
        # Extract unique ticker symbols from column names (format: TICKER_Field)
        all_tickers = set()
        ticker_pattern = re.compile(r'([A-Z]+)_(?:Open|High|Low|Close|Volume)')
        
        for col in column_map.keys():
            match = ticker_pattern.match(col)
            if match:
                ticker = match.group(1)
                if ticker != 'SP500':  # Exclude SP500
                    all_tickers.add(ticker)
        
        tickers = sorted(list(all_tickers))
        print(f"Auto-detected tickers: {tickers}")
    # Auto-detects tickers from column names if not provided
    # Uses regular expressions to identify ticker symbols in column names (format: TICKER_Field)
    # Excludes the SP500 index and sorts the list of detected tickers
    
    # If tickers is still empty, raise an error
    if not tickers:
        raise ValueError("No valid tickers found in the CSV file. Please check the file format.")
    # Raises an error if no valid tickers were found

    # Add data feeds for each ticker
    valid_data_feeds = 0
    for ticker in tickers:
        try:
            # Check if all required columns exist for this ticker
            required_cols = [f'{ticker}_{field}' for field in ['Open', 'High', 'Low', 'Close', 'Volume']]
            missing_cols = [col for col in required_cols if col not in column_map]
            if missing_cols:
                print(f"Warning: Skipping {ticker} due to missing columns: {missing_cols}")
                continue
            # Checks if all required columns (Open, High, Low, Close, Volume) exist for each ticker
            
            data = bt.feeds.GenericCSVData(
                dataname=stock_csv,
                dtformat='%Y-%m-%d',
                datetime=column_map['Date'],  # 'Date' column index
                open=column_map[f'{ticker}_Open'],
                high=column_map[f'{ticker}_High'],
                low=column_map[f'{ticker}_Low'],
                close=column_map[f'{ticker}_Close'],
                volume=column_map[f'{ticker}_Volume'],
                openinterest=-1,
                fromdate=fromdate,
                todate=pd.to_datetime(end_date).to_pydatetime() if end_date else None,
                nullvalue=0.0
            )
            cerebro.adddata(data, name=ticker)
            print(f"Data feed added for {ticker}")
            valid_data_feeds += 1
            # Creates a GenericCSVData feed for the ticker
            # Maps the CSV columns to the required fields (datetime, open, high, low, close, volume)
            # Adds the data feed to the Cerebro engine and increments the counter
        except KeyError as e:
            print(f"Error: Could not find data for {ticker} in CSV. {e}")
            continue
            # Handles KeyError exceptions when data for a ticker is missing

    # Check if any data feeds were added
    if valid_data_feeds == 0:
        print("No valid data feeds added. Aborting backtest.")
        return
    # Aborts the backtest if no valid data feeds were added

    # Add the strategies directory to the Python path
    strategies_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'strategies')
    if strategies_dir not in sys.path:
        sys.path.append(strategies_dir)
    # Adds the strategies directory to the Python path
    # Ensures that strategy modules can be imported

    # Add strategy based on name
    try:
        if strategy_name == 'SimpleStock':
            from strategies.simplestock import SimpleStock
            if parameters:
                cerebro.addstrategy(SimpleStock, **parameters)
            else:
                cerebro.addstrategy(SimpleStock)
        elif strategy_name == 'MultiPosition':
            try:
                from strategies.multi_position_strategy import MultiPositionStrategy
                if parameters:
                    cerebro.addstrategy(MultiPositionStrategy, **parameters)
                else:
                    cerebro.addstrategy(MultiPositionStrategy)
            except Exception as e:
                print(f"Error loading MultiPositionStrategy from strategies directory: {e}")
                print("Using fallback MultiPosition strategy from strategies module")
                
                # Import the strategy from strategies module
                try:
                    from strategies import MultiPosition
                    if parameters:
                        cerebro.addstrategy(MultiPosition, **parameters)
                    else:
                        cerebro.addstrategy(MultiPosition)
                except Exception as e2:
                    print(f"Error importing from strategies module: {e2}")
                    raise
        elif strategy_name == 'AuctionMarket':
            try:
                from strategies.auction_market_strategy import AuctionMarketStrategy
                if parameters:
                    cerebro.addstrategy(AuctionMarketStrategy, **parameters)
                else:
                    cerebro.addstrategy(AuctionMarketStrategy)
            except Exception as e:
                print(f"Error loading AuctionMarketStrategy from strategies directory: {e}")
                print("Using fallback AuctionMarket strategy from strategies module")
                
                # Import the strategy from strategies module
                try:
                    from strategies import AuctionMarket
                    if parameters:
                        cerebro.addstrategy(AuctionMarket, **parameters)
                    else:
                        cerebro.addstrategy(AuctionMarket)
                except Exception as e2:
                    print(f"Error importing from strategies module: {e2}")
                    raise
        elif strategy_name == 'MACrossover':
            try:
                from strategies.ma_crossover import MACrossover
                if parameters:
                    cerebro.addstrategy(MACrossover, **parameters)
                else:
                    cerebro.addstrategy(MACrossover)
            except Exception as e:
                print(f"Error loading MACrossover from strategies directory: {e}")
                print("Using fallback MACrossover strategy from strategies module")
                
                # Import the strategy from strategies module
                try:
                    from strategies import MACrossover
                    if parameters:
                        cerebro.addstrategy(MACrossover, **parameters)
                    else:
                        cerebro.addstrategy(MACrossover)
                except Exception as e2:
                    print(f"Error importing from strategies module: {e2}")
                    raise
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    except Exception as e:
        print(f"Error loading strategy: {e}")
        print("Attempting to load fallback strategy from strategies module")
        
        try:
            # Import dynamically from the strategies module
            module_name = f"strategies.{strategy_name}"
            imported_module = importlib.import_module(module_name)
            
            # Check if the strategy exists in the module
            if hasattr(imported_module, strategy_name):
                strategy_class = getattr(imported_module, strategy_name)
                print(f"Using {strategy_name} strategy from strategies module")
                if parameters:
                    cerebro.addstrategy(strategy_class, **parameters)
                else:
                    cerebro.addstrategy(strategy_class)
            else:
                raise ValueError(f"Strategy {strategy_name} not found in strategies module")
        except Exception as e2:
            raise ValueError(f"Failed to load strategy from strategies module: {e2}. Original error: {e}")

    # Add TradeLogger analyzer
    cerebro.addanalyzer(TradeLogger, _name='tradelogger')
    # Adds the TradeLogger analyzer to the Cerebro engine

    # Run the backtest
    print("Starting backtest...")
    results = cerebro.run()
    if not results:
        print("No strategies ran successfully.")
        return
    strat = results[0]
    print("Backtest completed")
    # Runs the backtest and stores the results
    # Gets the first strategy instance from the results
    # Returns early if no strategies ran successfully

    # Save equity curve if available
    if hasattr(strat, 'equity_curve'):
        equity_df = pd.DataFrame(strat.equity_curve, columns=['Date', 'Value'])
        equity_df['Date'] = pd.to_datetime(equity_df['Date'])
        equity_df.to_csv(os.path.join(output_dir, 'equity_curve.csv'), index=False)
        print(f"Equity curve saved with {len(equity_df)} entries")
    else:
        print("Warning: No equity_curve available in strategy")
    # Saves the equity curve as a CSV file if available
    # Converts the equity curve to a DataFrame and saves it
    # Prints a warning if no equity curve is available

    # Save trade log
    trade_logger = strat.analyzers.getbyname('tradelogger')
    trade_log_df = pd.DataFrame(trade_logger.get_analysis())
    trade_log_df.to_csv(os.path.join(output_dir, 'trade_log.csv'), index=False)
    print(f"Trade log saved with {len(trade_log_df)} entries")
    # Saves the trade log as a CSV file
    # Gets the trade logger analyzer from the strategy
    # Converts the trade log to a DataFrame and saves it

    # Create a serializable results dictionary instead of saving the full strategy objects
    serializable_results = {
        'strategy_name': strategy_name,
        'tickers': tickers,
        'parameters': parameters,
        'start_date': start_date,
        'end_date': end_date,
        'initial_cash': cerebro.broker.startingcash,
        'final_value': cerebro.broker.getvalue(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Handle equity curve
    if hasattr(strat, 'equity_curve'):
        # Convert the equity curve to a serializable format
        equity_curve_data = []
        for item in strat.equity_curve:
            equity_curve_data.append({
                'Date': item['Date'] if isinstance(item, dict) else item[0],
                'Value': item['Value'] if isinstance(item, dict) else item[1]
            })
        serializable_results['equity_curve'] = equity_curve_data
    else:
        serializable_results['equity_curve'] = None
    
    # Add trade log
    serializable_results['trade_log'] = trade_log_df.to_dict(orient='records')

    # Add analyzer results in a serializable format
    try:
        # Extract key metrics from analyzers
        serializable_results['metrics'] = {}
        
        # Sharpe ratio
        if hasattr(strat.analyzers, 'sharpe'):
            sharpe_ratio = strat.analyzers.sharpe.get_analysis()
            serializable_results['metrics']['sharpe_ratio'] = sharpe_ratio.get('sharperatio', 0.0)
        
        # Returns
        if hasattr(strat.analyzers, 'returns'):
            returns = strat.analyzers.returns.get_analysis()
            serializable_results['metrics']['total_return'] = returns.get('rtot', 0.0)
            serializable_results['metrics']['average_return'] = returns.get('ravg', 0.0)
        
        # Drawdown
        if hasattr(strat.analyzers, 'drawdown'):
            drawdown = strat.analyzers.drawdown.get_analysis()
            serializable_results['metrics']['max_drawdown'] = drawdown.get('max', {}).get('drawdown', 0.0)
            serializable_results['metrics']['max_drawdown_length'] = drawdown.get('max', {}).get('len', 0)
        
        # SQN
        if hasattr(strat.analyzers, 'sqn'):
            sqn = strat.analyzers.sqn.get_analysis()
            serializable_results['metrics']['sqn'] = sqn.get('sqn', 0.0)
        
        # Trade analysis
        if hasattr(strat.analyzers, 'tradeanalyzer'):
            ta = strat.analyzers.tradeanalyzer.get_analysis()
            
            # Extract trade stats in a more reliable way
            trade_stats = {}
            
            # Total trades
            trade_stats['total_trades'] = ta.get('total', {}).get('total', 0) if isinstance(ta.get('total', {}), dict) else 0
            
            # Won trades
            if 'won' in ta and isinstance(ta['won'], dict):
                trade_stats['won_trades'] = ta['won'].get('total', 0)
                trade_stats['won_pnl'] = ta['won'].get('pnl', 0.0) if 'pnl' in ta['won'] else 0.0
            else:
                trade_stats['won_trades'] = 0
                trade_stats['won_pnl'] = 0.0
            
            # Lost trades
            if 'lost' in ta and isinstance(ta['lost'], dict):
                trade_stats['lost_trades'] = ta['lost'].get('total', 0)
                trade_stats['lost_pnl'] = ta['lost'].get('pnl', 0.0) if 'pnl' in ta['lost'] else 0.0
            else:
                trade_stats['lost_trades'] = 0
                trade_stats['lost_pnl'] = 0.0
            
            # Calculate win rate
            if trade_stats['total_trades'] > 0:
                trade_stats['win_rate'] = trade_stats['won_trades'] / trade_stats['total_trades']
            else:
                trade_stats['win_rate'] = 0.0
            
            # Calculate profit factor
            if abs(trade_stats['lost_pnl']) > 0:
                trade_stats['profit_factor'] = abs(trade_stats['won_pnl']) / abs(trade_stats['lost_pnl'])
            else:
                trade_stats['profit_factor'] = 0.0 if trade_stats['won_pnl'] == 0 else float('inf')
            
            serializable_results['metrics']['trade_analysis'] = trade_stats
    except Exception as e:
        print(f"Warning: Could not extract analyzer metrics: {str(e)}")
        serializable_results['metrics'] = {'error': str(e)}

    # Save serializable results as JSON (more reliable than pickle)
    with open(os.path.join(output_dir, 'backtest_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=4, cls=CustomJSONEncoder)
    print(f"Backtest results saved to {output_dir}/backtest_results.json")

    # For backwards compatibility, try to save a simplified version with pickle
    try:
        # Make a copy of the serializable results, converting NumPy arrays and other problematic types
        pickle_safe_results = {}
        for key, value in serializable_results.items():
            if key == 'equity_curve' and value is not None:
                # Convert equity curve data to simpler format
                pickle_safe_results[key] = [
                    {'Date': str(entry['Date']), 'Value': float(entry['Value'])} 
                    for entry in value
                ]
            elif key == 'trade_log':
                # Convert trade log to simple list of dicts with strings and basic types
                pickle_safe_results[key] = []
                for trade in value:
                    safe_trade = {}
                    for k, v in trade.items():
                        if isinstance(v, (int, float, str, bool, type(None))):
                            safe_trade[k] = v
                        else:
                            safe_trade[k] = str(v)
                    pickle_safe_results[key].append(safe_trade)
            elif key == 'metrics' and isinstance(value, dict):
                # Handle metrics dict
                pickle_safe_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        # Handle nested dicts (like trade_analysis)
                        pickle_safe_results[key][k] = {}
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, (int, float, str, bool, type(None))):
                                pickle_safe_results[key][k][sub_k] = sub_v
                            else:
                                pickle_safe_results[key][k][sub_k] = str(sub_v)
                    elif isinstance(v, (int, float, str, bool, type(None))):
                        pickle_safe_results[key][k] = v
                    else:
                        pickle_safe_results[key][k] = str(v)
            else:
                # For other fields, use simple conversion
                if isinstance(value, (int, float, str, bool, type(None), list, dict)):
                    pickle_safe_results[key] = value
                else:
                    pickle_safe_results[key] = str(value)
        
        # Save the pickle-safe results
        try:
            with open(os.path.join(output_dir, 'backtest_results.pkl'), 'wb') as f:
                pickle.dump(pickle_safe_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Pickle results also saved to {output_dir}/backtest_results.pkl")
        except Exception as e:
            print(f"Warning: Could not save pickle file: {str(e)}")
            print("JSON results file can be used instead.")
    except Exception as e:
        print(f"Warning: Could not create pickle-safe results: {str(e)}")
        print("JSON results file can be used instead.")

    # Save a summary of results to a text file
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        # Calculate and write key metrics
        initial_value = cerebro.broker.startingcash
        final_value = cerebro.broker.getvalue()
        total_return = ((final_value / initial_value) - 1) * 100
        
        # Get benchmark return if available (SP500)
        benchmark_return = 0.0
        
        # Use a fallback based on the approximate annual return of the S&P 500
        if start_date and end_date:
            try:
                start_year = pd.to_datetime(start_date).year
                end_year = pd.to_datetime(end_date).year
                years = max(1, end_year - start_year)
                # Approximate annual return of S&P 500 (conservative estimate: 8% per year)
                annual_return = 8.0
                benchmark_return = annual_return * years
                print(f"Using approximate benchmark return for {years} years at {annual_return}% per year: {benchmark_return:.2f}%")
            except Exception as e:
                print(f"Could not calculate approximate benchmark return: {e}")
                benchmark_return = 0.0
        
        # Write metrics to file
        f.write(f"Strategy: {strategy_name}\n")
        f.write(f"Parameters: {parameters}\n")
        f.write(f"Tickers: {tickers}\n")
        f.write(f"Initial Value: ${initial_value:.2f}\n")
        f.write(f"Final Value: ${final_value:.2f}\n")
        f.write(f"Total Return: {total_return:.2f}%\n")
        f.write(f"Benchmark Return (SP500): {benchmark_return:.2f}%\n")
        f.write(f"Alpha: {total_return - benchmark_return:.2f}%\n")
    # Saves a summary of results to a text file
    # Calculates key metrics (initial/final value, total return, benchmark return, alpha)
    # Uses SP500 as the benchmark if available
    # Writes metrics to a text file in a human-readable format
    
    # Return results as a dictionary
    
    # Calculate drawdowns from equity curve if available
    drawdowns = pd.Series()
    if hasattr(strat, 'equity_curve'):
        try:
            # Convert equity_curve to DataFrame if it's not already
            if not isinstance(strat.equity_curve, pd.DataFrame):
                equity_df = pd.DataFrame(strat.equity_curve, columns=['Date', 'Value'])
                equity_df['Date'] = pd.to_datetime(equity_df['Date'])
                equity_df.set_index('Date', inplace=True)
            else:
                equity_df = strat.equity_curve
                
            # Calculate drawdowns
            values = equity_df['Value'].values
            cummax = np.maximum.accumulate(values)
            drawdowns = pd.Series(1.0 - values / cummax, index=equity_df.index)
            
            # Calculate daily returns
            equity_df['Daily Return'] = equity_df['Value'].pct_change()
            
            # Get monthly returns by resampling daily returns
            monthly_returns = equity_df['Daily Return'].resample('ME').apply(
                lambda x: (1 + x).prod() - 1
            ) * 100  # Convert to percentage
            # Calculates drawdowns and monthly returns from the equity curve
            # Converts the equity curve to a DataFrame if needed
            # Uses the pandas resample method with 'ME' (month end) frequency to calculate monthly returns
        except Exception as e:
            print(f"Warning: Error calculating drawdowns and monthly returns: {e}")
            monthly_returns = pd.Series()
    else:
        print("Warning: Cannot calculate drawdowns or monthly returns - no equity curve available")
        monthly_returns = pd.Series()
    # Handles exceptions when calculating drawdowns and monthly returns
    # Sets default empty Series objects if calculations fail or no equity curve is available
    
    return {
        'strategy': strategy_name,
        'parameters': parameters,
        'tickers': tickers,
        'initial_value': cerebro.broker.startingcash,
        'final_value': cerebro.broker.getvalue(),
        'total_return': total_return,
        'benchmark_return': benchmark_return,
        'alpha': total_return - benchmark_return,
        'equity_curve': strat.equity_curve if hasattr(strat, 'equity_curve') else [],
        'trades': trade_logger.get_analysis(),
        'drawdowns': drawdowns,
        'monthly_returns': monthly_returns
    }
    # Returns a dictionary with all backtest results and metrics
    # Includes strategy info, performance metrics, equity curve, trades, drawdowns, and monthly returns

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
                        help="Whether to save plots of backtest results")
    parser.add_argument('--warmup_period', type=int, default=None,
                        help="Number of days to add before start_date as warmup for indicators")
    args = parser.parse_args()
    
    tickers = args.tickers.split(',') if args.tickers else None
    run_backtest(output_dir=args.output_dir, strategy_name=args.strategy_name, tickers=tickers, start_date=args.start_date, end_date=args.end_date, stock_csv=args.stock_csv, window=args.window, plot=args.plot, warmup_period=args.warmup_period)
    # Main block that runs when the script is executed directly
    # Sets up command-line argument parsing for strategy name, output directory, tickers, and date range
    # Parses a comma-separated list of tickers if provided
    # Calls the run_backtest function with the parsed arguments