#!/usr/bin/env python3
# Shebang line allowing script to be executed directly
# run_backtest.py - Run a backtest for a strategy

import os
import sys
import argparse
import backtrader as bt
import pandas as pd
from datetime import datetime
import pickle
import re
import numpy as np
# Importing necessary libraries:
# - os/sys: For file and path operations
# - argparse: For parsing command-line arguments
# - backtrader (bt): Main backtesting framework
# - pandas (pd): For data manipulation
# - datetime: For date handling
# - pickle: For serialization of Python objects
# - re: For regular expressions
# - numpy (np): For numerical operations

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

def run_backtest(output_dir='output', strategy_name='SimpleStock', tickers=None, parameters=None, start_date=None, end_date=None):
    """
    Run a backtest with the specified strategy on multiple tickers using a single CSV file.
    
    Args:
        output_dir (str): Directory to save results.
        strategy_name (str): Name of the strategy to run (e.g., 'SimpleStock').
        tickers (list): List of stock ticker symbols. If None, will use all tickers found in the CSV except SP500.
        parameters (dict): Dictionary of parameters to pass to the strategy.
        start_date (str): Start date for the backtest in format 'YYYY-MM-DD'.
        end_date (str): End date for the backtest in format 'YYYY-MM-DD'.
    """
    # Main function that runs a backtest with specified parameters
    # Accepts parameters for output directory, strategy, tickers, strategy parameters, and date range
    
    # Get the project root directory (3 levels up from this file)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    # Determines the project root directory by moving up three levels from the current file's location
    
    # Use the provided output_dir directly
    # If it's a relative path, make it absolute
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    # Ensures that the output directory path is absolute
    
    # Create a subdirectory for this specific parameter set
    param_hash = hash(str(parameters)) % 10000  # Simple hash to identify parameter set
    results_dir = os.path.join(output_dir, f"params_{param_hash}")
    os.makedirs(results_dir, exist_ok=True)
    # Creates a unique subdirectory for the current parameter set
    # Uses a hash of the parameters to create a unique identifier

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    print(f"Initial cash: {cerebro.broker.getcash()}")
    # Initializes the backtrader Cerebro engine
    # Sets the initial cash to $100,000 and prints the initial cash amount

    # Load single CSV file with absolute path
    stock_csv = os.path.join(project_root, 'input', 'stock_data.csv')
    if not os.path.exists(stock_csv):
        raise FileNotFoundError(f"Stock data file not found at {stock_csv}")
    # Defines the path to the stock data CSV file
    # Raises an error if the file doesn't exist
    
    # Read CSV to map column positions
    df = pd.read_csv(stock_csv)
    # Reads the CSV file into a pandas DataFrame
    
    # Filter data by date range if provided
    if start_date or end_date:
        df['Date'] = pd.to_datetime(df['Date'])
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
        # Reset index after filtering
        df = df.reset_index(drop=True)
    # Filters the DataFrame by date range if start_date or end_date is provided
    # Converts the 'Date' column to datetime, applies filters, and resets the index
    
    column_map = {col: idx for idx, col in enumerate(df.columns)}
    print("CSV columns:", list(column_map.keys()))  # Debug output
    # Creates a mapping of column names to their indices
    # Prints the column names for debugging

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
                fromdate=pd.to_datetime(start_date).to_pydatetime() if start_date else None,
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
    if strategy_name == 'SimpleStock':
        from strategies.simplestock import SimpleStock
        if parameters:
            cerebro.addstrategy(SimpleStock, **parameters)
        else:
            cerebro.addstrategy(SimpleStock)
    elif strategy_name == 'MultiPosition':
        from strategies.multi_position_strategy import MultiPositionStrategy
        if parameters:
            cerebro.addstrategy(MultiPositionStrategy, **parameters)
        else:
            cerebro.addstrategy(MultiPositionStrategy)
    elif strategy_name == 'AuctionMarket':
        from strategies.auction_market_strategy import AuctionMarketStrategy
        if parameters:
            cerebro.addstrategy(AuctionMarketStrategy, **parameters)
        else:
            cerebro.addstrategy(AuctionMarketStrategy)
    elif strategy_name == 'MACrossover':
        from strategies.ma_crossover import MACrossover
        if parameters:
            cerebro.addstrategy(MACrossover, **parameters)
        else:
            cerebro.addstrategy(MACrossover)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    # Imports and adds the specified strategy to the Cerebro engine
    # Passes parameters to the strategy if provided
    # Handles different strategy types (SimpleStock, MultiPosition, AuctionMarket, MACrossover)
    # Raises an error if the strategy is unknown

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
        equity_df.to_csv(os.path.join(results_dir, 'equity_curve.csv'), index=False)
        print(f"Equity curve saved with {len(equity_df)} entries")
    else:
        print("Warning: No equity_curve available in strategy")
    # Saves the equity curve as a CSV file if available
    # Converts the equity curve to a DataFrame and saves it
    # Prints a warning if no equity curve is available

    # Save trade log
    trade_logger = strat.analyzers.getbyname('tradelogger')
    trade_log_df = pd.DataFrame(trade_logger.get_analysis())
    trade_log_df.to_csv(os.path.join(results_dir, 'trade_log.csv'), index=False)
    print(f"Trade log saved with {len(trade_log_df)} entries")
    # Saves the trade log as a CSV file
    # Gets the trade logger analyzer from the strategy
    # Converts the trade log to a DataFrame and saves it

    # Save full results
    with open(os.path.join(results_dir, 'backtest_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"Backtest results saved to {results_dir}/backtest_results.pkl")
    # Saves the full backtest results as a pickle file
    # Allows for later reloading of the complete results object

    # Save a summary of results to a text file
    with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
        # Calculate and write key metrics
        initial_value = cerebro.broker.startingcash
        final_value = cerebro.broker.getvalue()
        total_return = ((final_value / initial_value) - 1) * 100
        
        # Get benchmark return if available (SP500)
        benchmark_return = 0.0
        try:
            if 'SP500_Close' in df.columns:
                sp500_start = df['SP500_Close'].iloc[0]
                sp500_end = df['SP500_Close'].iloc[-1]
                benchmark_return = ((sp500_end / sp500_start) - 1) * 100
        except Exception as e:
            print(f"Warning: Could not calculate benchmark return: {e}")
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a backtest with a specified strategy on multiple tickers.")
    parser.add_argument('--strategy_name', type=str, default='SimpleStock',
                        help="Name of the strategy to run (e.g., SimpleStock, MultiPosition, AuctionMarket)")
    parser.add_argument('--output_dir', type=str, default='output',
                        help="Directory to save backtest results")
    parser.add_argument('--tickers', type=str, default=None,
                        help="Comma-separated list of stock tickers (e.g., MSFT,AAPL,GOOG). If not provided, will use all tickers in the CSV except SP500.")
    parser.add_argument('--start_date', type=str, default=None,
                        help="Start date for the backtest in format 'YYYY-MM-DD'")
    parser.add_argument('--end_date', type=str, default=None,
                        help="End date for the backtest in format 'YYYY-MM-DD'")
    args = parser.parse_args()
    
    tickers = args.tickers.split(',') if args.tickers else None
    run_backtest(output_dir=args.output_dir, strategy_name=args.strategy_name, tickers=tickers, start_date=args.start_date, end_date=args.end_date)
    # Main block that runs when the script is executed directly
    # Sets up command-line argument parsing for strategy name, output directory, tickers, and date range
    # Parses a comma-separated list of tickers if provided
    # Calls the run_backtest function with the parsed arguments