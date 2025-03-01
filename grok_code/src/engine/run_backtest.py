import os
import sys
import argparse
import backtrader as bt
import pandas as pd
from datetime import datetime
import pickle
import re

class TradeLogger(bt.Analyzer):
    def __init__(self):
        self.trades = []

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

    def get_analysis(self):
        return self.trades

def run_backtest(output_dir='output', strategy_name='SimpleStock', tickers=None):
    """
    Run a backtest with the specified strategy on multiple tickers using a single CSV file.
    
    Args:
        output_dir (str): Directory to save results.
        strategy_name (str): Name of the strategy to run (e.g., 'SimpleStock').
        tickers (list): List of stock ticker symbols. If None, will use all tickers found in the CSV except SP500.
    """
    # Get the project root directory (3 levels up from this file)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    strategy_dir = f"{timestamp}_{strategy_name}_portfolio"
    results_dir = os.path.join(project_root, output_dir, strategy_dir)
    os.makedirs(results_dir, exist_ok=True)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    print(f"Initial cash: {cerebro.broker.getcash()}")

    # Load single CSV file with absolute path
    stock_csv = os.path.join(project_root, 'input', 'stock_data.csv')
    if not os.path.exists(stock_csv):
        raise FileNotFoundError(f"Stock data file not found at {stock_csv}")
    
    # Read CSV to map column positions
    df = pd.read_csv(stock_csv)
    column_map = {col: idx for idx, col in enumerate(df.columns)}
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
    
    # If tickers is still empty, raise an error
    if not tickers:
        raise ValueError("No valid tickers found in the CSV file. Please check the file format.")

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
                fromdate=datetime(2020, 1, 1),
                todate=datetime(2025, 2, 25),
                nullvalue=0.0
            )
            cerebro.adddata(data, name=ticker)
            print(f"Data feed added for {ticker}")
            valid_data_feeds += 1
        except KeyError as e:
            print(f"Error: Could not find data for {ticker} in CSV. {e}")
            continue

    # Check if any data feeds were added
    if valid_data_feeds == 0:
        print("No valid data feeds added. Aborting backtest.")
        return

    # Add the strategies directory to the Python path
    strategies_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'strategies')
    if strategies_dir not in sys.path:
        sys.path.append(strategies_dir)

    # Add strategy based on name
    if strategy_name == 'SimpleStock':
        from simple_stock_strategy import SimpleStockStrategy
        cerebro.addstrategy(SimpleStockStrategy)
    elif strategy_name == 'CoveredCall':
        from strategy import CoveredCallStrategy
        cerebro.addstrategy(CoveredCallStrategy)
    elif strategy_name == 'MultiPosition':
        from multi_position_strategy import MultiPositionStrategy
        cerebro.addstrategy(MultiPositionStrategy)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Add TradeLogger analyzer
    cerebro.addanalyzer(TradeLogger, _name='tradelogger')

    # Run the backtest
    print("Starting backtest...")
    results = cerebro.run()
    if not results:
        print("No strategies ran successfully.")
        return
    strat = results[0]
    print("Backtest completed")

    # Save equity curve if available
    if hasattr(strat, 'equity_curve'):
        equity_df = pd.DataFrame(strat.equity_curve, columns=['Date', 'Value'])
        equity_df['Date'] = pd.to_datetime(equity_df['Date'])
        equity_df.to_csv(os.path.join(results_dir, 'equity_curve.csv'), index=False)
        print(f"Equity curve saved with {len(equity_df)} entries")
    else:
        print("Warning: No equity_curve available in strategy")

    # Save trade log
    trade_logger = strat.analyzers.getbyname('tradelogger')
    trade_log_df = pd.DataFrame(trade_logger.get_analysis())
    trade_log_df.to_csv(os.path.join(results_dir, 'trade_log.csv'), index=False)
    print(f"Trade log saved with {len(trade_log_df)} entries")

    # Save full results
    with open(os.path.join(results_dir, 'backtest_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"Backtest results saved to {results_dir}/backtest_results.pkl")

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
        
        # Calculate annualized metrics
        start_date = df['Date'].iloc[0]
        end_date = df['Date'].iloc[-1]
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        years = (end_date - start_date).days / 365.25
        annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Calculate volatility and Sharpe ratio if equity curve is available
        volatility = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        
        if hasattr(strat, 'equity_curve'):
            # Calculate daily returns
            equity_df = pd.DataFrame(strat.equity_curve, columns=['Date', 'Value'])
            equity_df['Date'] = pd.to_datetime(equity_df['Date'])
            equity_df.set_index('Date', inplace=True)
            equity_df['Return'] = equity_df['Value'].pct_change()
            
            # Calculate annualized volatility
            daily_volatility = equity_df['Return'].std()
            volatility = daily_volatility * (252 ** 0.5) * 100  # Annualized and as percentage
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Calculate maximum drawdown
            equity_df['Peak'] = equity_df['Value'].cummax()
            equity_df['Drawdown'] = (equity_df['Value'] - equity_df['Peak']) / equity_df['Peak'] * 100
            max_drawdown = abs(equity_df['Drawdown'].min())
        
        # Write results to file
        f.write(f"Strategy: {strategy_name}\n")
        f.write(f"Tickers: {', '.join(tickers)}\n")
        f.write(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
        f.write(f"Initial Capital: ${initial_value:.2f}\n")
        f.write(f"Final Capital: ${final_value:.2f}\n")
        f.write(f"Total Return: {total_return:.2f}%\n")
        f.write(f"Annualized Return: {annual_return:.2f}%\n")
        f.write(f"Annualized Volatility: {volatility:.2f}%\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
        f.write(f"Maximum Drawdown: {max_drawdown:.2f}%\n")
        f.write(f"Benchmark Total Return: {benchmark_return:.2f}%\n")
    
    print(f"Results summary saved to {results_dir}/results.txt")
    return results_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a backtest with a specified strategy on multiple tickers.")
    parser.add_argument('--strategy_name', type=str, default='SimpleStock',
                        help="Name of the strategy to run (e.g., SimpleStock)")
    parser.add_argument('--output_dir', type=str, default='output',
                        help="Directory to save backtest results")
    parser.add_argument('--tickers', type=str, default=None,
                        help="Comma-separated list of stock tickers (e.g., MSFT,AAPL,GOOG). If not provided, will use all tickers in the CSV except SP500.")
    args = parser.parse_args()
    
    tickers = args.tickers.split(',') if args.tickers else None
    run_backtest(output_dir=args.output_dir, strategy_name=args.strategy_name, tickers=tickers)