import os
import argparse
import backtrader as bt
import pandas as pd
from datetime import datetime
import pickle

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

def run_backtest(output_dir='output', strategy_name='SimpleStock', tickers=['MSFT']):
    """
    Run a backtest with the specified strategy on multiple tickers using a single CSV file.
    
    Args:
        output_dir (str): Directory to save results.
        strategy_name (str): Name of the strategy to run (e.g., 'SimpleStock').
        tickers (list): List of stock ticker symbols (e.g., ['MSFT', 'AAPL']).
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    strategy_dir = f"{timestamp}_{strategy_name}_portfolio"
    results_dir = os.path.join(output_dir, strategy_dir)
    os.makedirs(results_dir, exist_ok=True)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    print(f"Initial cash: {cerebro.broker.getcash()}")

    # Load single CSV file
    stock_csv = 'input/stock_data.csv'
    if not os.path.exists(stock_csv):
        raise FileNotFoundError(f"Stock data file not found at {stock_csv}")
    
    # Read CSV to map column positions
    df = pd.read_csv(stock_csv)
    column_map = {col: idx for idx, col in enumerate(df.columns)}
    print("CSV columns:", list(column_map.keys()))  # Debug output

    # Add data feeds for each ticker
    valid_data_feeds = 0
    for ticker in tickers:
        try:
            # Check if all required columns exist for this ticker
            required_cols = [f'{ticker}_{field}' for field in ['Open', 'High', 'Low', 'Close', 'Volume']]
            missing_cols = [col for col in required_cols if col not in column_map]
            if missing_cols:
                raise KeyError(f"Missing columns: {missing_cols}")
            
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

    # Add strategy based on name
    if strategy_name == 'SimpleStock':
        from simple_stock_strategy import SimpleStockStrategy
        cerebro.addstrategy(SimpleStockStrategy)
    elif strategy_name == 'CoveredCall':
        from strategy import CoveredCallStrategy
        cerebro.addstrategy(CoveredCallStrategy, **kwargs)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a backtest with a specified strategy on multiple tickers.")
    parser.add_argument('--strategy_name', type=str, default='SimpleStock',
                        help="Name of the strategy to run (e.g., SimpleStock)")
    parser.add_argument('--output_dir', type=str, default='output',
                        help="Directory to save backtest results")
    parser.add_argument('--tickers', type=str, default='MSFT,AAPL',
                        help="Comma-separated list of stock tickers (e.g., MSFT,AAPL,GOOG)")
    args = parser.parse_args()
    
    tickers = args.tickers.split(',')
    run_backtest(output_dir=args.output_dir, strategy_name=args.strategy_name, tickers=tickers)