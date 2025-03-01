import backtrader as bt
import pandas as pd

class MultiPositionStrategy(bt.Strategy):
    params = (
        ('sma_period', 20),  # Period for the Simple Moving Average
    )

    def __init__(self):
        # Create dictionaries to access close prices and indicators by ticker name
        self.data_close = {data._name: data.close for data in self.datas}
        self.sma = {data._name: bt.indicators.SimpleMovingAverage(data.close, period=self.params.sma_period) 
                   for data in self.datas}
        # Add a list to track equity curve
        self.equity_curve = []

    def next(self):
        # Log portfolio value at each step
        portfolio_value = self.broker.getvalue()
        dt = self.datas[0].datetime.date(0)
        self.equity_curve.append({'Date': dt, 'Value': portfolio_value})
        self.log(f"Portfolio Value: {portfolio_value:.2f}")

        # Iterate over each data feed (i.e., each ticker)
        for data in self.datas:
            ticker = data._name
            current_close = self.data_close[ticker][0]
            sma_value = self.sma[ticker][0]
            position_size = self.getposition(data).size

            # Buy logic: if close > SMA and no position exists
            if current_close > sma_value and position_size == 0:
                self.buy(data=data, size=100)
                self.log(f"Bought 100 shares of {ticker}")

            # Sell logic: if close < SMA and a position exists
            elif current_close < sma_value and position_size > 0:
                self.sell(data=data, size=position_size)
                self.log(f"Sold {position_size} shares of {ticker}")

    def log(self, txt):
        """Log messages with the current date"""
        dt = self.datas[0].datetime.date(0)
        print(f"{dt}: {txt}")

    def notify_order(self, order):
        """Handle order execution notifications"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED for {order.data._name}, Price: {order.executed.price}, "
                         f"Cost: {order.executed.value}, Comm: {order.executed.comm}")
            elif order.issell():
                self.log(f"SELL EXECUTED for {order.data._name}, Price: {order.executed.price}, "
                         f"Cost: {order.executed.value}, Comm: {order.executed.comm}")

    def notify_trade(self, trade):
        """Handle trade closure notifications"""
        if trade.isclosed:
            self.log(f"Trade closed for {trade.data._name}: PnL Gross {trade.pnl}, Net {trade.pnlcomm}")

    def stop(self):
        """Save the equity curve when the strategy ends"""
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv('equity_curve.csv', index=False)
        print(f"Equity curve saved with {len(equity_df)} entries")