#!/usr/bin/env python3
# data_feeds.py - Custom data feeds and analyzers for the backtest engine

import backtrader as bt


# Custom data feed that filters out invalid price data
class ValidatingCSVData(bt.feeds.GenericCSVData):
    def _load(self):
        # Load the next line from the data source
        if not super()._load():
            return False

        # Check if the current data point has valid prices
        if (self.lines.close[0] <= 0 or self.lines.open[0] <= 0 or
            self.lines.high[0] <= 0 or self.lines.low[0] <= 0):
            # Skip this data point by loading the next one
            print(f"DATAFEED: Skipping invalid price data on {self.datetime.date(0)} for {self._name}")
            return self._load()  # Recursively load next valid data point

        return True

class TradeLogger(bt.Analyzer):
    def __init__(self):
        self.trades = []
        self.execution_prices = {}  # Track execution prices from order notifications
        print("TradeLogger analyzer initialized")
    # Defining a TradeLogger class that inherits from backtrader's Analyzer
    # Initialize an empty list to store trades

    def start(self):
        """Called when the analyzer is started."""
        print("TradeLogger analyzer started")

    def stop(self):
        """Called when the analyzer is stopped."""
        print(f"TradeLogger analyzer stopped. Total trades logged: {len(self.trades)}")

    def notify_order(self, order):
        """Capture actual execution prices when orders are filled."""
        if order.status == order.Completed:
            # Check for invalid execution prices (holiday/missing data issue)
            exec_price = order.executed.price
            if exec_price <= 0:
                print(f"WARNING: Order executed with invalid price {exec_price} on {bt.num2date(order.executed.dt).strftime('%Y-%m-%d')} - likely holiday/missing data")
                # Don't store invalid execution prices
                return

            # Store the actual execution price with timestamp
            order_key = f"{order.data._name}_{bt.num2date(order.executed.dt).strftime('%Y-%m-%d')}"
            if order_key not in self.execution_prices:
                self.execution_prices[order_key] = []

            self.execution_prices[order_key].append({
                'price': exec_price,
                'size': order.executed.size,
                'action': 'buy' if order.executed.size > 0 else 'sell',
                'date': bt.num2date(order.executed.dt).strftime('%Y-%m-%d')
            })

    def notify_trade(self, trade):
        """Log both opening and closing trades using actual execution prices."""
        try:
            # For open trades
            if not trade.isclosed:
                date = bt.num2date(trade.dtopen).strftime('%Y-%m-%d')
                action = 'buy' if trade.size > 0 else 'sell'

                trade_entry = {
                    'date': date,
                    'action': action,
                    'type': 'open',
                    'price': trade.price,  # Opening price is always correct
                    'size': abs(trade.size),
                    'commission': trade.commission,
                    'pnl': 0.0,
                    'ticker': trade.data._name
                }

                self.trades.append(trade_entry)
                print(f"{date} - Open trade logged: {trade_entry}")

            else:
                # For closed trades, try to get the actual execution price
                date = bt.num2date(trade.dtclose).strftime('%Y-%m-%d')
                action = 'buy' if trade.size > 0 else 'sell'

                # Look for the closing execution price in our stored prices
                order_key = f"{trade.data._name}_{date}"
                closing_price = trade.price  # Default fallback

                if order_key in self.execution_prices:
                    # Find the most recent execution for this date
                    executions = self.execution_prices[order_key]
                    if executions:
                        # Use the last execution price for this date
                        closing_price = executions[-1]['price']
                else:
                    # No valid execution price found - likely holiday/missing data issue
                    # Calculate from PnL as backup (but warn about it)
                    if abs(trade.size) > 0 and trade.commission > 0:
                        # Use the original calculation method as fallback
                        calculated_price = (trade.pnl + trade.commission) / abs(trade.size) + trade.price
                        if calculated_price > 0:  # Only use if reasonable
                            closing_price = calculated_price
                        else:
                            # Use opening price as final fallback
                            closing_price = trade.price

                trade_entry = {
                    'date': date,
                    'action': action,
                    'type': 'close',
                    'price': closing_price,
                    'size': 0,  # Size is 0 for close trades in our logging format
                    'commission': trade.commission,
                    'pnl': trade.pnl,
                    'ticker': trade.data._name
                }

                self.trades.append(trade_entry)
                print(f"{date} - Close trade logged: {trade_entry}")

        except Exception as e:
            print(f"Error in TradeLogger.notify_trade: {e}")
    # Method triggered when a trade is opened or closed
    # - Formats the trade data (date, action, price, size, etc.) into a dictionary
    # - Appends the trade information to the trades list
    # - Prints trade information to the console

    def get_analysis(self):
        print(f"TradeLogger.get_analysis called. Returning {len(self.trades)} trades.")
        return self.trades
    # Returns the collected trade information when analysis is requested
