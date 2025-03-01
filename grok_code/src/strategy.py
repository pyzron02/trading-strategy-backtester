import backtrader as bt
from datetime import date

class CoveredCallStrategy(bt.Strategy):
    params = (
        ('stock_buy_threshold', 100.0),  # Price threshold to buy stock
        ('option_strike_diff', 5.0),     # Difference from stock price for option strike
    )

    def __init__(self):
        # Verify and assign data feeds
        print(f"Number of data feeds: {len(self.datas)}")
        if len(self.datas) < 2:
            raise ValueError("CoveredCallStrategy requires two data feeds: stock and options")
        self.stock = self.datas[0]  # Stock data
        self.option = self.datas[1]  # Options data
        self.equity_curve = []  # Equity curve tracking
        self.option_sell_date = None  # Track when option was sold

    def next(self):
        date = self.stock.datetime.date(0)
        current_stock_price = self.stock.close[0]
        current_option_price = self.option.close[0] if self.option.close[0] else 0.0
        cash = self.broker.getcash()
        stock_position = self.getposition(self.stock).size
        option_position = self.getposition(self.option).size
        value = self.broker.get_value()
        self.equity_curve.append((date.strftime('%Y-%m-%d'), value))

        # Debug print for current state
        print(f"{date} - Cash: {cash:.2f}, Stock Position: {stock_position}, "
              f"Option Position: {option_position}, Stock Price: {current_stock_price:.2f}, "
              f"Option Price: {current_option_price:.2f}")

        # Buy stock if it crosses above threshold and no position exists
        if current_stock_price > self.params.stock_buy_threshold and stock_position == 0:
            stock_size = 100  # Standard lot size for covered call
            self.buy(data=self.stock, size=stock_size)
            print(f"Buy stock order submitted on {date} - Size: {stock_size}, Price: {current_stock_price:.2f}")

        # Sell call option if stock position exists and no option position
        if stock_position > 0 and option_position == 0:
            option_sell_price = current_option_price  # Use current market price
            self.sell(data=self.option, size=1, exectype=bt.Order.Limit, price=option_sell_price)
            self.option_sell_date = date  # Record sell date
            print(f"Sell option order submitted on {date} - Size: 1, Price: {option_sell_price:.2f}")

        # Close stock position after 30 days
        if stock_position > 0 and len(self) > 30:  # After 30 trading days
            self.sell(data=self.stock, size=stock_position)
            print(f"Sell stock order submitted on {date} - Size: {stock_position}")

        # Close option position after 30 days from sell date
        if option_position < 0 and self.option_sell_date and (date - self.option_sell_date).days >= 30:
            self.close(data=self.option)
            print(f"Close option order submitted on {date} - Option Position: {option_position}")

    def notify_order(self, order):
        date = self.stock.datetime.date(0).strftime('%Y-%m-%d')
        status_name = order.getstatusname(order.status)
        print(f"{date} - Order {order.ref}: Status={status_name}, Size={order.size}, "
              f"IsBuy={order.isbuy()}, Price={order.price if order.price else 'N/A'}")
        if order.status == order.Completed:
            print(f"Order executed: Price={order.executed.price}, Cost={order.executed.value}")
        elif order.status in [order.Canceled, order.Rejected, order.Margin]:
            print(f"Order failed: {status_name}")