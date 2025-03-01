import backtrader as bt

class SimpleStockStrategy(bt.Strategy):
    params = (
        ('sma_period', 20),
        ('position_size', 10),
    )

    def __init__(self):
        # Create a dictionary of SMA indicators for each data feed
        self.sma = {data: bt.ind.SimpleMovingAverage(data.close, period=self.params.sma_period) for data in self.datas}
        self.equity_curve = []

    def next(self):
        # Log portfolio value for the equity curve
        date = self.data.datetime.date(0).isoformat()
        value = self.broker.getvalue()
        self.equity_curve.append({'Date': date, 'Value': value})

        # Iterate over each data feed (ticker)
        for data in self.datas:
            close = data.close[0]
            sma = self.sma[data][0]
            position = self.getposition(data).size

            print(f"{date} - {data._name}: Close: {close:.2f}, SMA: {sma:.2f}, Position: {position}")

            # Trading logic applied to each ticker
            if close > sma and position == 0:
                self.buy(data=data, size=self.params.position_size)
                print(f"{date} - Buy order submitted for {data._name}")
            elif close < sma and position > 0:
                self.sell(data=data, size=position)
                print(f"{date} - Sell order submitted for {data._name}")