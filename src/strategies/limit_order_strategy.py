#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Limit Order Strategy Implementation.

Replicates a limit-order-based trading system with:
- External signal file (CSV) for buy/sell triggers
- Dollar-based position sizing per ticker
- Priority-based cash allocation across tickers
- Limit orders placed at the current bar's close price
- Flat-fee commission support
"""

import os
import math
import csv
from datetime import datetime, date

import backtrader as bt


class LimitOrderStrategy(bt.Strategy):
    """
    A strategy that places limit orders based on external signals.

    Signals are loaded from a CSV file with columns: Date, Ticker, Signal
    where Signal is 1 (buy), -1 (sell), or 0 (no action).

    Orders are placed as limit orders at the current bar's close price.
    Position sizing is dollar-based: floor((stake_amount - txn_cost) / price)
    for buys, and floor(stake_amount / price) for sells.

    Args:
        signal_file: Path to CSV with Date,Ticker,Signal columns.
        stake_amounts: Dict mapping ticker names to dollar amounts per trade.
        ticker_priority: List of tickers in priority order for cash allocation.
        transaction_cost: Flat fee per trade (default $0.99).
        warmup_period: Bars to skip before trading (default 0).
    """

    params = (
        ('signal_file', None),
        ('stake_amounts', None),
        ('ticker_priority', None),
        ('transaction_cost', 0.99),
        ('warmup_period', 0),
    )

    def __init__(self):
        """Initialize the strategy: load signals and set up tracking."""
        self.signals = {}
        if self.params.signal_file:
            self.signals = self._load_signals(self.params.signal_file)

        # Build a name -> data mapping for quick lookup
        self.data_by_name = {}
        for d in self.datas:
            self.data_by_name[d._name] = d

        # Per-ticker pending order tracking (ticker -> order object)
        self.pending_orders = {}

        # Logging / output lists
        self.order_log = []
        self.equity_curve = []
        self.trades = []

        # Track open-trade entry info for P&L calculation
        # {ticker: [{price, size, date}, ...]}
        self._open_positions = {}

        self.bars_processed = 0

    # -----------------------------------------------------------------
    # Signal loading
    # -----------------------------------------------------------------
    def _load_signals(self, filepath):
        """Load signals from a CSV file.

        Args:
            filepath: Path to CSV with Date, Ticker, Signal columns.

        Returns:
            Dict keyed by (datetime.date, ticker_str) -> int signal.
        """
        signals = {}
        if not os.path.exists(filepath):
            print(f"[LimitOrder] WARNING: Signal file not found: {filepath}")
            return signals

        with open(filepath, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_date = row['Date'].strip()
                ticker = row['Ticker'].strip()
                sig = int(float(row['Signal']))
                if sig not in (-1, 0, 1):
                    raise ValueError(
                        f"Invalid signal {sig} on {raw_date} for {ticker}"
                    )
                if sig == 0:
                    continue
                # Parse date – accept YYYY-MM-DD or MM/DD/YYYY
                try:
                    dt = datetime.strptime(raw_date, '%Y-%m-%d').date()
                except ValueError:
                    dt = datetime.strptime(raw_date, '%m/%d/%Y').date()
                signals[(dt, ticker)] = sig

        print(f"[LimitOrder] Loaded {len(signals)} signals from {filepath}")
        return signals

    # -----------------------------------------------------------------
    # Core bar-by-bar logic
    # -----------------------------------------------------------------
    def next(self):
        """Execute trading logic on each bar."""
        self.bars_processed += 1

        # Record equity curve
        cur_date = self.datas[0].datetime.date(0)
        portfolio_value = self.broker.getvalue()
        self.equity_curve.append({
            'Date': cur_date.isoformat(),
            'Value': portfolio_value,
        })

        # Skip warmup
        if self.bars_processed <= self.params.warmup_period:
            return

        # Determine ticker processing order
        priority = self.params.ticker_priority
        if priority:
            ordered_datas = []
            for name in priority:
                if name in self.data_by_name:
                    ordered_datas.append(self.data_by_name[name])
            # Append any data feeds not in the priority list
            for d in self.datas:
                if d not in ordered_datas:
                    ordered_datas.append(d)
        else:
            ordered_datas = list(self.datas)

        # Get stake amounts (default to 0 if not specified)
        stake_amounts = self.params.stake_amounts or {}
        txn_cost = self.params.transaction_cost

        # Track available cash within this bar for priority ordering
        available_cash = self.broker.getcash()

        for data in ordered_datas:
            ticker = data._name
            close_price = data.close[0]

            # Skip invalid prices
            if close_price is None or close_price <= 0:
                continue

            stake_amount = stake_amounts.get(ticker, 0)
            if stake_amount <= 0:
                continue

            # Look up signal for today
            signal = self.signals.get((cur_date, ticker), 0)
            if signal == 0:
                continue

            # Cancel any existing pending order for this ticker
            if ticker in self.pending_orders:
                existing = self.pending_orders[ticker]
                if existing and existing.alive():
                    self.cancel(existing)
                    del self.pending_orders[ticker]

            if signal == 1:
                # BUY: units = floor((stake_amount - txn_cost) / close)
                net_amount = stake_amount - txn_cost
                if net_amount <= 0:
                    continue
                units = math.floor(net_amount / close_price)
                if units <= 0:
                    continue

                # Check available cash
                cost = units * close_price + txn_cost
                if cost > available_cash:
                    self.order_log.append({
                        'date': cur_date.isoformat(),
                        'ticker': ticker,
                        'action': 'buy',
                        'signal': signal,
                        'order_price': close_price,
                        'order_units': units,
                        'status': 'Rejected',
                        'reason': 'Insufficient Cash',
                    })
                    continue

                order = self.buy(
                    data=data,
                    size=units,
                    exectype=bt.Order.Limit,
                    price=close_price,
                )
                self.pending_orders[ticker] = order
                available_cash -= cost  # Reserve cash for priority

                self.order_log.append({
                    'date': cur_date.isoformat(),
                    'ticker': ticker,
                    'action': 'buy',
                    'signal': signal,
                    'order_price': close_price,
                    'order_units': units,
                    'status': 'Submitted',
                    'reason': '',
                })

            elif signal == -1:
                # SELL: units = floor(stake_amount / close)
                units = math.floor(stake_amount / close_price)
                if units <= 0:
                    continue

                # Check we have enough shares
                position = self.getposition(data).size
                if position < units:
                    if position <= 0:
                        self.order_log.append({
                            'date': cur_date.isoformat(),
                            'ticker': ticker,
                            'action': 'sell',
                            'signal': signal,
                            'order_price': close_price,
                            'order_units': units,
                            'status': 'Rejected',
                            'reason': 'Insufficient Stock',
                        })
                        continue
                    # Sell what we have
                    units = position

                order = self.sell(
                    data=data,
                    size=units,
                    exectype=bt.Order.Limit,
                    price=close_price,
                )
                self.pending_orders[ticker] = order

                self.order_log.append({
                    'date': cur_date.isoformat(),
                    'ticker': ticker,
                    'action': 'sell',
                    'signal': signal,
                    'order_price': close_price,
                    'order_units': units,
                    'status': 'Submitted',
                    'reason': '',
                })

    # -----------------------------------------------------------------
    # Order / trade notifications
    # -----------------------------------------------------------------
    def notify_order(self, order):
        """Handle order status changes."""
        ticker = order.data._name

        if order.status in [order.Completed]:
            fill_date = bt.num2date(order.executed.dt).date().isoformat()
            fill_price = order.executed.price
            fill_size = order.executed.size

            # Update order log
            for entry in reversed(self.order_log):
                if (entry['ticker'] == ticker and
                        entry['status'] == 'Submitted'):
                    entry['status'] = 'Filled'
                    entry['fill_date'] = fill_date
                    entry['fill_price'] = fill_price
                    entry['fill_size'] = abs(fill_size)
                    break

            # Track open position entries for P&L
            if order.isbuy():
                if ticker not in self._open_positions:
                    self._open_positions[ticker] = []
                self._open_positions[ticker].append({
                    'price': fill_price,
                    'size': abs(fill_size),
                    'date': fill_date,
                })
                self.trades.append({
                    'type': 'open',
                    'ticker': ticker,
                    'date': fill_date,
                    'price': fill_price,
                    'size': abs(fill_size),
                    'action': 'buy',
                })
            else:
                # Sell — match against open positions (FIFO)
                pnl = 0.0
                remaining = abs(fill_size)
                entries = self._open_positions.get(ticker, [])
                while remaining > 0 and entries:
                    entry = entries[0]
                    matched = min(remaining, entry['size'])
                    pnl += matched * (fill_price - entry['price'])
                    entry['size'] -= matched
                    remaining -= matched
                    if entry['size'] <= 0:
                        entries.pop(0)

                self.trades.append({
                    'type': 'close',
                    'ticker': ticker,
                    'date': fill_date,
                    'price': fill_price,
                    'size': abs(fill_size),
                    'pnl': pnl,
                    'action': 'sell',
                })

            # Clear pending
            if ticker in self.pending_orders:
                del self.pending_orders[ticker]

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            reason_map = {
                order.Canceled: 'Canceled',
                order.Margin: 'Margin',
                order.Rejected: 'Rejected',
            }
            reason = reason_map.get(order.status, 'Unknown')
            for entry in reversed(self.order_log):
                if (entry['ticker'] == ticker and
                        entry['status'] == 'Submitted'):
                    entry['status'] = reason
                    break

            if ticker in self.pending_orders:
                del self.pending_orders[ticker]
