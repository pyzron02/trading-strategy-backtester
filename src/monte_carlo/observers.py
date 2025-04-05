#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Observers and Analyzers for the Monte Carlo backtesting framework.
"""

import backtrader as bt


class PortfolioValue(bt.Observer):
    """Observer that tracks portfolio value throughout the backtest"""
    lines = ('value',)
    
    def next(self):
        self.lines.value[0] = self._owner.broker.getvalue()


class TradeLog(bt.Analyzer):
    """Analyzer that logs all trades during a backtest"""
    
    def __init__(self):
        self.log = []
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.log.append({
                'date': bt.num2date(trade.dtclose).strftime('%Y-%m-%d'),
                'type': 'SELL',
                'price': trade.price,
                'size': trade.size,
                'value': trade.price * trade.size,
                'pnl': trade.pnl,
                'commission': trade.commission
            })
        elif trade.justopened:
            self.log.append({
                'date': bt.num2date(trade.dtopen).strftime('%Y-%m-%d'),
                'type': 'BUY',
                'price': trade.price,
                'size': trade.size,
                'value': trade.price * trade.size,
                'pnl': 0.0,
                'commission': trade.commission
            }) 