#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Limit Order Validation Report Generator.

Generates per-ticker and portfolio-level CSV reports that can be compared
against the reference Excel spreadsheet for validation.
"""

import os
import pandas as pd
from typing import Dict, List, Any, Optional


def generate_ticker_report(
    ticker: str,
    order_log: List[Dict],
    ohlcv_df: pd.DataFrame,
    stake_amount: float,
    txn_cost: float,
) -> pd.DataFrame:
    """Generate a per-ticker report matching the Excel's column structure.

    Args:
        ticker: Ticker symbol.
        order_log: List of order-log dicts from the strategy.
        ohlcv_df: DataFrame with Date, Open, High, Low, Close, Volume columns.
        stake_amount: Dollar amount per trade for this ticker.
        txn_cost: Flat transaction cost per trade.

    Returns:
        DataFrame with columns matching the Excel's per-ticker sheet.
    """
    # Filter order log for this ticker
    ticker_orders = [o for o in order_log if o.get('ticker') == ticker]

    # Build an order lookup by date
    order_by_date = {}
    for o in ticker_orders:
        order_by_date[o['date']] = o

    rows = []
    for _, bar in ohlcv_df.iterrows():
        date_str = bar['Date']
        row = {
            'Date': date_str,
            'Open': bar.get('Open', 0),
            'High': bar.get('High', 0),
            'Low': bar.get('Low', 0),
            'Close': bar.get('Close', 0),
            'Volume': bar.get('Volume', 0),
        }

        order = order_by_date.get(date_str)
        if order:
            row['Signal'] = order.get('signal', 0)
            row['Stake_Amount'] = stake_amount
            row['Order_Price'] = order.get('order_price', 0)
            row['Order_Units'] = order.get('order_units', 0)
            row['Order_Status'] = order.get('status', '')
            row['Fill_Date'] = order.get('fill_date', '')
            row['Fill_Price'] = order.get('fill_price', '')
        else:
            row['Signal'] = 0
            row['Stake_Amount'] = 0
            row['Order_Price'] = 0
            row['Order_Units'] = 0
            row['Order_Status'] = ''
            row['Fill_Date'] = ''
            row['Fill_Price'] = ''

        rows.append(row)

    return pd.DataFrame(rows)


def generate_portfolio_report(
    equity_curve: List[Dict],
    initial_capital: float,
) -> pd.DataFrame:
    """Generate a portfolio-level report.

    Args:
        equity_curve: List of {Date, Value} dicts from the strategy.
        initial_capital: Starting capital.

    Returns:
        DataFrame with portfolio-level columns.
    """
    if not equity_curve:
        return pd.DataFrame()

    df = pd.DataFrame(equity_curve)
    df['Portfolio_Value'] = df['Value']
    df['Portfolio_Returns'] = df['Value'].pct_change().fillna(0)
    return df[['Date', 'Portfolio_Value', 'Portfolio_Returns']]


def save_validation_reports(
    output_dir: str,
    strategy_instance: Any,
    tickers: List[str],
    stake_amounts: Dict[str, float],
    txn_cost: float,
    initial_capital: float,
    ohlcv_data: Optional[Dict[str, pd.DataFrame]] = None,
):
    """Save all validation report CSVs to the output directory.

    Args:
        output_dir: Directory to write CSVs into.
        strategy_instance: The LimitOrderStrategy instance after backtesting.
        tickers: List of ticker symbols.
        stake_amounts: Dict mapping ticker -> dollar amount.
        txn_cost: Flat transaction cost.
        initial_capital: Starting capital.
        ohlcv_data: Optional dict of {ticker: DataFrame} with OHLCV data.
    """
    report_dir = os.path.join(output_dir, 'validation_reports')
    os.makedirs(report_dir, exist_ok=True)

    order_log = getattr(strategy_instance, 'order_log', [])
    equity_curve = getattr(strategy_instance, 'equity_curve', [])

    # Per-ticker reports (only if OHLCV data is provided)
    if ohlcv_data:
        for ticker in tickers:
            if ticker not in ohlcv_data:
                continue
            ticker_df = generate_ticker_report(
                ticker=ticker,
                order_log=order_log,
                ohlcv_df=ohlcv_data[ticker],
                stake_amount=stake_amounts.get(ticker, 0),
                txn_cost=txn_cost,
            )
            out_path = os.path.join(report_dir, f'{ticker}_report.csv')
            ticker_df.to_csv(out_path, index=False)
            print(f"[LimitOrder] Saved ticker report: {out_path}")

    # Portfolio report
    portfolio_df = generate_portfolio_report(equity_curve, initial_capital)
    if not portfolio_df.empty:
        out_path = os.path.join(report_dir, 'portfolio_report.csv')
        portfolio_df.to_csv(out_path, index=False)
        print(f"[LimitOrder] Saved portfolio report: {out_path}")

    # Order log
    if order_log:
        order_df = pd.DataFrame(order_log)
        out_path = os.path.join(report_dir, 'order_log.csv')
        order_df.to_csv(out_path, index=False)
        print(f"[LimitOrder] Saved order log: {out_path}")

    # Trade log
    trades = getattr(strategy_instance, 'trades', [])
    if trades:
        trades_df = pd.DataFrame(trades)
        out_path = os.path.join(report_dir, 'trade_log.csv')
        trades_df.to_csv(out_path, index=False)
        print(f"[LimitOrder] Saved trade log: {out_path}")
