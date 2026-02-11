#!/usr/bin/env python3
"""
results_serializer.py - Serialize and write backtest results to disk.

Provides functions for writing human-readable summary files and
serializing backtest results to JSON and pickle formats.
"""

import os
import json
import pickle
import pandas as pd
from typing import Any, Dict, List, Optional

from engine.serialization import CustomJSONEncoder


def write_summary_file(
    output_dir: str,
    metrics: Dict[str, Any],
    initial_value: float,
    final_value: float,
    strategy_name: str,
    tickers: List[str],
    start_date: str,
    end_date: str
) -> None:
    """
    Write a human-readable performance summary text file.

    Creates a 'results.txt' file in the specified output directory
    containing strategy information, performance summary, risk metrics,
    and trade statistics.

    Args:
        output_dir: Directory where the summary file will be saved.
        metrics: Dictionary of computed performance metrics.
        initial_value: Portfolio starting value.
        final_value: Portfolio ending value.
        strategy_name: Name of the strategy that was run.
        tickers: List of ticker symbols used in the backtest.
        start_date: Backtest start date string.
        end_date: Backtest end date string.
    """
    total_return = metrics.get('total_return', 0)
    benchmark_return = metrics.get('benchmark_return', 0)
    alpha = metrics.get('alpha', 0)
    parameters = metrics.get('parameters', {})

    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        # Strategy and test information
        f.write(f"Strategy: {strategy_name}\n")
        f.write(f"Parameters: {parameters}\n")
        f.write(f"Tickers: {tickers}\n")
        f.write(f"Period: {start_date} to {end_date}\n\n")

        # Performance summary
        f.write(f"==============================================================\n")
        f.write(f"PERFORMANCE SUMMARY\n")
        f.write(f"==============================================================\n")
        f.write(f"Initial Value: ${initial_value:.2f}\n")
        f.write(f"Final Value: ${final_value:.2f}\n")
        f.write(f"Absolute Return: ${final_value - initial_value:.2f}\n")
        f.write(f"Total Return: {total_return:.2%}\n")
        f.write(f"Benchmark Return: {benchmark_return:.2%}\n")
        f.write(f"Alpha: {alpha:.2%}\n")
        if 'annual_return' in metrics:
            f.write(f"Annual Return: {metrics['annual_return']:.2%}\n")
        f.write(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}\n")
        if metrics.get('win_rate', 0) > 0:
            f.write(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n")
        f.write("\n")

        # Risk metrics
        f.write(f"==============================================================\n")
        f.write(f"RISK METRICS\n")
        f.write(f"==============================================================\n")
        f.write(f"Maximum Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%\n")
        f.write(f"Maximum Drawdown (Money): ${metrics.get('max_drawdown_money', 0):.2f}\n")
        if metrics.get('sharpe_ratio', 0) > 0:
            calmar_ratio = metrics.get('annual_return', 0) / max(metrics.get('max_drawdown', 0.01), 0.01)
            f.write(f"Calmar Ratio: {calmar_ratio:.4f}\n")
            f.write(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}\n")
        f.write(f"Volatility (Annualized): {metrics.get('annualized_volatility', 0):.2%}\n\n")

        # Trade statistics
        f.write(f"==============================================================\n")
        f.write(f"TRADE STATISTICS\n")
        f.write(f"==============================================================\n")
        f.write(f"Total Trades: {metrics.get('total_trades', 0)}\n")
        if metrics.get('total_trades', 0) > 0:
            f.write(f"Winning Trades: {metrics.get('winning_trades', 0)} ({metrics.get('win_rate_pct', 0):.1f}%)\n")
            f.write(f"Losing Trades: {metrics.get('losing_trades', 0)} ({100 - metrics.get('win_rate_pct', 0):.1f}%)\n")
            f.write(f"Profit Factor: {metrics.get('profit_factor', 0):.4f}\n")
            f.write(f"Average Trade PnL: ${metrics.get('avg_trade_pnl', 0):.2f}\n")

            # More details for trades if we have them
            if metrics.get('gross_profit', 0) != 0 or metrics.get('gross_loss', 0) != 0:
                f.write(f"\nProfit & Loss:\n")
                f.write(f"  Gross Profit: ${metrics.get('gross_profit', 0):.2f}\n")
                f.write(f"  Gross Loss: ${metrics.get('gross_loss', 0):.2f}\n")
                f.write(f"  Net Profit: ${metrics.get('net_profit', 0):.2f}\n")

            if metrics.get('avg_win', 0) != 0 or metrics.get('avg_loss', 0) != 0:
                f.write(f"\nTrade Sizing:\n")
                f.write(f"  Average Win: ${metrics.get('avg_win', 0):.2f}\n")
                f.write(f"  Average Loss: ${metrics.get('avg_loss', 0):.2f}\n")
                win_loss_ratio = abs(metrics.get('avg_win', 0) / metrics.get('avg_loss', 1))
                f.write(f"  Win/Loss Ratio: {win_loss_ratio:.2f}\n")

            if metrics.get('max_consecutive_wins', 0) > 0 or metrics.get('max_consecutive_losses', 0) > 0:
                f.write(f"\nWin/Loss Streaks:\n")
                f.write(f"  Max Consecutive Wins: {metrics.get('max_consecutive_wins', 0)}\n")
                f.write(f"  Max Consecutive Losses: {metrics.get('max_consecutive_losses', 0)}\n")

            if metrics.get('avg_trade_length', 0) > 0:
                f.write(f"\nTrade Duration:\n")
                f.write(f"  Average Trade Length: {metrics.get('avg_trade_length', 0):.1f} bars\n")


def save_backtest_results(
    output_dir: str,
    result: Dict[str, Any],
    metrics: Dict[str, Any]
) -> None:
    """
    Serialize backtest results to JSON and pickle files.

    Creates 'backtest_results.json' and 'backtest_results.pkl' in the
    specified output directory.

    Args:
        output_dir: Directory where the result files will be saved.
        result: Full backtest result dictionary (may contain DataFrames).
        metrics: Dictionary of computed performance metrics (unused
            directly but kept for interface consistency with callers
            that may extend serialization).
    """
    # Save detailed results to JSON
    json_file = os.path.join(output_dir, 'backtest_results.json')
    with open(json_file, 'w') as f:
        # Convert non-serializable objects to strings or other
        # serializable types
        serializable_results = result.copy()
        if 'equity_curve' in serializable_results:
            serializable_results['equity_curve'] = (
                serializable_results['equity_curve'].to_dict('records')
                if serializable_results['equity_curve'] is not None
                else None
            )
        if 'trade_log' in serializable_results:
            serializable_results['trade_log'] = (
                serializable_results['trade_log'].to_dict('records')
                if serializable_results['trade_log'] is not None
                else None
            )

        json.dump(serializable_results, f, indent=4, cls=CustomJSONEncoder)

    # Save to pickle (can store more complex objects)
    pickle_file = os.path.join(output_dir, 'backtest_results.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(result, f)
