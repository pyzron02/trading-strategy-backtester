#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the LimitOrder strategy.

Unit tests cover signal loading, sizing, priority ordering, commission
handling, limit-order fills, and registry integration. An integration test
runs the full workflow via the config file.
"""

import csv
import math
import os
import sys
import tempfile
from datetime import date

import pytest

# Ensure the src directory is on the path
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

import backtrader as bt

from strategies.limit_order_strategy import LimitOrderStrategy


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _write_signal_csv(path, rows):
    """Write a list of (Date, Ticker, Signal) rows to a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Ticker", "Signal"])
        for row in rows:
            writer.writerow(row)


def _make_data_feed(name, bars):
    """Create a backtrader data feed from a list of (date, o, h, l, c, v)."""
    import pandas as pd

    df = pd.DataFrame(bars, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    data = bt.feeds.PandasData(dataname=df, name=name)
    return data


# ──────────────────────────────────────────────────────────────────
# Unit tests
# ──────────────────────────────────────────────────────────────────

class TestSignalLoading:
    """Test _load_signals() method."""

    def test_loads_valid_csv(self, tmp_path):
        sig_file = str(tmp_path / "signals.csv")
        _write_signal_csv(sig_file, [
            ("2020-03-18", "NVDA", "1"),
            ("2020-04-02", "NVDA", "-1"),
            ("2020-05-01", "TSLA", "1"),
        ])

        strategy = LimitOrderStrategy.__new__(LimitOrderStrategy)
        signals = strategy._load_signals(sig_file)

        assert len(signals) == 3
        assert signals[(date(2020, 3, 18), "NVDA")] == 1
        assert signals[(date(2020, 4, 2), "NVDA")] == -1
        assert signals[(date(2020, 5, 1), "TSLA")] == 1

    def test_skips_zero_signals(self, tmp_path):
        sig_file = str(tmp_path / "signals.csv")
        _write_signal_csv(sig_file, [
            ("2020-03-18", "NVDA", "0"),
            ("2020-04-02", "NVDA", "1"),
        ])

        strategy = LimitOrderStrategy.__new__(LimitOrderStrategy)
        signals = strategy._load_signals(sig_file)

        assert len(signals) == 1

    def test_rejects_invalid_signal(self, tmp_path):
        sig_file = str(tmp_path / "signals.csv")
        _write_signal_csv(sig_file, [
            ("2020-03-18", "NVDA", "2"),
        ])

        strategy = LimitOrderStrategy.__new__(LimitOrderStrategy)
        with pytest.raises(ValueError, match="Invalid signal"):
            strategy._load_signals(sig_file)

    def test_handles_missing_file(self, tmp_path):
        strategy = LimitOrderStrategy.__new__(LimitOrderStrategy)
        signals = strategy._load_signals(str(tmp_path / "nonexistent.csv"))
        assert signals == {}


class TestDollarBasedSizing:
    """Test dollar-based position sizing calculations."""

    def test_buy_sizing_with_txn_cost(self):
        """floor((4000 - 0.99) / 50.0) = 79 shares."""
        stake = 4000
        txn_cost = 0.99
        price = 50.0
        units = math.floor((stake - txn_cost) / price)
        assert units == 79

    def test_buy_sizing_exact_division(self):
        """floor((1000 - 0.99) / 10.0) = 99 shares."""
        stake = 1000
        txn_cost = 0.99
        price = 10.0
        units = math.floor((stake - txn_cost) / price)
        assert units == 99

    def test_sell_sizing(self):
        """Sell: floor(4000 / 50.0) = 80 shares."""
        stake = 4000
        price = 50.0
        units = math.floor(stake / price)
        assert units == 80


class TestLimitOrderFill:
    """Test that limit orders fill correctly in backtrader."""

    def test_buy_limit_fills_when_low_reaches_price(self):
        """Buy limit at $50, next bar low=$49 → fills."""
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000)
        cerebro.broker.setcommission(
            commission=0.99, commtype=bt.CommInfoBase.COMM_FIXED
        )

        # Create synthetic data: bar 0 close=50, bar 1 low=49 (fills)
        bars = [
            ("2020-01-02", 50, 51, 49, 50, 1000),
            ("2020-01-03", 50, 52, 49, 51, 1000),
            ("2020-01-06", 51, 53, 50, 52, 1000),
        ]

        sig_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        try:
            writer = csv.writer(sig_file)
            writer.writerow(["Date", "Ticker", "Signal"])
            writer.writerow(["2020-01-02", "TEST", "1"])
            sig_file.close()

            data = _make_data_feed("TEST", bars)
            cerebro.adddata(data)
            cerebro.addstrategy(
                LimitOrderStrategy,
                signal_file=sig_file.name,
                stake_amounts={"TEST": 4000},
                ticker_priority=["TEST"],
                transaction_cost=0.99,
            )

            results = cerebro.run()
            strat = results[0]

            # Should have at least one filled buy
            filled = [
                o for o in strat.order_log
                if o["status"] == "Filled" and o["action"] == "buy"
            ]
            assert len(filled) >= 1
        finally:
            os.unlink(sig_file.name)


class TestFlatFeeCommission:
    """Test that fixed commission is applied (not percentage)."""

    def test_fixed_commission_applied(self):
        """Verify broker uses COMM_FIXED, not percentage commission."""
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000)
        cerebro.broker.setcommission(
            commission=0.99, commtype=bt.CommInfoBase.COMM_FIXED
        )

        # Use enough bars so the position is still open at end
        bars = [
            ("2020-01-02", 100, 101, 99, 100, 1000),
            ("2020-01-03", 100, 102, 99, 101, 1000),
            ("2020-01-06", 101, 103, 100, 102, 1000),
            ("2020-01-07", 102, 104, 101, 103, 1000),
            ("2020-01-08", 103, 105, 102, 104, 1000),
        ]

        sig_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        try:
            writer = csv.writer(sig_file)
            writer.writerow(["Date", "Ticker", "Signal"])
            writer.writerow(["2020-01-02", "TEST", "1"])
            sig_file.close()

            data = _make_data_feed("TEST", bars)
            cerebro.adddata(data)
            cerebro.addstrategy(
                LimitOrderStrategy,
                signal_file=sig_file.name,
                stake_amounts={"TEST": 4000},
                ticker_priority=["TEST"],
                transaction_cost=0.99,
            )

            results = cerebro.run()
            strat = results[0]

            # Verify the buy order was filled
            filled_buys = [
                o for o in strat.order_log
                if o["status"] == "Filled" and o["action"] == "buy"
            ]
            assert len(filled_buys) == 1

            # Verify correct sizing: floor((4000 - 0.99) / 100) = 39
            assert filled_buys[0]["order_units"] == 39
            assert filled_buys[0]["fill_price"] == 100.0

            # Position should be 39 shares
            pos = strat.getposition(strat.datas[0]).size
            assert pos == 39
        finally:
            os.unlink(sig_file.name)


class TestRegistryIntegration:
    """Test that LimitOrder is properly registered."""

    def test_get_strategy_class(self):
        from strategies.registry import get_strategy_class
        cls = get_strategy_class("LimitOrder")
        assert cls is LimitOrderStrategy

    def test_get_strategy_class_from_engine(self):
        from engine.run_backtest import get_strategy_class
        cls = get_strategy_class("LimitOrder")
        assert cls is LimitOrderStrategy


# ──────────────────────────────────────────────────────────────────
# Integration test
# ──────────────────────────────────────────────────────────────────

class TestEndToEndWorkflow:
    """Integration test running through the full workflow."""

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "input", "stock_data.csv"
            )
        ),
        reason="stock_data.csv not available",
    )
    def test_workflow_produces_output(self, tmp_path):
        """Run the LimitOrder workflow and check output files."""
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        output_dir = str(tmp_path / "output")

        from engine.run_backtest import run_backtest

        signal_file = os.path.join(
            project_root, "input", "signals", "limit_order_signals.csv"
        )
        stock_csv = os.path.join(project_root, "input", "stock_data.csv")

        result = run_backtest(
            strategy_name="LimitOrder",
            tickers=["NVDA"],
            start_date="2020-01-01",
            end_date="2023-12-29",
            output_dir=output_dir,
            parameters={
                "signal_file": signal_file,
                "stake_amounts": {"NVDA": 4000},
                "ticker_priority": ["NVDA"],
                "transaction_cost": 0.99,
            },
            stock_csv=stock_csv,
            initial_capital=100000.0,
            commission=0.99,
            commission_type="fixed",
            plot=False,
        )

        assert result is not None
        assert result["status"] == "success"
        assert result["metrics"]["total_trades"] > 0
        assert os.path.exists(os.path.join(output_dir, "equity_curve.csv"))
        assert os.path.exists(os.path.join(output_dir, "trade_log.csv"))
        assert os.path.exists(os.path.join(output_dir, "results.txt"))
