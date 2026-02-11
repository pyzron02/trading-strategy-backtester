#!/usr/bin/env python3
"""Tests for engine/data_feeds.py (ValidatingCSVData and TradeLogger)."""
import pytest
import backtrader as bt

from engine.data_feeds import ValidatingCSVData, TradeLogger


class TestValidatingCSVDataClass:
    """Tests that ValidatingCSVData is a proper subclass of GenericCSVData."""

    def test_is_subclass_of_generic_csv(self):
        assert issubclass(ValidatingCSVData, bt.feeds.GenericCSVData)

    def test_has_load_method(self):
        assert hasattr(ValidatingCSVData, "_load")


class TestTradeLoggerClass:
    """Tests for the TradeLogger analyzer."""

    def test_is_subclass_of_analyzer(self):
        assert issubclass(TradeLogger, bt.Analyzer)

    def test_has_notify_trade(self):
        assert hasattr(TradeLogger, "notify_trade")

    def test_has_notify_order(self):
        assert hasattr(TradeLogger, "notify_order")

    def test_has_get_analysis(self):
        assert hasattr(TradeLogger, "get_analysis")


class TestTradeLoggerIntegration:
    """Basic integration test: run a trivial cerebro and verify TradeLogger."""

    @pytest.fixture
    def minimal_cerebro(self, tmp_path):
        """Create a minimal cerebro instance with synthetic data."""
        import pandas as pd
        import datetime

        # Create synthetic OHLCV data
        dates = pd.bdate_range(start="2023-01-02", periods=30)
        data = pd.DataFrame(
            {
                "Date": dates,
                "Open": range(100, 130),
                "High": range(101, 131),
                "Low": range(99, 129),
                "Close": range(100, 130),
                "Volume": [1000] * 30,
            }
        )
        csv_path = tmp_path / "test_data.csv"
        data.to_csv(csv_path, index=False)

        cerebro = bt.Cerebro()
        feed = bt.feeds.GenericCSVData(
            dataname=str(csv_path),
            dtformat="%Y-%m-%d",
            openinterest=-1,
        )
        cerebro.adddata(feed, name="TEST")
        cerebro.addanalyzer(TradeLogger, _name="trade_logger")
        return cerebro

    def test_trade_logger_runs_without_error(self, minimal_cerebro):
        """TradeLogger should not raise when cerebro runs (no strategy trades)."""
        results = minimal_cerebro.run()
        analyzer = results[0].analyzers.trade_logger
        trades = analyzer.get_analysis()
        assert isinstance(trades, list)
        # No trades expected from the default strategy
        assert len(trades) == 0
