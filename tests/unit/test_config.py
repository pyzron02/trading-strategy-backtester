#!/usr/bin/env python3
"""Tests for WorkflowConfig dataclass."""
import pytest
from workflows.config import WorkflowConfig


class TestWorkflowConfig:
    def test_default_values(self):
        cfg = WorkflowConfig()
        assert cfg.initial_capital == 100000.0
        assert cfg.commission == 0.001
        assert cfg.n_trials == 50
        assert cfg.strategy_name is None

    def test_from_kwargs_basic(self):
        cfg = WorkflowConfig.from_kwargs(
            strategy_name="SimpleStock",
            tickers=["AAPL"],
            start_date="2020-01-01",
            end_date="2023-01-01",
        )
        assert cfg.strategy_name == "SimpleStock"
        assert cfg.tickers == ["AAPL"]

    def test_from_kwargs_strategy_alias(self):
        cfg = WorkflowConfig.from_kwargs(strategy="SimpleStock")
        assert cfg.strategy_name == "SimpleStock"

    def test_from_kwargs_unknown_keys_in_extra(self):
        cfg = WorkflowConfig.from_kwargs(
            strategy_name="Test",
            unknown_param="value",
            another_unknown=42,
        )
        assert cfg.extra["unknown_param"] == "value"
        assert cfg.extra["another_unknown"] == 42

    def test_normalize_tickers_string(self):
        cfg = WorkflowConfig(tickers="AAPL,GOOG,MSFT")
        cfg.normalize_tickers()
        assert cfg.tickers == ["AAPL", "GOOG", "MSFT"]

    def test_normalize_tickers_none(self):
        cfg = WorkflowConfig()
        cfg.normalize_tickers()
        assert cfg.tickers == ["SPY"]

    def test_normalize_tickers_list(self):
        cfg = WorkflowConfig(tickers=["AAPL", "GOOG"])
        cfg.normalize_tickers()
        assert cfg.tickers == ["AAPL", "GOOG"]

    def test_resolve_strategy_name(self):
        cfg = WorkflowConfig()
        cfg.resolve_strategy_name("MyStrategy")
        assert cfg.strategy_name == "MyStrategy"

    def test_resolve_strategy_name_no_override(self):
        cfg = WorkflowConfig(strategy_name="Original")
        cfg.resolve_strategy_name("Override")
        assert cfg.strategy_name == "Original"

    def test_ensure_output_dir(self, tmp_path):
        cfg = WorkflowConfig(strategy_name="TestStrategy")
        cfg.output_dir = str(tmp_path / "test_output")
        cfg.ensure_output_dir("simple")
        assert (tmp_path / "test_output").exists()
