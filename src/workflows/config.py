#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow configuration dataclass for the trading strategy backtester.

Provides a single config object that replaces 15-27 keyword arguments
across all workflow functions.
"""
import os
import datetime
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from utils.path_manager import path_manager


@dataclass
class WorkflowConfig:
    """Configuration for all workflow types.

    Common parameters shared by every workflow are required or have sensible
    defaults.  Workflow-specific parameters live in dedicated sections below.
    """

    # ── Core (required for every workflow) ──────────────────────────
    strategy_name: str = None
    tickers: List[str] = None
    start_date: str = None
    end_date: str = None

    # ── Output / IO ─────────────────────────────────────────────────
    output_dir: Optional[str] = None
    data_dir: str = "input"
    stock_csv: Optional[str] = None
    verbose: bool = False

    # ── Capital / Costs ─────────────────────────────────────────────
    initial_capital: float = 100000.0
    commission: float = 0.001
    commission_type: str = "percentage"  # "percentage" or "fixed"
    slippage: float = 0.0

    # ── Strategy parameters ─────────────────────────────────────────
    parameters: Optional[Dict[str, Any]] = None
    param_file: Optional[str] = None

    # ── Optimization ────────────────────────────────────────────────
    n_trials: int = 50
    optimization_metric: str = "sharpe_ratio"
    max_combinations: Optional[int] = None

    # ── Monte Carlo ─────────────────────────────────────────────────
    n_simulations: int = 100
    keep_permuted_data: bool = False
    analyze_only: bool = False
    backtest_result: Optional[Any] = None
    confidence_level: float = 0.95
    bootstrap_pct: float = 0.5
    random_seed: Optional[int] = None

    # ── Walk-forward ────────────────────────────────────────────────
    window_size: int = 252
    step_size: int = 63
    reoptimize: str = "always"
    reoptimization_threshold: float = 0.05

    # ── Simple workflow extras ──────────────────────────────────────
    optimize_sharpe: bool = False
    live_mode: bool = False
    additional_data: Optional[Dict[str, Any]] = None
    force_download: bool = False

    # ── Progress / Frontend ─────────────────────────────────────────
    progress_callback: Optional[Any] = None
    progress_file: Optional[str] = None

    # ── Internal bookkeeping ────────────────────────────────────────
    _temp_files_to_cleanup: List[str] = field(default_factory=list)

    # ── Extra kwargs (catch-all) ────────────────────────────────────
    extra: Dict[str, Any] = field(default_factory=dict)

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def resolve_strategy_name(self, strategy_alias: Optional[str] = None):
        """Resolve the `strategy` / `strategy_name` aliasing.

        Accepts the old `strategy` kwarg as *strategy_alias* and stores
        the result in ``self.strategy_name``.
        """
        if strategy_alias is not None and self.strategy_name is None:
            self.strategy_name = strategy_alias

    def normalize_tickers(self):
        """Ensure tickers is a list of strings."""
        if self.tickers is None:
            self.tickers = ["SPY"]
        elif isinstance(self.tickers, str):
            self.tickers = [t.strip() for t in self.tickers.split(",") if t.strip()]

    def ensure_output_dir(self, workflow_type: str):
        """Create a unique output directory if none was given."""
        if not self.output_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = str(uuid.uuid4())[:8]
            self.output_dir = os.path.join(
                str(path_manager.output_dir),
                f"{self.strategy_name}_{workflow_type}_{timestamp}_{run_id}",
            )
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def from_kwargs(cls, **kwargs) -> "WorkflowConfig":
        """Build a WorkflowConfig from a flat keyword dict.

        Unknown keys are stored in ``extra`` so nothing is silently dropped.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        init_args = {k: v for k, v in kwargs.items() if k in known}
        extra = {k: v for k, v in kwargs.items() if k not in known}

        # Handle strategy / strategy_name alias
        if "strategy" in extra:
            if init_args.get("strategy_name") is None:
                init_args["strategy_name"] = extra.pop("strategy")
            else:
                extra.pop("strategy", None)

        cfg = cls(**init_args)
        cfg.extra = extra
        return cfg
