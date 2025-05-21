#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Strategy Package.

This package contains the trading strategies used by the backtester.
"""

try:
    from strategies.registry import (
        register_strategy,
        get_strategy_class,
        get_strategy_version,
        get_registered_strategies
    )
except ModuleNotFoundError:
    from src.strategies.registry import (
        register_strategy,
        get_strategy_class,
        get_strategy_version,
        get_registered_strategies
    )

__all__ = [
    'register_strategy',
    'get_strategy_class',
    'get_strategy_version',
    'get_registered_strategies'
]