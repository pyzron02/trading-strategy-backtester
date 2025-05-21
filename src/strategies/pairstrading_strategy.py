#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pairs Trading Strategy Implementation Alias.

This module is an alias for pairs_trading_strategy.py to help the framework locate it.
"""

# Import the real strategy class
try:
    from src.strategies.pairs_trading_strategy import PairsTradingStrategy as RealStrategy
except ImportError:
    try:
        from strategies.pairs_trading_strategy import PairsTradingStrategy as RealStrategy
    except ImportError:
        try:
            # Same directory import
            from pairs_trading_strategy import PairsTradingStrategy as RealStrategy
        except ImportError:
            from .pairs_trading_strategy import PairsTradingStrategy as RealStrategy

# Aliases for the strategy class to help the dynamic import mechanism find it
PairstradingStrategy = RealStrategy
PairsTradingStrategy = RealStrategy