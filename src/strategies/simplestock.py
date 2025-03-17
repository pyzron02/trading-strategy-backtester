#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimpleStock strategy implementation.

This module provides the SimpleStock strategy for the backtester.
"""

from strategies.simple_stock_strategy import SimpleStockStrategy

# Expose SimpleStockStrategy as SimpleStock for backward compatibility
SimpleStock = SimpleStockStrategy 