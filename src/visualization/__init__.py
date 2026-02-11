#!/usr/bin/env python3
"""
Unified visualization package for the trading strategy backtester.

Consolidates all plotting and charting functionality:
- backtest_charts: Candlestick, equity curve, trade markers, performance dashboard
- monte_carlo: Monte Carlo simulation visualizations
- performance: Tearsheet, heatmaps, equity curves (matplotlib)
- utils: Shared color schemes, layout templates, helpers
"""
from visualization.utils import COLORS, PLOTLY_LAYOUT, get_plotly_layout
