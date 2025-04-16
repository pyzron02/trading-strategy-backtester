#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple workflow runner for backtesting trading strategies.

This module provides a convenience function for running a complete
backtesting workflow with minimal configuration.
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

# Add parent directory to path to enable importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
if src_dir not in sys.path:
    sys.path.append(src_dir)

from engine.run_backtest import run_backtest
from utils.ensure_directories import ensure_output_directory


def run_simple_workflow(
    strategy_name: str,
    tickers: Union[str, List[str]],
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
    parameters: Optional[Dict[str, Any]] = None,
    initial_capital: float = 100000.0,
    commission: float = 0.001,
    output_dir: Optional[str] = None,
    data_dir: str = "input",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a complete backtesting workflow with minimal configuration.
    
    Args:
        strategy_name: Name of the strategy to backtest
        tickers: Ticker symbol(s) for the backtest
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        parameters: Optional strategy parameters
        initial_capital: Initial capital for the backtest
        commission: Commission rate for trades
        output_dir: Custom output directory (created if not specified)
        data_dir: Directory containing input data
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary containing backtest results
    """
    # Convert single ticker to list if needed
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # Create output directory if not specified
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", f"{strategy_name}_{timestamp}")
    
    # Ensure output directory exists
    ensure_output_directory(output_dir)
    
    # Run backtest
    if verbose:
        print(f"Running backtest for {strategy_name} on {tickers} from {start_date} to {end_date}")
        print(f"Output directory: {output_dir}")
        
    result = run_backtest(
        strategy_name=strategy_name,
        parameters=parameters,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission=commission,
        output_dir=output_dir,
        data_dir=data_dir,
        verbose=verbose
    )
    
    if verbose:
        print(f"Backtest completed with status: {result.get('status', 'unknown')}")
        if 'metrics' in result:
            print("\nPerformance Metrics:")
            for key, value in result['metrics'].items():
                print(f"  {key}: {value}")
    
    return result


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a simple backtesting workflow")
    parser.add_argument("--strategy", required=True, help="Strategy name")
    parser.add_argument("--tickers", required=True, help="Comma-separated list of tickers")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Run workflow
    run_simple_workflow(
        strategy_name=args.strategy,
        tickers=args.tickers.split(","),
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        commission=args.commission,
        output_dir=args.output_dir,
        verbose=args.verbose
    ) 