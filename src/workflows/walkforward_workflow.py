#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-forward analysis workflow module.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
project_root = os.path.dirname(src_dir)  # Go up to project root
if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the utilities
from workflows.workflow_utils import (
    print_header, print_section, print_parameters, print_metrics,
    save_results_summary, time_execution, find_strategy_param_file,
    logger, logging_system, print_workflow_log
)

# Import engine components
from engine.testing import WalkForwardTest
from workflows.simple_workflow import ensure_data_available

@time_execution("walkforward workflow")
def run_walkforward_workflow(
    strategy_name: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_dir: str,
    parameters: Optional[Dict[str, Any]] = None,
    param_file: Optional[str] = None,
    window_size: int = 252,
    step_size: int = 63,
    n_trials: int = 50,
    optimization_metric: str = "sharpe_ratio",
    verbose: bool = False,
    initial_capital: float = 100000.0,
    commission: float = 0.001,
    data_dir: str = "input"
) -> Dict[str, Any]:
    """
    Run a walk-forward analysis workflow for the given strategy.
    
    Args:
        strategy_name: Name of the strategy to run
        tickers: List of ticker symbols
        start_date: Start date for backtest in YYYY-MM-DD format
        end_date: End date for backtest in YYYY-MM-DD format
        output_dir: Directory to save results
        parameters: Dictionary of strategy parameters (overrides param_file)
        param_file: File with parameter definitions
        window_size: Size of the in-sample window in trading days
        step_size: Size of the out-of-sample window in trading days
        n_trials: Number of optimization trials for each in-sample period
        optimization_metric: Metric to optimize for
        verbose: Whether to print detailed output
        initial_capital: Initial capital for backtest
        commission: Commission rate for trades
        data_dir: Directory containing input data
    
    Returns:
        Dict containing the workflow results
    """
    # Log workflow start
    additional_info = {
        "output_dir": output_dir,
        "window_size": window_size,
        "step_size": step_size,
        "optimization_metric": optimization_metric,
        "n_trials": n_trials
    }
    print_workflow_log(
        workflow_name="Walk-Forward Analysis Workflow",
        strategy_name=strategy_name,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        status="STARTED",
        additional_info=additional_info
    )
    
    print_header(f"Walk-Forward Analysis: {strategy_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set logging level based on verbose flag
    if verbose:
        logging_system.set_level('DEBUG', 'workflows')
    
    # Find parameter grid file
    param_grid_file = os.path.join(
        project_root, "input", "parameter_grids", f"{strategy_name}_grid.json"
    )
    if not os.path.exists(param_grid_file):
        logger.error(f"Parameter grid file not found for {strategy_name}")
        return {"status": "error", "message": "Parameter grid file not found"}
    
    print_section("Running Walk-Forward Analysis")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Tickers: {', '.join(tickers)}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Window size: {window_size} days")
    logger.info(f"Step size: {step_size} days")
    logger.info(f"Optimization metric: {optimization_metric}")
    logger.info(f"Trials per window: {n_trials}")
    
    try:
        # Ensure stock_data.csv is available and contains required tickers
        stock_csv = ensure_data_available(tickers, start_date, end_date, data_dir)
        logger.info(f"Using stock data from: {stock_csv}")
        
        # Create walk-forward analyzer
        walkforward = WalkForwardAnalysis(
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            param_space=parameters,
            optimization_metric=optimization_metric,
            window_size=window_size,
            step_size=step_size,
            n_trials=n_trials,
            initial_capital=initial_capital,
            commission=commission,
            output_dir=output_dir,
            data_dir=data_dir,
            stock_csv=stock_csv,  # Pass the explicit stock_csv path
            verbose=verbose
        )
        
        # Run walk-forward analysis
        results = walkforward.run_analysis()
        
        # Extract and process results
        period_results = []
        combined_equity_curve = pd.DataFrame()
        combined_trade_log = pd.DataFrame()
        
        for period in results["periods"]:
            # Add period results
            period_results.append({
                "in_sample": {
                    "start_date": period["in_sample_start"],
                    "end_date": period["in_sample_end"]
                },
                "out_sample": {
                    "start_date": period["out_sample_start"],
                    "end_date": period["out_sample_end"],
                    "metrics": period["out_sample_metrics"]
                },
                "best_params": period["best_params"]
            })
            
            # Append equity curve and trade log
            if "out_sample_equity" in period and not period["out_sample_equity"].empty:
                if combined_equity_curve.empty:
                    combined_equity_curve = period["out_sample_equity"]
                else:
                    # Append making sure there's no overlap
                    last_date = combined_equity_curve.index[-1]
                    out_sample_curve = period["out_sample_equity"]
                    if out_sample_curve.index[0] <= last_date:
                        out_sample_curve = out_sample_curve[out_sample_curve.index > last_date]
                    
                    if not out_sample_curve.empty:
                        combined_equity_curve = pd.concat([combined_equity_curve, out_sample_curve])
            
            if "out_sample_trades" in period and not period["out_sample_trades"].empty:
                if combined_trade_log.empty:
                    combined_trade_log = period["out_sample_trades"]
                else:
                    combined_trade_log = pd.concat([combined_trade_log, period["out_sample_trades"]])
        
        # Calculate overall performance metrics from combined equity curve
        overall_metrics = {}
        if not combined_equity_curve.empty:
            # Basic metrics
            returns = combined_equity_curve.pct_change().dropna()
            
            try:
                total_return = (combined_equity_curve.iloc[-1] / combined_equity_curve.iloc[0]) - 1
                overall_metrics["total_return"] = total_return if np.isfinite(total_return) else 0.0
            except (IndexError, ZeroDivisionError) as e:
                logger.warning(f"Could not calculate total_return: {e}")
                overall_metrics["total_return"] = 0.0
            
            try:
                # Safely calculate annualized return
                days = len(combined_equity_curve)
                if days > 0:
                    annualized_return = (1 + overall_metrics["total_return"]) ** (252 / days) - 1
                    overall_metrics["annualized_return"] = annualized_return if np.isfinite(annualized_return) else 0.0
                else:
                    overall_metrics["annualized_return"] = 0.0
            except Exception as e:
                logger.warning(f"Could not calculate annualized_return: {e}")
                overall_metrics["annualized_return"] = 0.0
            
            try:
                # Safely calculate Sharpe ratio
                std = returns.std()
                if std > 0:
                    sharpe_ratio = returns.mean() / std * np.sqrt(252)
                    overall_metrics["sharpe_ratio"] = sharpe_ratio if np.isfinite(sharpe_ratio) else 0.0
                else:
                    # Can't divide by zero standard deviation
                    overall_metrics["sharpe_ratio"] = 0.0
            except Exception as e:
                logger.warning(f"Could not calculate sharpe_ratio: {e}")
                overall_metrics["sharpe_ratio"] = 0.0
            
            try:
                # Calculate drawdown
                equity_series = combined_equity_curve.iloc[:, 0]
                rolling_max = equity_series.cummax()
                drawdown = (equity_series / rolling_max) - 1
                max_dd = drawdown.min()
                overall_metrics["max_drawdown"] = max_dd if np.isfinite(max_dd) else 0.0
            except Exception as e:
                logger.warning(f"Could not calculate max_drawdown: {e}")
                overall_metrics["max_drawdown"] = 0.0
        
        # Calculate trade statistics from combined trade log
        if not combined_trade_log.empty:
            overall_metrics["total_trades"] = len(combined_trade_log)
            
            try:
                winning_trades = combined_trade_log[combined_trade_log["profit"] > 0]
                overall_metrics["win_rate"] = len(winning_trades) / len(combined_trade_log) if len(combined_trade_log) > 0 else 0.0
            except Exception as e:
                logger.warning(f"Could not calculate win_rate: {e}")
                overall_metrics["win_rate"] = 0.0
                
            try:
                losing_trades = combined_trade_log[combined_trade_log["profit"] <= 0]
                losing_sum = losing_trades["profit"].sum()
                winning_sum = winning_trades["profit"].sum()
                
                # Handle division by zero for profit factor
                if losing_sum != 0 and losing_sum < 0:
                    profit_factor = abs(winning_sum / losing_sum)
                    overall_metrics["profit_factor"] = profit_factor if np.isfinite(profit_factor) else 999.99
                elif winning_sum > 0:
                    # No losing trades but winning trades exist
                    overall_metrics["profit_factor"] = 999.99  # Special value indicating no losing trades
                    logger.info("Profit factor is infinity (no losing trades). Using value 999.99")
                else:
                    # No winning trades or both are zero
                    overall_metrics["profit_factor"] = 0.0
            except Exception as e:
                logger.warning(f"Could not calculate profit_factor: {e}")
                overall_metrics["profit_factor"] = 0.0
                
            try:
                avg_profit = combined_trade_log["profit"].mean()
                overall_metrics["avg_profit"] = avg_profit if np.isfinite(avg_profit) else 0.0
            except Exception as e:
                logger.warning(f"Could not calculate avg_profit: {e}")
                overall_metrics["avg_profit"] = 0.0
        
        # Final results
        results = {
            "strategy_name": strategy_name,
            "dates": {
                "start_date": start_date,
                "end_date": end_date
            },
            "walkforward_params": {
                "window_size": window_size,
                "step_size": step_size,
                "n_trials": n_trials,
                "optimization_metric": optimization_metric
            },
            "overall_metrics": overall_metrics,
            "period_results": period_results
        }
        
        print_section("Walk-Forward Analysis Results")
        
        logger.info("Overall Performance Metrics:")
        print_metrics(overall_metrics)
        
        logger.info(f"\nAnalyzed {len(period_results)} windows")
        
        # Save summary report
        summary_file = os.path.join(output_dir, "walkforward_summary.txt")
        save_results_summary(results, summary_file, "Walk-Forward Analysis Results")
        
        # Save combined equity curve
        if not combined_equity_curve.empty:
            combined_equity_curve.to_csv(os.path.join(output_dir, "combined_equity_curve.csv"))
        
        # Save combined trade log
        if not combined_trade_log.empty:
            combined_trade_log.to_csv(os.path.join(output_dir, "combined_trade_log.csv"))
        
        logger.info(f"\nDetailed results saved to: {output_dir}")
        
        # Reset logging level if it was changed
        if verbose:
            logging_system.set_level('INFO', 'workflows')
        
        # Log workflow completion
        completion_info = {
            "total_return": f"{overall_metrics.get('total_return', 0.0):.2%}",
            "win_rate": f"{overall_metrics.get('win_rate', 0.0):.2%}",
            "output_dir": output_dir
        }
        print_workflow_log(
            workflow_name="Walk-Forward Analysis Workflow",
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            status="COMPLETED",
            additional_info=completion_info
        )
        
        return {
            "status": "success",
            "results": results,
            "output_dir": output_dir
        }
    except Exception as e:
        logger.error(f"Error in run_walkforward_workflow: {e}")
        return {"status": "error", "message": str(e)} 