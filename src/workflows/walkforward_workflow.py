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
import datetime
from typing import Dict, List, Any, Optional, Tuple

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
    logger, logging_system, print_workflow_log,
    check_logs_for_errors, print_error_report
)

# Import engine components
from engine.testing.walk_forward_test import WalkForwardTest
from workflows.simple_workflow import ensure_data_available

@time_execution("walkforward workflow")
def run_walkforward_workflow(
    strategy_name: str = None,
    strategy: str = None,  # Alternative to strategy_name for compatibility
    tickers: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    output_dir: str = None,
    parameters: Optional[Dict[str, Any]] = None,
    param_file: Optional[str] = None,
    window_size: int = 252,
    step_size: int = 63,
    n_trials: int = 50,
    optimization_metric: str = "sharpe_ratio",
    verbose: bool = False,
    initial_capital: float = 100000.0,
    commission: float = 0.001,
    _temp_files_to_cleanup: Optional[List[str]] = None,
    data_dir: str = "input",
    plot: bool = False  # Whether to generate plots during backtests
) -> Dict[str, Any]:
    """
    Run a walk-forward analysis workflow for the given strategy.
    
    Args:
        strategy_name: Name of the strategy to run
        strategy: Alternative to strategy_name (for compatibility)
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
        plot: Whether to generate plots (ignored, included for compatibility)
    
    Returns:
        Dict containing the workflow results
    """
    # Allow strategy parameter as alternative to strategy_name for compatibility
    if strategy_name is None and strategy is not None:
        strategy_name = strategy
        
    # Track temporary files if not already tracking
    if _temp_files_to_cleanup is None:
        _temp_files_to_cleanup = []
        
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
    
    # Use param_file if provided, otherwise look for default grid file
    if param_file and os.path.exists(param_file):
        param_grid_file = param_file
        logger.info(f"Using parameter grid file: {param_grid_file}")
    else:
        # Try to find parameter grid file in multiple potential locations
        strategy_snake_case = ''.join(['_'+c.lower() if c.isupper() else c.lower() for c in strategy_name]).lstrip('_')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # List of possible locations to search
        possible_locations = [
            os.path.join(project_root, "input", "parameter_grids", f"{strategy_name.lower()}_grid.json"),
            os.path.join(project_root, "input", "parameter_grids", f"{strategy_name}_grid.json"),
            os.path.join(project_root, "input", "parameter_grids", f"{strategy_snake_case}_grid.json"),
            os.path.join(project_root, "input", f"{strategy_name.lower()}_grid.json"),
            os.path.join(project_root, "input", f"{strategy_name}_grid.json"),
        ]
        
        # Add special case paths for specific strategies
        if strategy_name == "MACrossover":
            possible_locations.extend([
                os.path.join(project_root, "input", "parameter_grids", "ma_crossover_grid.json"),
                os.path.join(project_root, "input", "ma_crossover_grid.json"),
            ])
        
        # Try to find the grid file
        param_grid_file = None
        for location in possible_locations:
            if os.path.exists(location):
                param_grid_file = location
                logger.info(f"Found parameter grid file: {location}")
                break
        
        # If no grid file is found, create one
        if not param_grid_file:
            logger.warning(f"No parameter grid file found for {strategy_name}. Creating a temporary one.")
            
            # Get default parameters from the strategy parameter file
            strategy_param_file = find_strategy_param_file(strategy_name)
            if strategy_param_file:
                try:
                    with open(strategy_param_file, 'r') as f:
                        params = json.load(f)
                    
                    # Create a simple grid with some variations
                    param_grid = {}
                    for key, value in params.items():
                        if isinstance(value, (int, float)) and key not in ['initial_capital', 'commission']:
                            # Create a range of values around the default
                            if value == 0:
                                param_grid[key] = [0, 1, 2, 5, 10]
                            else:
                                param_grid[key] = [
                                    value * 0.5,
                                    value * 0.75,
                                    value,
                                    value * 1.25,
                                    value * 1.5
                                ]
                    
                    # Ensure directory exists
                    os.makedirs(os.path.join(project_root, "input", "parameter_grids"), exist_ok=True)
                    
                    # Save the grid to a file
                    param_grid_file = os.path.join(project_root, "input", "parameter_grids", f"{strategy_name.lower()}_grid_{timestamp}.json")
                    with open(param_grid_file, 'w') as f:
                        json.dump(param_grid, f, indent=4)
                    
                    logger.info(f"Created temporary parameter grid file: {param_grid_file}")
                    
                    # Track for cleanup
                    _temp_files_to_cleanup.append(param_grid_file)
                except Exception as e:
                    logger.error(f"Error creating parameter grid: {e}")
                    return {"status": "error", "message": f"Parameter grid file not found and could not create one: {str(e)}"}
            else:
                logger.error(f"Error: Parameter grid file not found for {strategy_name} and no default parameters available")
                return {"status": "error", "message": "Parameter grid file not found and no default parameters available"}
    
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
        
        # Calculate in-sample and out-of-sample date ranges
        total_days = (datetime.datetime.strptime(end_date, '%Y-%m-%d') - 
                     datetime.datetime.strptime(start_date, '%Y-%m-%d')).days
        
        # Split into in-sample and out-of-sample periods
        # Using 80% for in-sample by default if window_size and step_size aren't specified
        in_sample_days = int(total_days * 0.8)
        out_sample_days = total_days - in_sample_days
        
        in_sample_start = start_date
        in_sample_end_date = datetime.datetime.strptime(start_date, '%Y-%m-%d') + datetime.timedelta(days=in_sample_days)
        in_sample_end = in_sample_end_date.strftime('%Y-%m-%d')
        
        out_sample_start = (in_sample_end_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        out_sample_end = end_date
        
        logger.info(f"In-sample period: {in_sample_start} to {in_sample_end}")
        logger.info(f"Out-of-sample period: {out_sample_start} to {out_sample_end}")
        
        # Create walk-forward analyzer
        walkforward = WalkForwardTest(
            strategy_name=strategy_name,
            tickers=tickers,
            in_sample_start=in_sample_start,
            in_sample_end=in_sample_end,
            out_sample_start=out_sample_start,
            out_sample_end=out_sample_end,
            output_dir=output_dir,
            parameters=parameters,
            plot=plot  # Respect the plot parameter from user configuration
        )
        
        # Run walk-forward analysis
        results = walkforward.run_test()
        
        # Extract and process results
        period_results = []
        combined_equity_curve = pd.DataFrame()
        combined_trade_log = pd.DataFrame()

        # Ensure combined_equity_curve and combined_trade_log are properly initialized
        if not isinstance(combined_equity_curve, pd.DataFrame):
            combined_equity_curve = pd.DataFrame()
        if not isinstance(combined_trade_log, pd.DataFrame):
            combined_trade_log = pd.DataFrame()
            
        # Add this period's results since run_test() doesn't properly populate periods
        if 'in_sample_results' in results and 'out_sample_results' in results:
            period_results.append({
                "in_sample": {
                    "start_date": in_sample_start,
                    "end_date": in_sample_end
                },
                "out_sample": {
                    "start_date": out_sample_start,
                    "end_date": out_sample_end,
                    "metrics": results.get('out_sample_results', {})
                },
                "best_params": parameters
            })
        
        # Check if the comparison DataFrame is available from the results
        comparison_data = None
        if 'comparison' in results and isinstance(results['comparison'], pd.DataFrame) and not results['comparison'].empty:
            comparison_data = results['comparison']
            logger.info("Found performance comparison data in results")
            
            # Save a copy of the comparison data with proper formatting
            comparison_file = os.path.join(output_dir, 'performance_comparison_formatted.csv')
            
            # Format the values for display (convert decimals to percentages)
            formatted_comparison = comparison_data.copy()
            for col in ['In-Sample', 'Out-of-Sample', 'Difference']:
                formatted_comparison[col] = formatted_comparison[col].apply(
                    lambda x: f"{float(x)*100:.2f}%" if isinstance(x, (int, float)) 
                    else (f"{float(x)*100:.2f}%" if isinstance(x, str) and x.replace('.', '', 1).isdigit() 
                          else x))
            
            formatted_comparison.to_csv(comparison_file)
            logger.info(f"Saved formatted performance comparison to {comparison_file}")
        else:
            logger.warning("No performance comparison data found in results")
        
        # Look for the summary file which contains extracted metrics
        summary_file = results.get('summary_file')
        if summary_file and os.path.exists(summary_file):
            logger.info(f"Found summary file: {summary_file}")
            
            # Extract key metrics from the summary file for the final report
            try:
                with open(summary_file, 'r') as f:
                    summary_content = f.read()
                    
                # Include summary in workflow summary
                summary_section = "\nWalk Forward Analysis Summary:\n"
                summary_section += "-----------------------------\n"
                summary_section += summary_content
                
                # Create a workflow summary file
                with open(os.path.join(output_dir, 'workflow_summary.txt'), 'w') as f:
                    f.write(summary_section)
            except Exception as e:
                logger.error(f"Error processing summary file: {e}")
        
        for period in results.get("periods", []):
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
            if "out_sample_equity" in period and isinstance(period["out_sample_equity"], pd.DataFrame) and not period["out_sample_equity"].empty:
                if isinstance(combined_equity_curve, pd.DataFrame) and combined_equity_curve.empty:
                    combined_equity_curve = period["out_sample_equity"]
                elif isinstance(combined_equity_curve, pd.DataFrame):
                    # Append making sure there's no overlap
                    last_date = combined_equity_curve.index[-1]
                    out_sample_curve = period["out_sample_equity"]
                    if out_sample_curve.index[0] <= last_date:
                        out_sample_curve = out_sample_curve[out_sample_curve.index > last_date]
                    
                    if not out_sample_curve.empty:
                        combined_equity_curve = pd.concat([combined_equity_curve, out_sample_curve])
                else:
                    combined_equity_curve = period["out_sample_equity"]
            
            if "out_sample_trades" in period and isinstance(period["out_sample_trades"], pd.DataFrame) and not period["out_sample_trades"].empty:
                if isinstance(combined_trade_log, pd.DataFrame) and combined_trade_log.empty:
                    combined_trade_log = period["out_sample_trades"]
                elif isinstance(combined_trade_log, pd.DataFrame):
                    combined_trade_log = pd.concat([combined_trade_log, period["out_sample_trades"]])
                else:
                    combined_trade_log = period["out_sample_trades"]
        
        # Calculate overall performance metrics from combined equity curve
        overall_metrics = {}
        if isinstance(combined_equity_curve, pd.DataFrame) and not combined_equity_curve.empty:
            # Convert to numeric if possible
            try:
                # If it contains non-numeric columns, only select numeric ones
                numeric_cols = combined_equity_curve.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    # Use only numeric columns
                    numeric_df = combined_equity_curve[numeric_cols]
                else:
                    # Try to convert to numeric
                    numeric_df = combined_equity_curve.apply(pd.to_numeric, errors='coerce')
                    
                # Use the first column if there are multiple
                if len(numeric_df.columns) > 0:
                    equity_series = numeric_df.iloc[:, 0]
                    
                    # Basic metrics
                    returns = equity_series.pct_change().dropna()
                    
                    try:
                        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
                        overall_metrics["total_return"] = total_return if np.isfinite(total_return) else 0.0
                    except (IndexError, ZeroDivisionError) as e:
                        logger.warning(f"Could not calculate total_return: {e}")
                        overall_metrics["total_return"] = 0.0
                else:
                    logger.warning("No numeric columns found in equity curve")
                    overall_metrics["total_return"] = 0.0
            except Exception as e:
                logger.warning(f"Could not convert equity curve to numeric: {e}")
                # Default metrics since conversion failed
                overall_metrics["total_return"] = 0.0
                overall_metrics["annualized_return"] = 0.0
                overall_metrics["sharpe_ratio"] = 0.0
                overall_metrics["max_drawdown"] = 0.0
            
            # These calculations will only run if we successfully created an equity_series in the try block above
            if 'equity_series' in locals() and 'returns' in locals():
                try:
                    # Safely calculate annualized return
                    days = len(equity_series)
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
                    rolling_max = equity_series.cummax()
                    drawdown = (equity_series / rolling_max) - 1
                    max_dd = drawdown.min()
                    overall_metrics["max_drawdown"] = max_dd if np.isfinite(max_dd) else 0.0
                except Exception as e:
                    logger.warning(f"Could not calculate max_drawdown: {e}")
                    overall_metrics["max_drawdown"] = 0.0
            else:
                # Set default values if equity_series wasn't created
                overall_metrics["annualized_return"] = 0.0
                overall_metrics["sharpe_ratio"] = 0.0
                overall_metrics["max_drawdown"] = 0.0
        
        # Calculate trade statistics from combined trade log
        if isinstance(combined_trade_log, pd.DataFrame) and not combined_trade_log.empty:
            overall_metrics["total_trades"] = len(combined_trade_log)
            
            try:
                # Check for common profit column names
                profit_col = None
                possible_profit_cols = ["profit", "pnl", "PnL", "Profit", "profit_loss", "P&L"]
                
                for col in possible_profit_cols:
                    if col in combined_trade_log.columns:
                        profit_col = col
                        break
                
                if profit_col is not None:
                    # Try to convert profit column to numeric if it's not already
                    if not pd.api.types.is_numeric_dtype(combined_trade_log[profit_col]):
                        combined_trade_log[profit_col] = pd.to_numeric(combined_trade_log[profit_col], errors='coerce')
                        
                    # Calculate win rate
                    winning_trades = combined_trade_log[combined_trade_log[profit_col] > 0]
                    overall_metrics["win_rate"] = len(winning_trades) / len(combined_trade_log) if len(combined_trade_log) > 0 else 0.0
                    
                    # Calculate profit factor
                    try:
                        losing_trades = combined_trade_log[combined_trade_log[profit_col] <= 0]
                        losing_sum = losing_trades[profit_col].sum()
                        winning_sum = winning_trades[profit_col].sum()
                        
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
                    
                    # Calculate average profit
                    try:
                        avg_profit = combined_trade_log[profit_col].mean()
                        overall_metrics["avg_profit"] = avg_profit if np.isfinite(avg_profit) else 0.0
                    except Exception as e:
                        logger.warning(f"Could not calculate avg_profit: {e}")
                        overall_metrics["avg_profit"] = 0.0
                else:
                    logger.warning(f"No profit column found in trade log. Looked for: {possible_profit_cols}")
                    logger.warning(f"Available columns: {combined_trade_log.columns.tolist()}")
                    overall_metrics["win_rate"] = 0.0
                    overall_metrics["profit_factor"] = 0.0
                    overall_metrics["avg_profit"] = 0.0
                    
            except Exception as e:
                logger.warning(f"Could not calculate trade statistics: {e}")
                overall_metrics["win_rate"] = 0.0
                overall_metrics["profit_factor"] = 0.0
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
            "period_results": period_results,
            "parameters": {  # Add parameters to match expected structure in save_results_summary
                "window_size": window_size,
                "step_size": step_size,
                "n_trials": n_trials,
                "optimization_metric": optimization_metric
            },
            "metrics": overall_metrics  # Add metrics to match expected structure in save_results_summary
        }
        
        print_section("Walk-Forward Analysis Results")
        
        logger.info("Overall Performance Metrics:")
        print_metrics(overall_metrics)
        
        logger.info(f"\nAnalyzed {len(period_results)} windows")
        
        # Save combined equity curve and trade log
        if not combined_equity_curve.empty:
            combined_equity_path = os.path.join(output_dir, "combined_equity_curve.csv")
            combined_equity_curve.to_csv(combined_equity_path)
            logger.info(f"Combined equity curve saved to {combined_equity_path}")
        
        if not combined_trade_log.empty:
            combined_trades_path = os.path.join(output_dir, "combined_trade_log.csv")
            combined_trade_log.to_csv(combined_trades_path)
            logger.info(f"Combined trade log saved to {combined_trades_path}")
        
        # Save the performance metrics to a CSV file
        performance_metrics = {
            "total_return": overall_metrics.get("total_return", 0.0) if overall_metrics else 0.0,
            "sharpe_ratio": overall_metrics.get("sharpe_ratio", 0.0) if overall_metrics else 0.0, 
            "max_drawdown": overall_metrics.get("max_drawdown", 0.0) if overall_metrics else 0.0,
            "total_trades": overall_metrics.get("total_trades", 0) if overall_metrics else 0
        }
        
        # Create a copy of the parameters for the performance metrics
        parameter_summary = {}
        if parameters:
            parameter_summary = parameters.copy()
        else:
            # If no direct parameters, try to get them from the period results
            for period in period_results:
                if 'best_params' in period and period['best_params']:
                    parameter_summary = period['best_params'].copy()
                    logger.info(f"Using parameters from period results: {parameter_summary}")
                    break
        
        # If still no parameters, check results for parameters
        if not parameter_summary and hasattr(results, 'get') and results.get('best_params'):
            parameter_summary = results.get('best_params', {}).copy()
            logger.info(f"Using best parameters from results: {parameter_summary}")
            
        # Ensure parameters are in the proper format for the summary
        if parameter_summary:
            logger.info(f"Parameters to include in summary: {parameter_summary}")
        
        # Read performance comparison data if available
        if os.path.exists(os.path.join(output_dir, "performance_comparison.csv")):
            try:
                comparison_df = pd.read_csv(os.path.join(output_dir, "performance_comparison.csv"), index_col=0)
                if not comparison_df.empty:
                    for idx, row in comparison_df.iterrows():
                        metric_name = idx.lower().replace(' ', '_')
                        performance_metrics[f"{metric_name}_in_sample"] = row["In-Sample"]
                        performance_metrics[f"{metric_name}_out_sample"] = row["Out-of-Sample"]
                        performance_metrics[f"{metric_name}_difference"] = row["Difference"]
                        performance_metrics[f"{metric_name}_degradation"] = row["Degradation %"]
                    
                    # Update the overall metrics with the out-of-sample values
                    # These are more reliable than the ones calculated from the combined equity curve
                    if "total_return" in comparison_df.index:
                        performance_metrics["total_return"] = comparison_df.loc["total_return", "Out-of-Sample"]
            except Exception as e:
                logger.error(f"Error reading performance comparison: {e}")
        
        # Create summary dictionary with all results
        summary = {
            "strategy": strategy_name,
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "period_results": period_results,
            "overall_metrics": overall_metrics or {},
            "performance_metrics": performance_metrics,
            "parameters": parameter_summary,
            "window_size": window_size,
            "step_size": step_size,
            "output_dir": output_dir,
            "status": "success"
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, "walkforward_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate and save results summary
        results_for_summary = {
            "strategy_name": strategy_name,
            "parameters": parameter_summary,  # Use the copied parameters
            "dates": {
                "start_date": start_date,
                "end_date": end_date
            },
            "metrics": performance_metrics,
            "notes": f"Walk-forward analysis with {len(period_results)} periods, {performance_metrics.get('total_trades', 0)} trades"
        }
        
        summary_text = save_results_summary(
            results=results_for_summary,
            filename=os.path.join(output_dir, "walkforward_summary.txt"),
            title=f"Walk-Forward Analysis: {strategy_name}"
        )
        
        # Ensure parameters are properly displayed in summary file
        def update_summary_file(summary_file, parameters, total_trades):
            """Update summary file to ensure parameters and total trades are properly displayed."""
            try:
                if not os.path.exists(summary_file):
                    logger.error(f"Summary file not found: {summary_file}")
                    return
                    
                with open(summary_file, 'r') as f:
                    lines = f.readlines()
                
                logger.info(f"Updating summary file with parameters: {parameters}")
                
                # Update parameter section if empty
                param_section_start = -1
                param_section_end = -1
                for i, line in enumerate(lines):
                    if "Parameters:" in line:
                        param_section_start = i
                        logger.debug(f"Found Parameters section at line {i}")
                    elif param_section_start > 0 and line.startswith('-' * 10):
                        param_section_end = i
                        logger.debug(f"Found end of Parameters section at line {i}")
                        break
                
                if param_section_start > 0 and param_section_end > param_section_start:
                    # Check if parameter section is empty
                    empty_param_section = True
                    for i in range(param_section_start + 1, param_section_end):
                        if lines[i].strip() and not lines[i].startswith("-"):
                            empty_param_section = False
                            break
                    
                    # If empty, add parameters
                    if empty_param_section and parameters:
                        logger.info(f"Adding {len(parameters)} parameters to summary file")
                        new_lines = []
                        for i, line in enumerate(lines):
                            new_lines.append(line)
                            if i == param_section_start:
                                # For safety, check if parameters is a dictionary before iterating
                                if isinstance(parameters, dict):
                                    for key, value in parameters.items():
                                        new_lines.append(f"{key}: {value}\n")
                                # If it's None or empty, add a default placeholder
                                elif parameters is None or not parameters:
                                    new_lines.append("Default strategy parameters\n")
                        
                        # Write the updated file
                        with open(summary_file, 'w') as f:
                            f.writelines(new_lines)
                        logger.info(f"Updated summary file with parameters")
                        return
                    else:
                        logger.info(f"Parameter section already has content or no parameters to add")
                else:
                    logger.warning(f"Could not find Parameters section in summary file")
                
                # Handle total trades section separately if we didn't update parameters
                # Ensure total trades are shown in metrics section
                metrics_section_start = -1
                for i, line in enumerate(lines):
                    if "Performance Metrics:" in line:
                        metrics_section_start = i
                        break
                
                if metrics_section_start > 0:
                    # Check if total_trades is already in metrics
                    found_total_trades = False
                    for i in range(metrics_section_start + 1, len(lines)):
                        if "total_trades:" in lines[i].lower():
                            found_total_trades = True
                            break
                    
                    # If not found, add it to the metrics section
                    if not found_total_trades:
                        new_lines = []
                        for i, line in enumerate(lines):
                            new_lines.append(line)
                            if i == metrics_section_start + 1:  # Add after the section header line
                                new_lines.append(f"total_trades: {total_trades}\n")
                        
                        # Write the updated file
                        with open(summary_file, 'w') as f:
                            f.writelines(new_lines)
                        logger.info(f"Added total_trades to summary file")
                
            except Exception as e:
                logger.error(f"Error updating summary file: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Update the summary file to ensure parameters and total trades are displayed
        update_summary_file(
            os.path.join(output_dir, "walkforward_summary.txt"),
            parameter_summary,
            performance_metrics.get("total_trades", 0)
        )
        
        # Create a direct summary file with parameters if the update didn't work
        # Always create this file, even if parameter_summary is empty
        direct_summary_path = os.path.join(output_dir, "parameters_summary.txt")
        try:
            with open(direct_summary_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write(f"Parameters Used in Walk-Forward Analysis: {strategy_name}\n")
                f.write("=" * 80 + "\n\n")
                
                if isinstance(parameter_summary, dict) and parameter_summary:
                    for key, value in parameter_summary.items():
                        f.write(f"{key}: {value}\n")
                else:
                    # If we don't have parameters, include walkforward settings
                    f.write("Default strategy parameters were used\n\n")
                    f.write("Walk-Forward Settings:\n")
                    f.write(f"window_size: {window_size}\n")
                    f.write(f"step_size: {step_size}\n")
                    f.write(f"n_trials: {n_trials}\n")
                    f.write(f"optimization_metric: {optimization_metric}\n")
                    
            logger.info(f"Created direct parameter summary at {direct_summary_path}")
        except Exception as e:
            logger.error(f"Error creating direct parameter summary: {e}")
        
        logger.info(f"\nDetailed results saved to: {output_dir}")
        
        # Reset logging level if it was changed
        if verbose:
            logging_system.set_level('INFO', 'workflows')
        
        # Log workflow completion
        total_return_value = overall_metrics.get('total_return', 0.0) if overall_metrics else 0.0
        win_rate_value = overall_metrics.get('win_rate', 0.0) if overall_metrics else 0.0
        completion_info = {
            "total_return": f"{total_return_value * 100:.2f}%",
            "win_rate": f"{win_rate_value * 100:.2f}%",
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
        
        # Clean up temporary files
        files_to_delete = []
        files_skipped = []
        
        for temp_file in _temp_files_to_cleanup:
            if os.path.exists(temp_file):
                # Skip files in the workflow_configs directory
                if "workflow_configs" in temp_file:
                    files_skipped.append(temp_file)
                else:
                    files_to_delete.append(temp_file)
        
        if files_skipped:
            logger.info(f"Skipping cleanup of {len(files_skipped)} workflow config files")
            for file_path in files_skipped:
                logger.debug(f"Preserved file: {file_path}")
        
        for temp_file in files_to_delete:
            try:
                os.remove(temp_file)
                logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Error cleaning up temporary file {temp_file}: {str(e)}")
        
        # Check logs for errors
        logger.info("Checking logs for errors...")
        error_logs = check_logs_for_errors(output_dir)
        
        if error_logs:
            # Add log errors to the results
            summary["log_errors"] = {
                "count": sum(len(errors) for errors in error_logs.values()),
                "files": len(error_logs)
            }
            
            # Generate error report and save to file
            error_report_path = os.path.join(output_dir, "error_report.txt")
            print_error_report(error_logs, error_report_path)
            logger.warning(f"Found errors in logs. Error report saved to: {error_report_path}")
        else:
            logger.info("No errors found in logs.")
            summary["log_errors"] = {"count": 0, "files": 0}
        
        return {
            "status": "success",
            "results": results,
            "output_dir": output_dir
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error in run_walkforward_workflow: {e}")
        logger.error(f"Traceback: {error_trace}")
        
        # Clean up temporary files even on error
        for temp_file in _temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up temporary file {temp_file}: {str(cleanup_error)}")
        
        # Check logs for errors
        logger.info("Checking logs for errors...")
        error_logs = check_logs_for_errors(output_dir)
        
        if error_logs:
            # Generate error report and save to file
            error_report_path = os.path.join(output_dir, "error_report.txt")
            print_error_report(error_logs, error_report_path)
            logger.warning(f"Found errors in logs. Error report saved to: {error_report_path}")
        
        return {"status": "error", "message": str(e)} 