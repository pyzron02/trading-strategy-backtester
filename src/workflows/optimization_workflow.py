#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization workflow module.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import datetime
import uuid
from tqdm import tqdm

# Import the utilities
from workflows.workflow_utils import (
    print_header, print_section, print_parameters, print_metrics,
    save_results_summary, time_execution, find_strategy_param_file,
    logger, logging_system, print_workflow_log, adapt_strategy_parameters,
    setup_output_dir_logging, remove_output_dir_logging,
    check_logs_for_errors, print_error_report,
    workflow_setup, workflow_teardown
)
from workflows.config import WorkflowConfig
from utils.path_manager import path_manager

# Import engine components
from engine.run_backtest import run_backtest
from engine.testing.in_sample_excellence import InSampleExcellence
from workflows.simple_workflow import ensure_data_available
from utils.error_reporting import create_stage_error_report, StageError, StageErrorReport

@time_execution("optimization workflow")
def run_optimization_workflow(
    strategy=None,  # New parameter to support unified_workflow
    strategy_name=None,  # Original parameter
    tickers=None,
    start_date=None,
    end_date=None,
    output_dir=None,
    parameters=None,
    param_file=None,
    n_trials=50,
    optimization_metric="sharpe_ratio",
    max_combinations=None,
    verbose=False,
    initial_capital=100000.0,
    commission=0.001,
    data_dir="input",
    progress_callback=None,
    progress_file=None,
    stock_csv=None,
    _temp_files_to_cleanup=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run an optimization workflow for a trading strategy.
    
    Args:
        strategy: Name of the strategy to optimize (alternative to strategy_name)
        strategy_name: Name of the strategy to optimize
        tickers: List of ticker symbols
        start_date: Start date for backtest
        end_date: End date for backtest
        output_dir: Directory for output files
        parameters: Dictionary of strategy parameters (overrides param_file)
        param_file: File with parameter grid definitions
        n_trials: Number of trials to run
        optimization_metric: Metric to optimize for
        max_combinations: Maximum parameter combinations to try
        verbose: Whether to print detailed logs
        initial_capital: Initial capital for backtest
        commission: Commission per trade
        data_dir: Directory with data files
        plot: Whether to plot results
        progress_callback: Callback for progress updates
        progress_file: File to write progress updates
        stock_csv: CSV file with stock data
        _temp_files_to_cleanup: List of temporary files to clean up
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with optimization results
    """
    config = WorkflowConfig.from_kwargs(
        strategy=strategy, strategy_name=strategy_name, tickers=tickers,
        start_date=start_date, end_date=end_date, output_dir=output_dir,
        parameters=parameters, param_file=param_file, verbose=verbose,
        initial_capital=initial_capital, commission=commission, data_dir=data_dir,
        n_trials=n_trials, optimization_metric=optimization_metric,
        max_combinations=max_combinations, progress_callback=progress_callback,
        progress_file=progress_file, stock_csv=stock_csv,
        _temp_files_to_cleanup=_temp_files_to_cleanup or [], **kwargs
    )
    try:
        workflow_setup(config, "optimization")
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    # Extract commonly used values from config
    strategy_name = config.strategy_name
    tickers = config.tickers
    output_dir = config.output_dir
    
    # Find parameter grid file if not provided
    if not param_file:
        # Look in several possible locations for parameter grid files
        # Also search for lowercase strategy name and snake_case variation
        strategy_snake_case = ''.join(['_'+c.lower() if c.isupper() else c.lower() for c in strategy_name]).lstrip('_')
        possible_locations = [
            os.path.join(str(path_manager.input_dir), "parameter_grids", f"{strategy_name}_grid.json"),
            os.path.join(str(path_manager.input_dir), "parameter_grids", f"{strategy_name.lower()}_grid.json"),
            os.path.join(str(path_manager.input_dir), "parameter_grids", f"{strategy_snake_case}_grid.json"),
            os.path.join(str(path_manager.input_dir), f"{strategy_name}_grid.json"),
            os.path.join(str(path_manager.input_dir), "grids", f"{strategy_name}_grid.json"),
            os.path.join(str(path_manager.input_dir), f"{strategy_name.lower()}_grid.json"),
        ]
        
        # Debug all locations we're checking
        logger.info(f"Searching for parameter grid file for {strategy_name} in:")
        for location in possible_locations:
            logger.info(f"  Checking: {location} (exists: {os.path.exists(location)})")
            if os.path.exists(location):
                param_file = location
                logger.info(f"  Found parameter grid file: {location}")
                break
        
        if not param_file:
            # If no grid file is found, create a basic one from the strategy parameters
            logger.warning(f"No parameter grid file found for {strategy_name}. Creating a basic grid.")
            
            # Get default parameters
            param_file = find_strategy_param_file(strategy_name)
            if param_file:
                try:
                    with open(param_file, 'r') as f:
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
                    
                    # Save the grid to a temporary file
                    param_file = os.path.join(output_dir, f"{strategy_name}_grid.json")
                    with open(param_file, 'w') as f:
                        json.dump(param_grid, f, indent=4)
                    
                    logger.info(f"Created parameter grid file: {param_file}")
                    
                    # Track for cleanup
                    config._temp_files_to_cleanup.append(param_file)
                except Exception as e:
                    logger.error(f"Error creating parameter grid: {e}")
                    msg = f"Parameter grid file not found and could not create one: {str(e)}"
                    workflow_teardown(config, "optimization", None, error=Exception(msg))
                    try:
                        create_stage_error_report(output_dir, 'optimization', strategy_name)
                    except Exception as report_err:
                        logger.error(f"Error generating stage error report: {report_err}")
                    return {"status": "error", "message": msg}
            else:
                msg = f"Parameter grid file not found for {strategy_name} and no default parameters available"
                logger.error(f"Error: {msg}")
                workflow_teardown(config, "optimization", None, error=Exception(msg))
                return {"status": "error", "message": msg}
    
    if not os.path.exists(param_file):
        msg = f"Parameter grid file does not exist: {param_file}"
        logger.error(f"Error: {msg}")
        workflow_teardown(config, "optimization", None, error=Exception(msg))
        return {"status": "error", "message": msg}
    
    print_section("Running Optimization")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Tickers: {', '.join(tickers)}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Optimization metric: {optimization_metric}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Parameter grid file: {param_file}")
    
    try:
        # Extract keep_all_results from kwargs if available
        keep_all_results = kwargs.get("keep_all_results", False)
        logger.info(f"Keep all parameter set results: {keep_all_results}")
        
        # Pre-download data once for all optimization runs
        logger.info("Pre-downloading stock data for optimization...")
        try:
            from data_preprocessing.data_setup import fetch_stock_data
            # Ensure data is available before running optimization
            stock_data_path = fetch_stock_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                output_path=stock_csv if stock_csv else 'input/stock_data.csv',
                force_refresh=False  # Don't force refresh, use cached data if available
            )
            logger.info(f"Stock data available at: {stock_data_path}")
        except Exception as data_error:
            logger.warning(f"Could not pre-download data: {data_error}. Will download during optimization.")
        
        # Extract parallel processing parameters from kwargs
        n_jobs = kwargs.get('n_jobs', None)
        parallel_backend = kwargs.get('parallel_backend', 'multiprocessing')
        batch_size = kwargs.get('batch_size', None)
        
        # Log parallel processing settings
        if n_jobs is not None:
            logger.info(f"Parallel processing: {n_jobs} jobs ({'all cores' if n_jobs == -1 else n_jobs})")
        else:
            logger.info("Parallel processing: auto (all available cores)")
        if batch_size:
            logger.info(f"Batch size: {batch_size}")
        
        # Initialize optimizer
        optimizer = InSampleExcellence(
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            param_grid_file=param_file,
            n_trials=n_trials,
            optimization_metric=optimization_metric,
            output_dir=output_dir,
            initial_capital=initial_capital,
            commission=commission,
            data_dir=data_dir,
            max_combinations=max_combinations,
            verbose=verbose,
            keep_results=keep_all_results,  # Pass keep_all_results as keep_results
            stock_csv=stock_csv if stock_csv else 'input/stock_data.csv',  # Pass the stock data path
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            batch_size=batch_size
        )
        
        # Run optimization
        result = optimizer.run()
        
        # Handle different return types from optimizer.run()
        if isinstance(result, tuple) and len(result) == 2:
            best_params, trials_df = result
        elif isinstance(result, dict) and 'parameters' in result:
            best_params = result.get('parameters', {})
            trials_df = result.get('all_results', pd.DataFrame())
        else:
            best_params = None
            trials_df = None
            logger.error(f"Unexpected result type from optimizer.run(): {type(result)}")
        
        # Check if we have valid results
        if best_params is None or trials_df is None:
            valid_results = False
        elif isinstance(trials_df, pd.DataFrame):
            valid_results = not trials_df.empty
        else:
            valid_results = False
            
        if not valid_results:
            msg = "Optimization failed to produce valid results"
            logger.error(msg)
            workflow_teardown(config, "optimization", None, error=Exception(msg))
            return {
                "status": "error",
                "message": msg,
                "output_dir": output_dir
            }
        
        # Check if optimization metric is present in trials_df
        if optimization_metric not in trials_df.columns:
            msg = f"Optimization metric '{optimization_metric}' not found in results"
            logger.error(msg)
            workflow_teardown(config, "optimization", None, error=Exception(msg))
            return {
                "status": "error",
                "message": msg,
                "output_dir": output_dir
            }
            
        # Ensure we have valid values in the optimization metric
        if not isinstance(trials_df, pd.DataFrame):
            msg = f"trials_df is not a DataFrame, it's a {type(trials_df)}"
            logger.error(msg)
            workflow_teardown(config, "optimization", None, error=Exception(msg))
            return {
                "status": "error",
                "message": msg,
                "output_dir": output_dir
            }
            
        if optimization_metric not in trials_df.columns:
            msg = f"Optimization metric '{optimization_metric}' not found in results columns: {list(trials_df.columns)}"
            logger.error(msg)
            workflow_teardown(config, "optimization", None, error=Exception(msg))
            return {
                "status": "error",
                "message": msg,
                "output_dir": output_dir
            }
            
        if trials_df[optimization_metric].isna().all():
            msg = f"All values for optimization metric '{optimization_metric}' are NaN"
            logger.error(msg)
            workflow_teardown(config, "optimization", None, error=Exception(msg))
            return {
                "status": "error",
                "message": msg,
                "output_dir": output_dir
            }
        
        # Run backtest with best parameters
        print_section("Running Backtest with Optimized Parameters")
        print_parameters(best_params)
        
        # Save best parameters to a file
        best_params_file = os.path.join(output_dir, f"{strategy_name}_best_params.json")
        with open(best_params_file, "w") as f:
            json.dump(best_params, f, indent=4)
        
        # Convert numerical parameters from numpy types to Python types and integers where needed
        converted_params = {}
        for key, value in best_params.items():
            # Convert float parameters to int for parameters that likely need integers
            if isinstance(value, (np.float64, np.float32, float)) and key in ['sma_period', 'max_positions']:
                converted_params[key] = int(value)
            elif isinstance(value, (np.float64, np.float32, np.int64, np.int32)):
                # Convert numpy types to Python native types
                converted_params[key] = value.item()
            else:
                converted_params[key] = value
                
        logger.info(f"Converting parameters for backtest: {best_params} -> {converted_params}")
        
        # Run backtest with converted best parameters
        backtest_result = run_backtest(
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            parameters=converted_params,
            initial_capital=initial_capital,
            commission=commission,
            data_dir=data_dir,
            stock_csv=stock_csv if stock_csv else 'input/stock_data.csv',  # Use cached data
            verbose=verbose
        )
        
        if not backtest_result:
            msg = "Backtest with optimized parameters failed"
            logger.error(msg)
            workflow_teardown(config, "optimization", None, error=Exception(msg))
            return {
                "status": "error",
                "message": msg,
                "output_dir": output_dir
            }
        
        # Extract and display results
        results = {
            "strategy_name": strategy_name,
            "dates": {
                "start_date": start_date,
                "end_date": end_date
            },
            "best_parameters": best_params,
            "metrics": backtest_result.get("metrics", {}),
            "trials_summary": {
                "n_trials": n_trials,
                "optimization_metric": optimization_metric,
                "best_value": None  # Initialize to None
            }
        }
        
        # Safely calculate best value
        try:
            # Make sure we have a valid DataFrame and column
            if isinstance(trials_df, pd.DataFrame) and optimization_metric in trials_df.columns:
                if optimization_metric.startswith("max_drawdown"):
                    best_value = trials_df[optimization_metric].min()
                else:
                    best_value = trials_df[optimization_metric].max()
                    
                # Check if best_value is valid
                if not pd.isna(best_value) and np.isfinite(best_value):
                    results["trials_summary"]["best_value"] = best_value
                else:
                    logger.warning(f"Best value for {optimization_metric} is not valid: {best_value}")
                    results["trials_summary"]["best_value"] = "N/A"
            else:
                logger.warning(f"Cannot calculate best value: trials_df is not a valid DataFrame or missing column {optimization_metric}")
                results["trials_summary"]["best_value"] = "N/A"
        except Exception as e:
            logger.warning(f"Could not calculate best value for {optimization_metric}: {str(e)}")
            results["trials_summary"]["best_value"] = "N/A"
        
        # Save trials dataframe
        trials_file = os.path.join(output_dir, "optimization_trials.csv")
        if isinstance(trials_df, pd.DataFrame):
            trials_df.to_csv(trials_file, index=False)
        else:
            logger.warning(f"Cannot save trials data to CSV: trials_df is not a DataFrame, it's a {type(trials_df)}")
            # Create an empty CSV with column headers so downstream code doesn't break
            pd.DataFrame(columns=['param_' + k for k in best_params.keys()] + [optimization_metric]).to_csv(trials_file, index=False)
        
        print_section("Optimization Results")
        logger.info(f"Best {optimization_metric}: {results['trials_summary']['best_value']}")
        
        logger.info("\nOptimized Parameters:")
        print_parameters(results["best_parameters"])
        
        logger.info("\nPerformance Metrics with Best Parameters:")
        print_metrics(results["metrics"])
        
        # Save summary report
        summary_file = os.path.join(output_dir, "optimization_summary.txt")
        save_results_summary(results, summary_file, "Optimization Results")
        logger.info(f"\nDetailed results saved to: {output_dir}")
    
    except Exception as e:
        logger.error(f"Optimization workflow failed: {str(e)}")
        if verbose:
            logger.exception("Full error traceback:")
        workflow_teardown(config, "optimization", None, error=e)
        return {
            "status": "error",
            "message": f"Optimization workflow failed: {str(e)}",
            "output_dir": output_dir
        }
    
    log_errors = workflow_teardown(config, "optimization", None,
                                   additional_info={
                                       "best_value": results["trials_summary"]["best_value"],
                                       "total_trials": n_trials,
                                   })
    results["log_errors"] = log_errors

    return {
        "status": "success",
        "results": results,
        "output_dir": output_dir
    }