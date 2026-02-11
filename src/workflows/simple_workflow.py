#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple backtest workflow module.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import datetime
import uuid
import logging
import tempfile

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
from engine.parameter_management import ParameterManager
from utils.error_reporting import create_stage_error_report, StageError, StageErrorReport

def _is_parameter_grid_value(value) -> bool:
    """Check if a list value looks like a parameter grid (list of numbers)."""
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(v, (int, float)) for v in value)


def convert_grid_to_single_values(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert parameter grids to single values by taking the first value from each list.

    Only converts lists of numeric values (parameter grids). Lists that contain
    strings, dicts, or other complex types are left as-is since they are
    legitimate list-typed parameters (e.g. ticker_priority).

    Args:
        parameters: Dictionary of parameters, potentially containing lists

    Returns:
        Dictionary with single values for each parameter
    """
    single_params = {}
    for key, value in parameters.items():
        if _is_parameter_grid_value(value):
            single_params[key] = value[0]
            logger.info(f"Parameter '{key}' is a grid. Using first value: {value[0]}")
        else:
            single_params[key] = value

    return single_params

def ensure_data_available(tickers: List[str], start_date: str, end_date: str, data_dir: str = "input"):
    """
    Ensure that stock_data.csv is available with the required tickers and date range.
    If not available, generate it using data_setup.py.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_dir: Directory containing input data
    
    Returns:
        Path to the stock data CSV file
    """
    # Ensure tickers is a list
    if tickers is None:
        tickers = ["SPY"]
    elif isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(',') if t.strip()]
    
    logger.info(f"Ensuring data available for tickers: {tickers}")
    
    # Construct the path to stock_data.csv
    stock_data_path = os.path.join(str(path_manager.input_dir), "stock_data.csv")
    
    # Check if stock_data.csv exists
    if not os.path.exists(stock_data_path):
        logger.info(f"stock_data.csv not found at {stock_data_path}. Generating it...")
        try:
            from data_preprocessing.data_setup import fetch_stock_data
            # Force refresh to ensure all tickers are included
            fetch_stock_data(tickers, start_date, end_date, force_refresh=False)
            logger.info(f"Generated stock_data.csv at {stock_data_path}")
        except Exception as e:
            logger.error(f"Error generating stock_data.csv: {e}")
            raise
    else:
        # Verify if the file contains the required tickers
        try:
            df = pd.read_csv(stock_data_path)
            columns = df.columns.tolist()
            
            # Check if all required tickers are present
            missing_tickers = []
            for ticker in tickers:
                required_column = f"{ticker}_Close" 
                if required_column not in columns:
                    missing_tickers.append(ticker)
            
            if missing_tickers:
                logger.warning(f"stock_data.csv is missing data for tickers: {missing_tickers}. Regenerating...")
                from data_preprocessing.data_setup import fetch_stock_data
                # Force refresh to ensure all tickers are included
                fetch_stock_data(tickers, start_date, end_date, force_refresh=False)
                logger.info(f"Regenerated stock_data.csv with all required tickers")
        except Exception as e:
            logger.error(f"Error verifying stock_data.csv: {e}")
            raise
    
    return stock_data_path

@time_execution("simple workflow")
def run_simple_workflow(
    strategy=None,  # New parameter to support unified_workflow
    strategy_name=None,  # Original parameter
    tickers=None,
    start_date=None,
    end_date=None,
    output_dir=None,
    parameters=None,
    param_file=None,
    verbose=False,
    initial_capital=100000.0,
    commission=0.001,
    data_dir="input",
    slippage=0.0,
    optimize_sharpe=False,
    live_mode=False,
    additional_data=None,
    commission_type="percentage",
    progress_callback=None,
    progress_file=None,
    stock_csv=None,
    _temp_files_to_cleanup=None,
    force_download=False,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a simple backtest for a single strategy with fixed parameters.
    
    Args:
        strategy: Name of the strategy to run (alternative to strategy_name)
        strategy_name: Name of the strategy to run
        tickers: List of ticker symbols
        start_date: Start date for backtest in YYYY-MM-DD format
        end_date: End date for backtest in YYYY-MM-DD format
        output_dir: Directory to save results
        parameters: Dictionary of strategy parameters (overrides param_file)
        param_file: File with parameter definitions
        plot: Whether to generate plots
        verbose: Whether to print detailed output
        initial_capital: Initial capital for backtest
        commission: Commission rate for trades
        data_dir: Directory containing input data
        slippage: Slippage per trade
        optimize_sharpe: Whether to optimize for Sharpe ratio
        live_mode: Whether to run in live mode
        additional_data: Additional data for the strategy
        progress_callback: Callback for progress updates
        progress_file: File to write progress updates
        stock_csv: CSV file with stock data
        _temp_files_to_cleanup: List of temporary files to clean up
        **kwargs: Additional arguments
    
    Returns:
        Dict containing the workflow results
    """
    config = WorkflowConfig.from_kwargs(
        strategy=strategy, strategy_name=strategy_name, tickers=tickers,
        start_date=start_date, end_date=end_date, output_dir=output_dir,
        parameters=parameters, param_file=param_file, verbose=verbose,
        initial_capital=initial_capital, commission=commission,
        commission_type=commission_type, data_dir=data_dir,
        slippage=slippage, optimize_sharpe=optimize_sharpe, live_mode=live_mode,
        additional_data=additional_data, progress_callback=progress_callback,
        progress_file=progress_file, stock_csv=stock_csv,
        _temp_files_to_cleanup=_temp_files_to_cleanup or [],
        force_download=force_download, **kwargs
    )
    try:
        workflow_setup(config, "simple")
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    # Extract commonly used values from config
    strategy_name = config.strategy_name
    tickers = config.tickers
    output_dir = config.output_dir

    if not param_file:
        param_file = find_strategy_param_file(strategy_name)
        if not param_file:
            # Check if we have parameters directly specified
            if parameters:
                # Create a temporary parameter file
                temp_param_file = os.path.join(str(path_manager.parameters_dir),
                                             f"{strategy_name.lower()}_params_temp.json")
                
                try:
                    with open(temp_param_file, 'w') as f:
                        json.dump(parameters, f, indent=4)
                    logger.info(f"Created temporary parameter file from provided parameters: {temp_param_file}")
                    param_file = temp_param_file
                    # Track for cleanup
                    config._temp_files_to_cleanup.append(temp_param_file)
                except Exception as e:
                    logger.error(f"Error creating temporary parameter file: {str(e)}")
            else:
                logger.warning(f"No parameter file found for strategy {strategy_name}. Using default parameters.")
        else:
            logger.info(f"Found parameter file: {param_file}")
    
    # Update progress if callback or file is provided
    if progress_callback:
        progress_callback(10, 100, "Loading data")
    
    if progress_file:
        with open(progress_file, 'w') as f:
            json.dump({
                "progress": 10,
                "status": "Loading data",
                "current_step": "Data preparation",
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=4)
    
    # Load parameters from file if available
    strategy_params = {}
    if param_file:
        try:
            with open(param_file, 'r') as f:
                strategy_params = json.load(f)
            logger.info(f"Loaded parameters from {param_file}")
            logger.debug(f"Parameters: {strategy_params}")
        except Exception as e:
            error_msg = f"Error loading parameters: {str(e)}"
            logger.error(f"Error loading parameters from {param_file}: {str(e)}")
            workflow_teardown(config, "simple", None, error=Exception(error_msg))
            return {"status": "error", "message": error_msg, "output_dir": config.output_dir}
    
    # Override with any directly provided parameters
    if parameters:
        # Only use valid parameters according to strategy adapter
        param_manager = ParameterManager()
        adapted_params = param_manager.adapt_strategy_parameters(strategy_name, parameters)
        strategy_params.update(adapted_params)
        logger.info("Updated parameters with provided values")
        logger.debug(f"Updated parameters: {strategy_params}")
    
    # Check for parameter grids and convert to single values if needed
    has_grid = any(_is_parameter_grid_value(v) for v in strategy_params.values())
    if has_grid:
        logger.info("Parameter grid detected in simple workflow. Converting to single values.")
        original_params = strategy_params.copy()
        strategy_params = convert_grid_to_single_values(strategy_params)
        logger.info("Converted parameters from grid to single values.")
    
    # Print parameters
    print_section("Strategy Parameters")
    print_parameters(strategy_params)
    
    try:
        # Check if we need to download data for our tickers
        stock_data_path = os.path.join(data_dir, "stock_data.csv")
        
        # Always use ensure_data_available to check if we need to fetch ticker data
        logger.info(f"Ensuring data is available for tickers: {tickers}")
        stock_csv = ensure_data_available(tickers, start_date, end_date, data_dir)
        
        if not os.path.exists(stock_data_path):
            error_msg = f"Failed to create or locate stock data file at: {stock_data_path}."
            logger.error(error_msg)
            workflow_teardown(config, "simple", None, error=Exception(error_msg))
            return {
                "status": "error",
                "message": error_msg,
                "output_dir": config.output_dir
            }
        
        # Run backtest
        print_section("Running Backtest")
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"Tickers: {', '.join(tickers)}")
        logger.info(f"Period: {start_date} to {end_date}")
        
        backtest_result = run_backtest(
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            parameters=strategy_params,
            stock_csv=stock_csv,  # Pass the explicit stock_csv path
            initial_capital=initial_capital,
            commission=commission,
            commission_type=config.commission_type,
            data_dir=data_dir,
            verbose=verbose,
            slippage=slippage,
            optimize_sharpe=optimize_sharpe,
            live_mode=live_mode,
            additional_data=additional_data,
            force_download=force_download
        )
        
        if not backtest_result:
            error_msg = "Backtest failed to produce valid results"
            logger.error(error_msg)
            workflow_teardown(config, "simple", None, error=Exception(error_msg))
            return {
                "status": "error",
                "message": error_msg,
                "output_dir": config.output_dir
            }
        
        # Extract and display results
        print_section("Backtest Results")
        metrics = backtest_result.get("metrics", {})
        
        # Print key metrics
        print_metrics(metrics)
        
        # Save results summary
        summary_file = os.path.join(output_dir, f"{strategy_name}_summary.txt")
        save_results_summary(backtest_result, summary_file, "Backtest Results")
        
        logger.info(f"\nDetailed results saved to: {output_dir}")
        
    except Exception as e:
        error_msg = f"Simple workflow failed: {str(e)}"
        logger.error(error_msg)
        if verbose:
            logger.exception("Full error traceback:")
        workflow_teardown(config, "simple", None, error=e)
        return {
            "status": "error",
            "message": error_msg,
            "output_dir": config.output_dir
        }

    # Create a combined result
    workflow_result = {
        "status": "success",
        "strategy_name": strategy_name,
        "dates": {
            "start_date": start_date,
            "end_date": end_date
        },
        "parameters": strategy_params,
        "metrics": metrics,
        "output_dir": output_dir
    }

    log_errors = workflow_teardown(config, "simple", workflow_result,
                                   additional_info={
                                       "total_return": f"{metrics.get('total_return', 0.0):.2%}",
                                       "sharpe_ratio": f"{metrics.get('sharpe_ratio', 0.0):.2f}",
                                   })
    workflow_result["log_errors"] = log_errors
    return workflow_result