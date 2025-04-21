#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple backtest workflow module.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import datetime
import uuid

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
    logger, logging_system, print_workflow_log, adapt_strategy_parameters,
    setup_output_dir_logging, remove_output_dir_logging
)

# Import engine components
from engine.run_backtest import run_backtest

def convert_grid_to_single_values(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert parameter grids to single values by taking the first value from each list.
    
    Args:
        parameters: Dictionary of parameters, potentially containing lists
        
    Returns:
        Dictionary with single values for each parameter
    """
    single_params = {}
    for key, value in parameters.items():
        if isinstance(value, list):
            if value:  # Check if list is not empty
                single_params[key] = value[0]  # Take the first value
                logger.info(f"Parameter '{key}' is a list. Using first value: {value[0]}")
            else:
                single_params[key] = None
                logger.warning(f"Parameter '{key}' is an empty list. Setting to None.")
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
    # Construct the path to stock_data.csv
    stock_data_path = os.path.join(project_root, data_dir, "stock_data.csv")
    
    # Check if stock_data.csv exists
    if not os.path.exists(stock_data_path):
        logger.info(f"stock_data.csv not found at {stock_data_path}. Generating it...")
        try:
            from data_preprocessing.data_setup import fetch_stock_data
            fetch_stock_data(tickers, start_date, end_date)
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
                fetch_stock_data(tickers, start_date, end_date)
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
    plot=True,
    verbose=False,
    initial_capital=100000.0,
    commission=0.001,
    data_dir="input",
    slippage=0.0,
    enhanced_plots=False,
    optimize_sharpe=False,
    live_mode=False,
    additional_data=None,
    progress_callback=None,
    progress_file=None,
    stock_csv=None,
    _temp_files_to_cleanup=None,
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
        enhanced_plots: Whether to create enhanced plots
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
    # Use strategy if provided, otherwise use strategy_name
    if strategy is not None and strategy_name is None:
        strategy_name = strategy
    elif strategy is None and strategy_name is None:
        return {
            "status": "error",
            "message": "Either strategy or strategy_name must be provided"
        }
    
    # Track new temporary files if not already tracking
    if _temp_files_to_cleanup is None:
        _temp_files_to_cleanup = []
    
    # Create a unique output directory if none is provided
    if not output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = str(uuid.uuid4())[:8]  # For uniqueness
        output_dir = os.path.join(project_root, "output", f"{strategy_name}_simple_{timestamp}_{run_id}")
        logger.info(f"Creating unique output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging for this run
    setup_output_dir_logging(output_dir, strategy_name, "simple")
    
    # Log the start of the workflow
    print_header("SIMPLE WORKFLOW")
    print_workflow_log("Simple", strategy_name, tickers, start_date, end_date)
    
    # Add parameters to additional info if provided
    additional_info = {}
    if param_file:
        additional_info["param_file"] = param_file
    if parameters:
        additional_info["parameters"] = parameters
    
    # Create progress tracking file if specified
    if progress_file:
        # Initialize progress tracking
        with open(progress_file, 'w') as f:
            json.dump({
                "progress": 0,
                "status": "Starting",
                "current_step": "Initializing",
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=4)
    
    if not param_file:
        param_file = find_strategy_param_file(strategy_name)
        if not param_file:
            # Check if we have parameters directly specified
            if parameters:
                # Create a temporary parameter file
                temp_param_file = os.path.join(project_root, "input", "parameters", 
                                             f"{strategy_name.lower()}_params_temp.json")
                
                try:
                    with open(temp_param_file, 'w') as f:
                        json.dump(parameters, f, indent=4)
                    logger.info(f"Created temporary parameter file from provided parameters: {temp_param_file}")
                    param_file = temp_param_file
                    # Track for cleanup
                    _temp_files_to_cleanup.append(temp_param_file)
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
            logger.error(f"Error loading parameters from {param_file}: {str(e)}")
            return {"status": "error", "message": f"Error loading parameters: {str(e)}"}
    
    # Override with any directly provided parameters
    if parameters:
        # Only use valid parameters according to strategy adapter
        param_manager = ParameterManager()
        adapted_params = param_manager.adapt_strategy_parameters(strategy_name, parameters)
        strategy_params.update(adapted_params)
        logger.info("Updated parameters with provided values")
        logger.debug(f"Updated parameters: {strategy_params}")
    
    # Check for parameter grids and convert to single values if needed
    has_grid = any(isinstance(v, list) for v in strategy_params.values())
    if has_grid:
        logger.info("Parameter grid detected in simple workflow. Converting to single values.")
        original_params = strategy_params.copy()
        strategy_params = convert_grid_to_single_values(strategy_params)
        logger.info("Converted parameters from grid to single values.")
    
    # Print parameters
    print_section("Strategy Parameters")
    print_parameters(strategy_params)
    
    try:
        # Check if stock_data.csv exists but don't regenerate it
        stock_data_path = os.path.join(data_dir, "stock_data.csv")
        if os.path.exists(stock_data_path):
            stock_csv = stock_data_path
            logger.info(f"Using existing stock data from: {stock_csv}")
        else:
            error_msg = f"Stock data file not found at: {stock_data_path}. Please run data_setup.py first."
            logger.error(error_msg)
            
            # Log workflow failure
            print_workflow_log(
                workflow_name="Simple Workflow",
                strategy_name=strategy_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                status="FAILED",
                additional_info={"error": error_msg}
            )
            
            return {
                "status": "error",
                "message": error_msg,
                "output_dir": output_dir
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
            plot=plot,
            initial_capital=initial_capital,
            commission=commission,
            data_dir=data_dir,
            verbose=verbose,
            slippage=slippage,
            enhanced_plots=enhanced_plots,
            optimize_sharpe=optimize_sharpe,
            live_mode=live_mode,
            additional_data=additional_data
        )
        
        if not backtest_result:
            error_msg = "Backtest failed to produce valid results"
            logger.error(error_msg)
            
            # Log workflow failure
            print_workflow_log(
                workflow_name="Simple Workflow",
                strategy_name=strategy_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                status="FAILED",
                additional_info={"error": error_msg}
            )
            
            return {
                "status": "error", 
                "message": error_msg,
                "output_dir": output_dir
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
        
        # Log workflow failure
        print_workflow_log(
            workflow_name="Simple Workflow",
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            status="FAILED",
            additional_info={"error": error_msg}
        )
        
        return {
            "status": "error",
            "message": error_msg,
            "output_dir": output_dir
        }
    
    # Reset logging level if it was changed
    if verbose:
        logging_system.set_level('INFO', 'workflows')
    
    # Create a combined result
    workflow_result = {
        "status": "success",
        "strategy_name": strategy_name,
        "dates": {
            "start_date": start_date,
            "end_date": end_date
        },
        "parameters": strategy_params,  # Original parameters
        "metrics": metrics,
        "output_dir": output_dir
    }
    
    # Log workflow completion
    completion_info = {
        "total_return": f"{metrics.get('total_return', 0.0):.2%}",
        "sharpe_ratio": f"{metrics.get('sharpe_ratio', 0.0):.2f}",
        "output_dir": output_dir
    }
    print_workflow_log(
        workflow_name="Simple Workflow",
        strategy_name=strategy_name,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        status="COMPLETED",
        additional_info=completion_info
    )
    
    # Clean up temporary files
    for temp_file in _temp_files_to_cleanup:
        try:
            os.remove(temp_file)
            logger.info(f"Cleaned up temporary file: {temp_file}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary file: {str(e)}")
    
    return workflow_result 