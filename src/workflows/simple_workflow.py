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
    logger, logging_system, print_workflow_log, adapt_strategy_parameters
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
    strategy_name: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_dir: str,
    parameters: Optional[Dict[str, Any]] = None,
    param_file: Optional[str] = None,
    plot: bool = True,
    verbose: bool = False,
    initial_capital: float = 100000.0,
    commission: float = 0.001,
    data_dir: str = "input"
) -> Dict[str, Any]:
    """
    Run a simple workflow for the given strategy.
    
    Args:
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
    
    Returns:
        Dict containing the workflow results
    """
    # Log workflow start
    additional_info = {
        "output_dir": output_dir,
        "plot": plot,
        "initial_capital": initial_capital,
        "commission": commission,
        "data_dir": data_dir
    }
    if param_file:
        additional_info["param_file"] = param_file
        
    print_workflow_log(
        workflow_name="Simple Workflow",
        strategy_name=strategy_name,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        status="STARTED",
        additional_info=additional_info
    )
    
    print_header(f"Simple Workflow: {strategy_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set logging level based on verbose flag
    if verbose:
        logging_system.set_level('DEBUG', 'workflows')
    
    # If parameters not provided, load from file
    if not parameters:
        if not param_file:
            param_file = find_strategy_param_file(strategy_name)
            if not param_file:
                error_msg = f"No parameter file found for strategy: {strategy_name}"
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
                
                return {"status": "error", "message": error_msg}
        
        try:
            with open(param_file, 'r') as f:
                parameters = json.load(f)
        except Exception as e:
            error_msg = f"Error loading parameter file: {str(e)}"
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
            
            return {"status": "error", "message": error_msg}
    
    # Check for parameter grids and convert to single values if needed
    has_grid = any(isinstance(v, list) for v in parameters.values())
    if has_grid:
        logger.info("Parameter grid detected in simple workflow. Converting to single values.")
        original_params = parameters.copy()
        parameters = convert_grid_to_single_values(parameters)
        logger.info("Converted parameters from grid to single values.")
    
    # Print parameters
    print_section("Strategy Parameters")
    print_parameters(parameters)
    
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
        
        # Adapt parameters for this strategy if needed
        adapted_parameters = adapt_strategy_parameters(strategy_name, parameters)
        
        # If parameters were adapted, log this
        if adapted_parameters != parameters and adapted_parameters:
            logger.info(f"Using adapted parameters for {strategy_name}")
            print_section("Adapted Strategy Parameters")
            print_parameters(adapted_parameters)
        
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
            parameters=adapted_parameters,
            stock_csv=stock_csv,  # Pass the explicit stock_csv path
            plot=plot,
            initial_capital=initial_capital,
            commission=commission,
            data_dir=data_dir,
            verbose=verbose
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
        "parameters": parameters,  # Original parameters
        "adapted_parameters": adapted_parameters,  # Adapted parameters used in the backtest
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
    
    return workflow_result 