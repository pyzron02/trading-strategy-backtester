#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified workflow for backtest, optimization, and walk-forward validation.
Main entry point for the workflow system.
"""
import os
import sys
import json
from typing import Dict, Any, Optional

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
project_root = os.path.dirname(src_dir)  # Go up to project root
if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import workflow modules
from workflows.workflow_utils import logger, logging_system
from workflows.simple_workflow import run_simple_workflow, ensure_data_available
from workflows.optimization_workflow import run_optimization_workflow
from workflows.monte_carlo_workflow import run_monte_carlo_workflow
from workflows.walkforward_workflow import run_walkforward_workflow
from workflows.complete_workflow import run_complete_workflow
# Remove circular import
# from workflows.cli import run_cli

def is_parameter_grid(param_file: str) -> bool:
    """
    Check if a parameter file contains grid values (lists of values).
    
    Args:
        param_file: Path to the parameter file
        
    Returns:
        True if the file contains grid values, False otherwise
    """
    if not param_file or not os.path.exists(param_file):
        return False
        
    try:
        with open(param_file, 'r') as f:
            params = json.load(f)
        
        # Check if any parameter is a list
        return any(isinstance(value, list) for value in params.values())
    except Exception as e:
        logger.warning(f"Error checking parameter file {param_file}: {str(e)}")
        return False

def cleanup_duplicate_logs(output_dir: str, strategy_name: str, workflow_type: str):
    """
    Clean up duplicate log files to prevent clutter.
    
    Args:
        output_dir: Directory containing the log files
        strategy_name: Name of the strategy
        workflow_type: Type of workflow being run
    """
    if not output_dir or not os.path.exists(output_dir):
        return
        
    # Define duplicate log files to check
    workflow_name_simple = workflow_type.lower().replace(" ", "_")
    
    duplicate_logs = [
        f"{strategy_name}_{workflow_name_simple}_workflow.log",  # Extra log with 'workflow' in name
        f"{strategy_name}_{workflow_type}.log",                 # Log with original workflow type
    ]
    
    # Keep only the main log file
    main_log = f"{strategy_name}_{workflow_name_simple}.log"
    
    # Check and remove duplicate log files
    for log_file in duplicate_logs:
        full_path = os.path.join(output_dir, log_file)
        if os.path.exists(full_path) and log_file != main_log:
            try:
                os.remove(full_path)
                logger.debug(f"Removed duplicate log file: {full_path}")
            except Exception as e:
                logger.warning(f"Failed to remove duplicate log file {full_path}: {str(e)}")

def run_unified_workflow(workflow_type, **kwargs):
    """
    Run a specific workflow with the provided parameters.
    
    Args:
        workflow_type: Type of workflow to run (simple, optimization, monte_carlo, walkforward, complete)
        **kwargs: Workflow-specific parameters
        
    Returns:
        Dictionary with workflow results
    """
    # Initialize parameters
    result = None
    
    # Get key parameters
    tickers = kwargs.get('tickers')
    start_date = kwargs.get('start_date')
    end_date = kwargs.get('end_date')
    data_dir = kwargs.get('data_dir', 'input')
    param_file = kwargs.get('param_file')
    
    # Log that we're using stock_data.csv as the default input
    logger.info("All workflows will prioritize using stock_data.csv as the data source")
    
    # Check if stock_data.csv exists but do not regenerate it
    stock_csv = None
    try:
        stock_data_path = os.path.join(data_dir, "stock_data.csv")
        if os.path.exists(stock_data_path):
            stock_csv = stock_data_path
            logger.info(f"Using existing stock data at: {stock_csv}")
        else:
            logger.warning(f"Stock data file not found at: {stock_data_path}. Please run data_setup.py first.")
    except Exception as e:
        logger.warning(f"Error accessing stock_data.csv: {e}")
    
    # Check if the workflow type should be auto-selected based on parameter file
    if workflow_type == "simple" and param_file and is_parameter_grid(param_file):
        logger.warning(
            f"Parameter file '{param_file}' contains grid values, which are better suited for optimization. "
            f"The simple workflow will use only the first value from each parameter grid."
        )
    
    # Create a copy of kwargs to avoid modifying the original
    workflow_kwargs = kwargs.copy()
    
    # Remove stock_csv from kwargs to avoid errors with functions that don't accept it
    if 'stock_csv' in workflow_kwargs:
        del workflow_kwargs['stock_csv']
    
    # Run the selected workflow
    if workflow_type == "simple":
        result = run_simple_workflow(**workflow_kwargs)
    elif workflow_type == "optimization":
        result = run_optimization_workflow(**workflow_kwargs)
    elif workflow_type == "monte_carlo":
        result = run_monte_carlo_workflow(**workflow_kwargs)
    elif workflow_type == "walkforward":
        result = run_walkforward_workflow(**workflow_kwargs)
    elif workflow_type == "complete":
        result = run_complete_workflow(**workflow_kwargs)
    else:
        logger.error(f"Unknown workflow type: {workflow_type}")
        result = {"status": "error", "message": f"Unknown workflow type: {workflow_type}"}
    
    # Clean up duplicate log files
    output_dir = kwargs.get('output_dir')
    strategy_name = kwargs.get('strategy_name')
    if output_dir and strategy_name and result and result.get('status') == 'success':
        cleanup_duplicate_logs(output_dir, strategy_name, workflow_type)
    
    return result

def main():
    """
    Main entry point for the unified workflow.
    Run the command-line interface.
    """
    # Import run_cli here to avoid circular import
    from workflows.cli import run_cli
    run_cli()

if __name__ == "__main__":
    main() 