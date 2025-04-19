#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified workflow for backtest, optimization, and walk-forward validation.
Main entry point for the workflow system.
"""
import os
import sys
import json
import datetime
import argparse
from typing import Dict, Any, Optional, List, Union

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
project_root = os.path.dirname(src_dir)  # Go up to project root
if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import workflow modules
from workflows.workflow_utils import logger, logging_system, find_strategy_param_file
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

def load_config_file(config_file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a config file.
    
    Args:
        config_file_path: Path to the config file
        
    Returns:
        Dictionary with configuration parameters
    """
    if not os.path.exists(config_file_path):
        logger.error(f"Config file not found: {config_file_path}")
        return {}
        
    try:
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded configuration from: {config_file_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config file {config_file_path}: {str(e)}")
        return {}

def process_config(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Process configuration dictionary to extract workflow parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with processed workflow parameters
    """
    if not config:
        return {}
    
    workflow_params = {}
    
    # Get workflow type
    workflow_type = config.get('workflow_type', 'simple')
    
    # Get common parameters
    common_params = config.get('common_params', {})
    
    # Get strategy-specific parameters
    strategies = config.get('strategies', {})
    
    # For each strategy in the config
    for strategy_name, strategy_config in strategies.items():
        # Create a new set of parameters for this strategy
        params = common_params.copy()
        
        # Update with strategy-specific parameters
        params.update({
            'strategy_name': strategy_name
        })
        
        # Set output directory if not specified
        if 'output_dir' not in params:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(project_root, "output", f"{strategy_name}_{workflow_type}_{timestamp}")
            params['output_dir'] = output_dir
            logger.info(f"No output directory specified. Using: {output_dir}")
        
        # Ensure the output directory exists
        os.makedirs(params['output_dir'], exist_ok=True)
        
        # Handle strategy parameter file
        param_file = strategy_config.get('param_file')
        if not param_file and 'parameters' in strategy_config:
            # Create a temporary parameter file from inline parameters
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            param_dir = os.path.join(project_root, "input", "parameters")
            os.makedirs(param_dir, exist_ok=True)
            param_file = os.path.join(param_dir, f"{strategy_name.lower()}_params_{timestamp}.json")
            
            try:
                with open(param_file, 'w') as f:
                    json.dump(strategy_config['parameters'], f, indent=2)
                logger.info(f"Created temporary parameter file: {param_file}")
            except Exception as e:
                logger.error(f"Error creating parameter file: {str(e)}")
                param_file = None
                
        if param_file:
            params['param_file'] = param_file
            
        # Handle grid parameter file
        grid_file = strategy_config.get('grid_file')
        if not grid_file and 'parameter_grid' in strategy_config:
            # Create a temporary grid file from inline parameters
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            grid_dir = os.path.join(project_root, "input", "parameter_grids")
            os.makedirs(grid_dir, exist_ok=True)
            grid_file = os.path.join(grid_dir, f"{strategy_name.lower()}_grid_{timestamp}.json")
            
            try:
                with open(grid_file, 'w') as f:
                    json.dump(strategy_config['parameter_grid'], f, indent=2)
                logger.info(f"Created temporary grid file: {grid_file}")
                
                # For optimization workflows, use the grid file as the param file
                if workflow_type in ['optimization', 'complete', 'walkforward']:
                    params['param_file'] = grid_file
            except Exception as e:
                logger.error(f"Error creating grid file: {str(e)}")
        
        # Add workflow-specific parameters
        if workflow_type in strategy_config:
            params.update(strategy_config[workflow_type])
            
        # Store the parameters for this strategy
        workflow_params[strategy_name] = {
            'workflow_type': workflow_type,
            'params': params
        }
    
    return workflow_params

def update_progress_file(progress_file: str, progress: int, status: str = None, current_step: str = None):
    """
    Update the progress file with current status information.
    
    Args:
        progress_file: Path to the progress file
        progress: Current progress percentage (0-100)
        status: Current status text
        current_step: Current workflow step name
    """
    if not progress_file or not os.path.exists(progress_file):
        return
    
    try:
        # Read current progress data
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        
        # Update with new values
        if progress is not None:
            progress_data['progress'] = min(100, max(0, progress))
        if status:
            progress_data['status'] = status
        if current_step:
            progress_data['current_step'] = current_step
        
        # Always update timestamp
        progress_data['last_update'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Write updated progress data
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=4)
    except Exception as e:
        logger.error(f"Error updating progress file: {e}")

def run_unified_workflow_from_config(config_file: str) -> Dict[str, Any]:
    """
    Run unified workflow using parameters from a config file.
    
    Args:
        config_file: Path to the config file
        
    Returns:
        Dictionary with workflow results
    """
    config = load_config_file(config_file)
    if not config:
        return {"status": "error", "message": f"Failed to load config file: {config_file}"}
    
    # Check for progress file in the config
    progress_file = None
    if 'frontend' in config and 'progress_file' in config['frontend']:
        progress_file = config['frontend']['progress_file']
        update_progress_file(progress_file, 10, "Processing config", "Reading configuration")
    
    workflow_params = process_config(config)
    if not workflow_params:
        return {"status": "error", "message": "No valid workflow configurations found in the config file"}
    
    results = {}
    total_strategies = len(workflow_params)
    current_strategy = 0
    
    # Run workflows for each strategy
    for strategy_name, workflow_config in workflow_params.items():
        current_strategy += 1
        workflow_type = workflow_config['workflow_type']
        params = workflow_config['params']
        
        # Update progress with strategy info
        if progress_file:
            progress_value = 10 + int(85 * (current_strategy - 1) / total_strategies)
            update_progress_file(
                progress_file, 
                progress_value,
                f"Running {workflow_type} workflow",
                f"Strategy {current_strategy}/{total_strategies}: {strategy_name}"
            )
        
        logger.info(f"Running {workflow_type} workflow for strategy: {strategy_name}")
        result = run_unified_workflow(workflow_type, **params)
        results[strategy_name] = result
        
        # Update progress after each strategy
        if progress_file:
            progress_value = 10 + int(85 * current_strategy / total_strategies)
            strategy_status = "Completed" if result.get("status") == "success" else "Failed"
            update_progress_file(
                progress_file,
                progress_value,
                f"Strategy {strategy_status}",
                f"Finished {workflow_type} for {strategy_name}"
            )
    
    # Final progress update
    if progress_file:
        update_progress_file(progress_file, 95, "Finalizing", "Generating final results")
    
    # Return combined results
    return {
        "status": "success",
        "results": results
    }

def main():
    """
    Main entry point for the unified workflow.
    Run the command-line interface.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a unified workflow with a config file")
    parser.add_argument("config_file", nargs="?", help="Path to the workflow config file")
    parser.add_argument("--progress-file", help="Path to a file for tracking progress (for frontend integration)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging_system.set_level('DEBUG')
        logger.debug("Debug logging enabled")
    
    # Check if a config file was provided
    if args.config_file and os.path.exists(args.config_file):
        config_file = args.config_file
        logger.info(f"Running with config file: {config_file}")
        
        # Update progress if progress file is provided
        if args.progress_file and os.path.exists(args.progress_file):
            try:
                with open(args.progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                # Update progress with workflow start
                progress_data.update({
                    'status': 'Running',
                    'current_step': 'Initializing workflow',
                    'progress': 5
                })
                
                with open(args.progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=4)
            except Exception as e:
                logger.error(f"Error updating progress file: {e}")
        
        # Run the workflow
        result = run_unified_workflow_from_config(config_file)
        
        # Update final status in progress file
        if args.progress_file and os.path.exists(args.progress_file):
            try:
                with open(args.progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                if result.get("status") == "success":
                    progress_data.update({
                        'status': 'Completed',
                        'current_step': 'Workflow completed successfully',
                        'progress': 100,
                        'current_step_progress': 100,
                        'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                else:
                    progress_data.update({
                        'status': 'Failed',
                        'current_step': f"Error: {result.get('message', 'Unknown error')}",
                        'progress': 100,
                        'current_step_progress': 100,
                        'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                with open(args.progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=4)
            except Exception as e:
                logger.error(f"Error updating progress file: {e}")
        
        # Log final status
        if result.get("status") == "success":
            logger.info("Workflow completed successfully")
        else:
            logger.error(f"Workflow failed: {result.get('message', 'Unknown error')}")
    else:
        # Import run_cli here to avoid circular import
        from workflows.cli import run_cli
        run_cli()

if __name__ == "__main__":
    main() 