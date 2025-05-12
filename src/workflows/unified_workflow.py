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
import uuid

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
project_root = os.path.dirname(src_dir)  # Go up to project root
if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import workflow modules
from workflows.workflow_utils import (
    logger, logging_system, find_strategy_param_file,
    check_logs_for_errors, print_error_report
)
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
    Run a workflow based on the workflow type.
    
    Args:
        workflow_type: Type of workflow to run (simple, optimization, monte_carlo, complete)
        **kwargs: Additional arguments for the workflow
    
    Returns:
        Dictionary with workflow results
    """
    # Initialize temporary files tracking
    temp_files_to_cleanup = kwargs.pop('_temp_files_to_cleanup', [])
    
    # Handle simple values that may have been passed as strings
    for k, v in kwargs.items():
        if isinstance(v, str):
            if v.lower() == 'true':
                kwargs[k] = True
            elif v.lower() == 'false':
                kwargs[k] = False
            elif v.isdigit():
                kwargs[k] = int(v)
            elif v.replace('.', '', 1).isdigit() and v.count('.') < 2:
                kwargs[k] = float(v)
    
    param_file = kwargs.get('param_file')
    if param_file and is_parameter_grid(param_file):
        if workflow_type == "simple" and param_file and is_parameter_grid(param_file):
            logger.warning(
                f"Parameter file '{param_file}' contains grid values, which are better suited for optimization. "
                f"Continuing, but only the first value in each grid will be used."
            )
    
    # Check if this is a nested workflow call from a complete workflow
    is_nested = kwargs.pop('_is_nested_workflow', False)
    # Log nested status for debugging
    logger.debug(f"Running {workflow_type} workflow, nested={is_nested}")
    
    # Get the output directory from kwargs
    output_dir = kwargs.get('output_dir')
    
    # Filter workflow-specific parameters
    filtered_kwargs = kwargs.copy()
    
    # Define workflow-specific parameters
    workflow_specific_params = {
        "simple": ['plot'],
        "optimization": ['n_trials', 'optimization_metric', 'max_combinations'],
        "monte_carlo": ['n_simulations', 'keep_permuted_data', 'enhanced_plots', 'plot'],
        "walkforward": ['window_size', 'step_size', 'n_trials', 'optimization_metric'],
        "complete": []  # Complete workflow can use all parameters
    }
    
    # Remove parameters that are not supported by the current workflow
    if workflow_type != "complete":  # Complete workflow accepts all parameters
        allowed_params = workflow_specific_params.get(workflow_type, [])
        # Add common parameters that are allowed for all workflows
        common_params = ['strategy_name', 'strategy', 'tickers', 'start_date', 'end_date', 
                        'output_dir', 'verbose', 'initial_capital', 'commission', 
                        'param_file', 'data_dir', '_temp_files_to_cleanup', '_is_nested_workflow']
        
        allowed_params.extend(common_params)
        
        # Remove parameters that are not in the allowed list
        parameters_to_remove = []
        for param in filtered_kwargs:
            if param not in allowed_params:
                parameters_to_remove.append(param)
        
        for param in parameters_to_remove:
            logger.debug(f"Removing '{param}' parameter from {workflow_type} workflow")
            filtered_kwargs.pop(param, None)
    
    # Add _temp_files_to_cleanup back to kwargs
    filtered_kwargs['_temp_files_to_cleanup'] = temp_files_to_cleanup
    
    try:
        if workflow_type == "simple":
            result = run_simple_workflow(**filtered_kwargs)
        elif workflow_type == "optimization":
            result = run_optimization_workflow(**filtered_kwargs)
        elif workflow_type == "monte_carlo":
            result = run_monte_carlo_workflow(**filtered_kwargs)
        elif workflow_type == "complete":
            result = run_complete_workflow(**filtered_kwargs)
        elif workflow_type == "walkforward":
            # Ensure we're passing the plot parameter correctly
            if 'plot' not in filtered_kwargs:
                # Default to False if not provided
                filtered_kwargs['plot'] = False
            result = run_walkforward_workflow(**filtered_kwargs)
        else:
            logger.error(f"Unknown workflow type: {workflow_type}")
            return {"status": "error", "message": f"Unknown workflow type: {workflow_type}"}
        
        # If output directory exists and we have a valid result, check for log errors
        if output_dir and os.path.exists(output_dir) and result.get("status") == "success":
            # Check logs for errors
            logger.info("Checking logs for errors...")
            error_logs = check_logs_for_errors(output_dir)
            
            if error_logs:
                # Add log errors to the results
                if "results" in result:
                    result["results"]["log_errors"] = {
                        "count": sum(len(errors) for errors in error_logs.values()),
                        "files": len(error_logs)
                    }
                else:
                    result["log_errors"] = {
                        "count": sum(len(errors) for errors in error_logs.values()),
                        "files": len(error_logs)
                    }
                
                # Generate error report and save to file
                error_report_path = os.path.join(output_dir, "error_report.txt")
                print_error_report(error_logs, error_report_path)
                logger.warning(f"Found errors in logs. Error report saved to: {error_report_path}")
            else:
                logger.info("No errors found in logs.")
                if "results" in result:
                    result["results"]["log_errors"] = {"count": 0, "files": 0}
                else:
                    result["log_errors"] = {"count": 0, "files": 0}
        
        return result
    except Exception as e:
        logger.exception(f"Error in {workflow_type} workflow: {str(e)}")
        
        # Check logs for errors even on exception
        if output_dir and os.path.exists(output_dir):
            # Check logs for errors
            logger.info("Checking logs for errors...")
            error_logs = check_logs_for_errors(output_dir)
            
            if error_logs:
                # Generate error report and save to file
                error_report_path = os.path.join(output_dir, "error_report.txt")
                print_error_report(error_logs, error_report_path)
                logger.warning(f"Found errors in logs. Error report saved to: {error_report_path}")
        
        return {"status": "error", "message": f"Error in {workflow_type} workflow: {str(e)}"}

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
    Process the configuration to extract workflow parameters.
    
    Args:
        config: Dictionary with the configuration
        
    Returns:
        Dictionary with workflow parameters for each strategy
    """
    if not config or 'strategies' not in config:
        logger.error("Invalid config: 'strategies' section not found")
        return {}
    
    workflow_type = config.get('workflow_type', 'simple')
    common_params = config.get('common_params', {})
    
    # Base output directory - will be used as parent for strategy-specific dirs
    base_output_dir = common_params.get('output_dir')
    if not base_output_dir:
        base_output_dir = os.path.join(project_root, "output")
        os.makedirs(base_output_dir, exist_ok=True)
    
    # Initialize temporary files list to track files that need cleanup
    temp_files_to_cleanup = []
    
    # Define workflow-specific parameters
    workflow_specific_params = {
        "simple": ['plot'],
        "optimization": ['n_trials', 'optimization_metric', 'max_combinations'],
        "monte_carlo": ['n_simulations', 'keep_permuted_data', 'enhanced_plots'],
        "walkforward": ['window_size', 'step_size', 'n_trials', 'optimization_metric'],
        "complete": []  # Complete workflow can use all parameters
    }
    
    workflow_params = {}
    
    for strategy_name, strategy_config in config['strategies'].items():
        params = common_params.copy()
        params['strategy'] = strategy_name
        
        # Create a unique output directory for each strategy
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = str(uuid.uuid4())[:8]  # First 8 chars of a UUID for uniqueness
        
        # Create strategy-specific output directory under the base output dir
        strategy_output_dir = os.path.join(base_output_dir, f"{strategy_name}_{workflow_type}_{timestamp}_{run_id}")
        params['output_dir'] = strategy_output_dir
        
        # Ensure the output directory exists
        os.makedirs(strategy_output_dir, exist_ok=True)
        logger.info(f"Created output directory for {strategy_name}: {strategy_output_dir}")
        
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
                # Track temporary file for cleanup
                temp_files_to_cleanup.append(param_file)
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
                # Track temporary file for cleanup
                temp_files_to_cleanup.append(grid_file)
                
                # For optimization workflows, use the grid file as the param file
                if workflow_type in ['optimization', 'complete', 'walkforward']:
                    params['param_file'] = grid_file
            except Exception as e:
                logger.error(f"Error creating grid file: {str(e)}")
        
        # Add workflow-specific parameters
        if workflow_type in strategy_config:
            # Only add parameters that are relevant to this workflow type
            allowed_params = workflow_specific_params.get('complete', [])  # All params for complete workflow
            if workflow_type != 'complete':
                allowed_params = workflow_specific_params.get(workflow_type, [])
            
            # Add only the parameters that are allowed for this workflow type
            for key, value in strategy_config[workflow_type].items():
                if workflow_type == 'complete' or key in allowed_params:
                    params[key] = value
                else:
                    logger.debug(f"Skipping parameter '{key}' as it's not applicable to {workflow_type} workflow")
            
        # For complete workflow, pass optimization and monte_carlo config if available
        if workflow_type == 'complete':
            # Pass optimization configuration if available
            if 'optimization' in strategy_config:
                # Copy optimization parameters to the main parameters
                for key, value in strategy_config['optimization'].items():
                    # Only add if not already set by the complete workflow parameters
                    if key not in params:
                        params[key] = value
                logger.info(f"Passing optimization configuration to complete workflow: {strategy_config['optimization']}")
                
            # Pass monte_carlo configuration if available
            if 'monte_carlo' in strategy_config:
                params['monte_carlo_config'] = strategy_config['monte_carlo']
                logger.info(f"Passing Monte Carlo configuration to complete workflow: {strategy_config['monte_carlo']}")
        
        # Add temp files list to parameters for cleanup
        params['_temp_files_to_cleanup'] = temp_files_to_cleanup
        
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

def cleanup_temporary_files(temp_files):
    """
    Clean up temporary parameter files and grid files created during workflow execution.
    
    Args:
        temp_files: List of temporary file paths to delete
    """
    if not temp_files:
        return
    
    # Filter out any files from workflow_configs directory
    files_to_delete = []
    files_skipped = []
    
    for file_path in temp_files:
        if os.path.exists(file_path):
            # Skip files in the workflow_configs directory
            if "workflow_configs" in file_path:
                files_skipped.append(file_path)
            else:
                files_to_delete.append(file_path)
    
    if files_skipped:
        logger.info(f"Skipping cleanup of {len(files_skipped)} workflow config files")
        for file_path in files_skipped:
            logger.debug(f"Preserved file: {file_path}")
    
    logger.info(f"Cleaning up {len(files_to_delete)} temporary parameter/grid files")
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            logger.debug(f"Deleted temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {file_path}: {str(e)}")

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
    
    # Keep track of all temporary files
    all_temp_files = []
    
    # Run workflows for each strategy
    for strategy_name, workflow_config in workflow_params.items():
        current_strategy += 1
        workflow_type = workflow_config['workflow_type']
        params = workflow_config['params']
        
        # Extract temp files from params for cleanup
        temp_files = params.get('_temp_files_to_cleanup', [])
        if temp_files:
            all_temp_files.extend(temp_files)
        
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
    
    # Clean up temporary parameter files
    cleanup_temporary_files(all_temp_files)
    
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