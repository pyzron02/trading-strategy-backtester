#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for the unified workflow.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import logging
import datetime
from functools import wraps
from typing import Dict, List, Any, Optional, Union, Tuple

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
project_root = os.path.dirname(src_dir)  # Go up to project root
if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the logging system
from engine.logging_system import LoggingSystem

# Initialize the logging system
logging_system = LoggingSystem(
    console_output=True,
    file_output=True,
    default_level='INFO',
    async_logging=True
)
# Register the workflows component if it's not already in the list
if 'workflows' not in LoggingSystem.COMPONENTS:
    LoggingSystem.COMPONENTS.append('workflows')
logger = logging_system.get_logger('workflows')

# Dictionary to track active file handlers for each output directory
_output_file_handlers = {}

def setup_output_dir_logging(output_dir, strategy_name, workflow_name):
    """Set up a file handler for the logger to write to the output directory.
    
    Args:
        output_dir: Directory to save log file
        strategy_name: Name of the strategy
        workflow_name: Name of the workflow
        
    Returns:
        The file handler that was added to the logger
    """
    if not output_dir:
        return None
        
    # Create a unique key for this output directory
    handler_key = f"{output_dir}_{strategy_name}_{workflow_name}"
    
    # Check if we already have a handler for this directory
    if handler_key in _output_file_handlers:
        return _output_file_handlers[handler_key]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file path - use only one consistent naming pattern
    workflow_name_simple = workflow_name.lower().replace(" ", "_") 
    log_file = os.path.join(output_dir, f"{strategy_name}_{workflow_name_simple}.log")
    
    # Create and configure file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)  # Capture all log levels
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Store handler for future reference
    _output_file_handlers[handler_key] = file_handler
    
    return file_handler

def remove_output_dir_logging(output_dir, strategy_name, workflow_name):
    """Remove the file handler for the specified output directory.
    
    Args:
        output_dir: Directory of the log file
        strategy_name: Name of the strategy
        workflow_name: Name of the workflow
    """
    handler_key = f"{output_dir}_{strategy_name}_{workflow_name}"
    
    if handler_key in _output_file_handlers:
        file_handler = _output_file_handlers[handler_key]
        logger.removeHandler(file_handler)
        file_handler.close()
        del _output_file_handlers[handler_key]

# ====== Report Utilities ======

def print_header(title, width=80):
    """Print a formatted header."""
    header = "\n" + "=" * width + "\n" + title.center(width) + "\n" + "=" * width + "\n"
    logger.info(header)

def print_section(title, width=80):
    """Print a formatted section header."""
    section = "\n" + "-" * width + "\n" + title + "\n" + "-" * width
    logger.info(section)

def print_workflow_log(workflow_name, strategy_name, tickers, start_date, end_date, status="STARTED", additional_info=None):
    """Print detailed workflow execution log.
    
    Args:
        workflow_name: Name of the workflow being executed
        strategy_name: Name of the strategy being tested
        tickers: List of ticker symbols
        start_date: Start date of the backtest
        end_date: End date of the backtest
        status: Status of the workflow (STARTED, COMPLETED, FAILED)
        additional_info: Additional information to include in the log
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_width = 100
    
    log_header = f"\n{'*' * log_width}\n"
    log_header += f"* WORKFLOW {status}: {workflow_name}\n"
    log_header += f"* Timestamp: {timestamp}\n"
    log_header += f"* Strategy: {strategy_name}\n"
    log_header += f"* Tickers: {', '.join(tickers)}\n"
    log_header += f"* Period: {start_date} to {end_date}\n"
    
    if additional_info:
        for key, value in additional_info.items():
            log_header += f"* {key}: {value}\n"
    
    log_header += f"{'*' * log_width}\n"
    
    # Extract output directory from additional_info if available
    output_dir = None
    if additional_info and "output_dir" in additional_info:
        output_dir = additional_info["output_dir"]
    
    # Setup logging to the output directory if this is the start of a workflow
    if status == "STARTED" and output_dir:
        setup_output_dir_logging(output_dir, strategy_name, workflow_name)
    
    # Log the header
    logger.info(log_header)
    
    # Save to the general workflow log file
    log_dir = os.path.join(project_root, "logs", "workflows")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"workflow_log_{datetime.datetime.now().strftime('%Y%m%d')}.txt")
    with open(log_file, "a") as f:
        f.write(log_header)
    
    # Also save to the output directory summary file (single concise log file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        workflow_name_simple = workflow_name.lower().replace(" ", "_")
        workflow_log_file = os.path.join(output_dir, f"{strategy_name}_{workflow_name_simple}_log.txt")
        
        # Determine if we need to create a new file or append to existing one
        write_mode = "a"  # Default to append
        if status == "STARTED" and os.path.exists(workflow_log_file):
            # For a new run, create a new file
            write_mode = "w"
        
        with open(workflow_log_file, write_mode) as f:
            f.write(log_header)
    
    # Remove the output directory logging if this is the end of a workflow
    if (status == "COMPLETED" or status == "FAILED") and output_dir:
        remove_output_dir_logging(output_dir, strategy_name, workflow_name)
    
    return log_header

def print_parameters(params, indent=0):
    """Print parameters in a readable format."""
    param_str = ""
    indent_str = " " * indent
    for key, value in params.items():
        if isinstance(value, dict):
            param_str += f"{indent_str}{key}:\n"
            param_str += print_parameters(value, indent + 2)
        else:
            param_str += f"{indent_str}{key}: {value}\n"
    
    logger.info(param_str if param_str else "No parameters available")
    return param_str

def print_metrics(metrics, indent=0):
    """Print performance metrics in a readable format."""
    metrics_str = ""
    indent_str = " " * indent
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            # Format numeric values
            if abs(value) < 0.01 and value != 0:
                # Scientific notation for very small numbers
                metrics_str += f"{indent_str}{key}: {value:.4e}\n"
            elif isinstance(value, float):
                # Four decimal places for other floats
                metrics_str += f"{indent_str}{key}: {value:.4f}\n"
            else:
                # Just print integers normally
                metrics_str += f"{indent_str}{key}: {value}\n"
        elif isinstance(value, dict):
            # Recurse for nested dictionaries
            metrics_str += f"{indent_str}{key}:\n"
            metrics_str += print_metrics(value, indent + 2)
        else:
            # Just print other types directly
            metrics_str += f"{indent_str}{key}: {value}\n"
    
    logger.info(metrics_str if metrics_str else "No metrics available")
    return metrics_str

def save_results_summary(results, filename, title="Backtest Results"):
    """
    Save a summary of the results to a file.
    
    Args:
        results: Dictionary of results
        filename: File to save to
        title: Title for the summary
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write(f"{title}" + "\n")  # Remove duplicated "Summary"
        f.write("=" * 80 + "\n\n")
        
        # Strategy information
        f.write(f"Strategy: {results.get('strategy_name', 'Unknown')}\n")
        
        # Date period - ensure this information is correctly included
        dates = results.get('dates', {})
        start_date = dates.get('start_date', results.get('start_date', 'Unknown'))
        end_date = dates.get('end_date', results.get('end_date', 'Unknown'))
        f.write(f"Period: {start_date} to {end_date}\n\n")
        
        # Parameters
        f.write("-" * 80 + "\n")
        f.write("Parameters:" + "\n")
        f.write("-" * 80 + "\n")
        parameters = results.get('parameters', {})
        for key, value in parameters.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Metrics
        f.write("-" * 80 + "\n")
        f.write("Performance Metrics:" + "\n")
        f.write("-" * 80 + "\n")
        metrics = results.get('metrics', {})
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Format special values like infinity
                if str(value) == 'inf' or str(value) == 'Infinity':
                    f.write(f"{key}: Infinity\n")
                elif str(value) == '-inf' or str(value) == '-Infinity':
                    f.write(f"{key}: -Infinity\n")
                elif str(value) == 'nan' or str(value) == 'NaN':
                    f.write(f"{key}: N/A\n")
                else:
                    f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Results summary saved to {filename}")

# ====== Timing Utilities ======

class Timing:
    """Class to track execution timing."""
    timings = {}
    
    @classmethod
    def record(cls, name, duration):
        """Record the duration of a step."""
        if name not in cls.timings:
            cls.timings[name] = []
        cls.timings[name].append(duration)
    
    @classmethod
    def get_average(cls, name):
        """Get the average duration of a step."""
        if name not in cls.timings or not cls.timings[name]:
            return None
        return sum(cls.timings[name]) / len(cls.timings[name])
    
    @classmethod
    def get_total(cls, name):
        """Get the total duration of a step."""
        if name not in cls.timings or not cls.timings[name]:
            return None
        return sum(cls.timings[name])
    
    @classmethod
    def get_count(cls, name):
        """Get the number of times a step was executed."""
        if name not in cls.timings:
            return 0
        return len(cls.timings[name])
    
    @classmethod
    def reset(cls):
        """Reset all timings."""
        cls.timings = {}

def time_execution(description):
    """Decorator to time the execution of a function and log the result."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Started {description}")
            start_time = datetime.datetime.now()
            try:
                result = func(*args, **kwargs)
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).total_seconds()
                Timing.record(description, duration)
                logger.info(f"Completed {description} (timing information unavailable)")
                logger.info(f"Performance: {func.__name__}_{description} took {duration:.4f} seconds")
                return result
            except Exception as e:
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).total_seconds()
                Timing.record(f"{description}_error", duration)
                logger.error(f"Error in {description} after {duration:.4f} seconds: {str(e)}")
                raise  # Re-raise the exception
        return wrapper
    return decorator

# ====== File Utilities ======

def find_strategy_param_file(strategy_name: str) -> Optional[str]:
    """
    Find the parameter file for a strategy based on the strategy name.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Path to the parameter file, or None if not found
    """
    # Also check with snake_case
    strategy_snake_case = ''.join(['_'+c.lower() if c.isupper() else c.lower() for c in strategy_name]).lstrip('_')
    
    # Special case for MACrossover => ma_crossover
    special_cases = {}
    if strategy_name == "MACrossover":
        special_cases["ma_crossover"] = True
    
    # Search paths in order of preference
    search_paths = [
        os.path.join(project_root, "parameters", f"{strategy_name.lower()}_params.json"),
        os.path.join(project_root, "parameters", f"{strategy_name}_params.json"),
        os.path.join(project_root, "input", "parameters", f"{strategy_name.lower()}_params.json"),
        os.path.join(project_root, "input", "parameters", f"{strategy_name}_params.json"),
        os.path.join(project_root, "input", "parameters", f"{strategy_snake_case}_params.json"),
        os.path.join(project_root, "input", f"{strategy_name.lower()}_params.json"),
        os.path.join(project_root, "input", f"{strategy_name}_params.json"),
        os.path.join(project_root, "input", f"{strategy_snake_case}_params.json"),
        os.path.join(project_root, "src", "strategies", "parameters", f"{strategy_name.lower()}_params.json"),
        os.path.join(project_root, "src", "strategies", "parameters", f"{strategy_name}_params.json"),
        os.path.join(project_root, "src", "examples", f"{strategy_name.lower()}_params.json"),
        os.path.join(project_root, "src", "examples", f"{strategy_name}_params.json"),
    ]
    
    # Add special case paths
    if strategy_name == "MACrossover":
        special_paths = [
            os.path.join(project_root, "input", "parameters", "ma_crossover_params.json"),
            os.path.join(project_root, "input", "ma_crossover_params.json"),
            os.path.join(project_root, "src", "examples", "ma_crossover_params.json"),
        ]
        search_paths = special_paths + search_paths
    
    # Debug output for parameter file search
    logger.info(f"Looking for parameter file for strategy: {strategy_name}")
    for path in search_paths:
        logger.info(f"  Checking path: {path} (exists: {os.path.exists(path)})")
        if os.path.exists(path):
            logger.info(f"  Found parameter file at: {path}")
            return path
            
    # If not found, return None
    logger.warning(f"No parameter file found for strategy {strategy_name}")
    return None

# ====== Progress Callback ======

def progress_callback(current, total, description="Progress"):
    """
    Default progress callback for lengthy operations.
    
    Args:
        current: Current progress value
        total: Total progress value
        description: Description of the operation
    """
    if total > 0:
        percentage = (current / total) * 100
        logger.info(f"{description}: {current}/{total} ({percentage:.1f}%)")
    else:
        logger.info(f"{description}: {current}") 