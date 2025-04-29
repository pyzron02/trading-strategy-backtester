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
import re
from pathlib import Path

# Add the parent directory to the path so we can import from engine
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_manager import path_manager

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

def remove_output_dir_logging(output_dir=None, strategy_name=None, workflow_name=None):
    """Remove the file handler for the specified output directory.
    
    Args:
        output_dir: Directory of the log file. If None, removes all handlers.
        strategy_name: Name of the strategy. If None, removes all handlers.
        workflow_name: Name of the workflow. If None, removes all handlers.
    """
    # If any parameter is None, remove all handlers
    if output_dir is None or strategy_name is None or workflow_name is None:
        # Make a copy of keys to avoid modifying dict during iteration
        handler_keys = list(_output_file_handlers.keys())
        for key in handler_keys:
            file_handler = _output_file_handlers[key]
            logger.removeHandler(file_handler)
            file_handler.close()
            del _output_file_handlers[key]
        return
        
    # Otherwise, remove only the specific handler
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
    
    # No longer saving to general workflow log file in logs/workflows
    # All workflow logs are now saved only in their respective output directories
    
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
        parameters = results.get('parameters', {}) or {}  # Ensure parameters is a dict even if None
        for key, value in parameters.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Metrics
        f.write("-" * 80 + "\n")
        f.write("Performance Metrics:" + "\n")
        f.write("-" * 80 + "\n")
        metrics = results.get('metrics', {}) or {}  # Ensure metrics is a dict even if None
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
    
    # Search paths in order of preference using pathlib for containerization-friendly paths
    search_paths = [
        path_manager.base_dir / "parameters" / f"{strategy_name.lower()}_params.json",
        path_manager.base_dir / "parameters" / f"{strategy_name}_params.json",
        path_manager.parameters_dir / f"{strategy_name.lower()}_params.json",
        path_manager.parameters_dir / f"{strategy_name}_params.json",
        path_manager.parameters_dir / f"{strategy_snake_case}_params.json",
        path_manager.input_dir / f"{strategy_name.lower()}_params.json",
        path_manager.input_dir / f"{strategy_name}_params.json",
        path_manager.input_dir / f"{strategy_snake_case}_params.json",
        path_manager.src_dir / "strategies" / "parameters" / f"{strategy_name.lower()}_params.json",
        path_manager.src_dir / "strategies" / "parameters" / f"{strategy_name}_params.json",
        path_manager.src_dir / "examples" / f"{strategy_name.lower()}_params.json",
        path_manager.src_dir / "examples" / f"{strategy_name}_params.json",
    ]
    
    # Add special case paths
    if strategy_name == "MACrossover":
        special_paths = [
            path_manager.parameters_dir / "ma_crossover_params.json",
            path_manager.input_dir / "ma_crossover_params.json",
            path_manager.src_dir / "examples" / "ma_crossover_params.json",
        ]
        search_paths = special_paths + search_paths
    
    # Debug output for parameter file search
    logger.info(f"Looking for parameter file for strategy: {strategy_name}")
    for path in search_paths:
        logger.info(f"  Checking path: {path} (exists: {path.exists()})")
        if path.exists():
            logger.info(f"  Found parameter file at: {path}")
            return str(path)
            
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

# ====== Strategy Parameter Mapping ======

# Dictionary mapping strategy names to parameter adapters
STRATEGY_PARAMETER_MAPS = {
    'AuctionMarket': {
        # Map workflow config parameters to strategy parameters
        'param_preset': 'param_preset',  # Default, aggressive, conservative
        'value_area': 'value_area',      # Value Area percentage (0.7 = 70%)
        'use_vwap': 'use_vwap',          # Use VWAP in analysis
        'use_volume_profile': 'use_volume_profile',  # Use volume profile
        'position_size': 'position_size', # Default position size
        'risk_percent': 'risk_percent',   # Risk percentage per trade
        'use_atr_sizing': 'use_atr_sizing', # Use ATR for position sizing
        'atr_period': 'atr_period',       # ATR calculation period
    }
}

def adapt_strategy_parameters(strategy_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt parameters based on strategy-specific requirements.
    
    Args:
        strategy_name: Name of the strategy
        parameters: Dictionary of parameters from workflow
        
    Returns:
        Dictionary with adapted parameters suitable for the strategy
    """
    # If strategy doesn't need adaptation or isn't in our map, return as-is
    if strategy_name not in STRATEGY_PARAMETER_MAPS or not parameters:
        return parameters
        
    # Get parameter map for this strategy
    param_map = STRATEGY_PARAMETER_MAPS[strategy_name]
    
    # Create new parameters dictionary with only valid parameters
    adapted_params = {}
    
    # Use only parameters that are in the map
    for config_param, strategy_param in param_map.items():
        if config_param in parameters:
            adapted_params[strategy_param] = parameters[config_param]
    
    # Log the adaptation
    logger.info(f"Adapted parameters for {strategy_name} strategy")
    logger.debug(f"Original parameters: {parameters}")
    logger.debug(f"Adapted parameters: {adapted_params}")
    
    return adapted_params

def check_logs_for_errors(output_dir=None, keywords=None, context_lines=2, max_errors_per_file=20):
    """
    Check log files in the output directory for error messages.
    
    Args:
        output_dir: Directory to search for logs. If None, checks all output directories.
        keywords: List of keywords to search for. Defaults to ["ERROR", "Exception", "Traceback"]
        context_lines: Number of lines before and after each error to include for context
        max_errors_per_file: Maximum number of errors to report per file
        
    Returns:
        dict: Dictionary with log file paths as keys and lists of error messages as values
    """
    # Set default output dir to main output directory
    if output_dir is None:
        output_dir = str(path_manager.output_dir)
    
    # Default error keywords to look for
    if keywords is None:
        keywords = ["ERROR", "Exception", "Traceback"]
    
    # Additional keywords to check in summary files
    summary_keywords = ["failed", "error:", "division by zero", "failed with exception", 
                       "WORKFLOW STATUS: ERROR", "FAILED", "Critical"]
    
    # Exclude patterns for common messages that aren't actual errors
    exclude_strings = [
        "Checking logs for errors",
        "Found errors in logs",
        "No errors found in logs",
        "Error report saved to",
        "Importing strategy",
        "Log directory",
        "Async logging",
        "checking for errors",
        "error_report",
        "load_strategy_module"
    ]
    
    # Keywords that indicate critical errors vs. warnings
    critical_keywords = ["CRITICAL", "Traceback", "Exception", "AssertionError", "RuntimeError", 
                        "division by zero", "WORKFLOW STATUS: ERROR"]
    
    # Dictionary to store errors
    error_logs = {}
    
    # Flag to track if we found any log files
    found_log_files = False
    
    # Find all log files and summary files
    for root, _, files in os.walk(output_dir):
        for file in files:
            # Check log files with .log extension
            if file.endswith('.log') or (file.endswith('.txt') and "error_report" not in file):
                found_log_files = True
                
                # Skip files that are error reports themselves to avoid recursion
                if "error_report" in file:
                    continue
                
                log_path = os.path.join(root, file)
                error_entries = []
                
                try:
                    # Read all lines from the file
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        all_lines = f.readlines()
                    
                    # Dictionary to track unique error signatures to avoid duplication
                    error_signatures = {}
                    
                    # Process the file line by line
                    for i, line in enumerate(all_lines, 1):
                        # Skip lines containing exclude strings
                        if any(exclude_str in line for exclude_str in exclude_strings):
                            continue
                        
                        # Check for error patterns
                        is_error = False
                        is_critical = False
                        
                        # Use additional keywords for summary files
                        keywords_to_check = keywords
                        if file.endswith('_summary.txt') or "summary" in file:
                            keywords_to_check = keywords + summary_keywords
                        
                        for keyword in keywords_to_check:
                            if keyword.upper() in line.upper():
                                # Additional verification to ensure it's a real error
                                if keyword.upper() == "ERROR":
                                    # Must be a proper ERROR log level entry or be in a summary file
                                    if re.search(r' - ERROR - ', line) or "summary" in file.lower():
                                        is_error = True
                                else:
                                    # Exception or Traceback are clear indicators of errors
                                    is_error = True
                                    
                                # Check if this is a critical error
                                for critical_keyword in critical_keywords:
                                    if critical_keyword.upper() in line.upper():
                                        is_critical = True
                                        break
                                
                                if is_error:
                                    # Create a simplified error signature for deduplication
                                    # Strip timestamp, line numbers, and variable values
                                    error_sig = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', '', line)
                                    error_sig = re.sub(r'line \d+', 'line XXX', error_sig)
                                    error_sig = re.sub(r':\d+:', ':XXX:', error_sig)
                                    
                                    # Check if we've seen a similar error already
                                    if error_sig in error_signatures:
                                        error_signatures[error_sig]['count'] += 1
                                        # Update the most recent line number
                                        error_signatures[error_sig]['line_nums'].append(i)
                                    else:
                                        # Get context lines (before and after)
                                        context_before = []
                                        context_after = []
                                        
                                        # Get lines before error
                                        start_idx = max(0, i - context_lines - 1)
                                        for j in range(start_idx, i - 1):
                                            context_before.append(f"Line {j+1}: {all_lines[j].strip()}")
                                        
                                        # Get lines after error
                                        end_idx = min(len(all_lines), i + context_lines)
                                        for j in range(i, end_idx):
                                            context_after.append(f"Line {j+1}: {all_lines[j].strip()}")
                                        
                                        error_signatures[error_sig] = {
                                            'count': 1,
                                            'line_nums': [i],
                                            'severity': 'CRITICAL' if is_critical else 'ERROR',
                                            'message': line.strip(),
                                            'context_before': context_before,
                                            'context_after': context_after
                                        }
                                    break
                        
                        # Limit the number of unique errors reported per file
                        if len(error_signatures) >= max_errors_per_file:
                            break
                    
                    # Process the error signatures into error entries
                    for error_sig, info in error_signatures.items():
                        if info['count'] > 1:
                            # For duplicate errors, just show the first occurrence with context
                            error_entry = {
                                'severity': info['severity'],
                                'message': f"[{info['severity']}] Line {info['line_nums'][0]}: {info['message']} (repeated {info['count']} times at lines {', '.join(map(str, info['line_nums']))})",
                                'context_before': info['context_before'],
                                'context_after': info['context_after']
                            }
                        else:
                            # For single occurrences, show the full context
                            error_entry = {
                                'severity': info['severity'],
                                'message': f"[{info['severity']}] Line {info['line_nums'][0]}: {info['message']}",
                                'context_before': info['context_before'],
                                'context_after': info['context_after']
                            }
                        
                        error_entries.append(error_entry)
                    
                    if error_entries:
                        error_logs[log_path] = error_entries
                    
                except Exception as e:
                    logger.warning(f"Could not read log file {log_path}: {e}")
    
    # Special case for workflow summary files when no log files were found
    if not found_log_files or not error_logs:
        # Check specifically for summary files if no log files were found
        for root, _, files in os.walk(output_dir):
            for file in files:
                if "_summary.txt" in file or file.endswith('_summary.txt'):
                    summary_path = os.path.join(root, file)
                    try:
                        with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
                            summary_content = f.readlines()
                        
                        error_entries = []
                        error_found = False
                        workflow_status = None
                        
                        for i, line in enumerate(summary_content, 1):
                            # Check for workflow status
                            if "WORKFLOW STATUS:" in line:
                                workflow_status = line.strip()
                                
                            # Check for error information section and error messages
                            if any(keyword.lower() in line.lower() for keyword in 
                                  ["Error Information", "Error:", "failed", "division by zero"]):
                                error_found = True
                                
                                # Process surrounding context
                                context_before = []
                                context_after = []
                                
                                # Get lines before error
                                start_idx = max(0, i - context_lines - 1)
                                for j in range(start_idx, i - 1):
                                    context_before.append(f"Line {j+1}: {summary_content[j].strip()}")
                                
                                # Get lines after error
                                end_idx = min(len(summary_content), i + context_lines)
                                for j in range(i, end_idx):
                                    context_after.append(f"Line {j+1}: {summary_content[j].strip()}")
                                
                                error_entry = {
                                    'severity': 'CRITICAL',
                                    'message': f"[CRITICAL] Line {i}: {line.strip()}",
                                    'context_before': context_before,
                                    'context_after': context_after
                                }
                                error_entries.append(error_entry)
                        
                        # If we found an error status but no specific error message
                        if workflow_status and "ERROR" in workflow_status and not error_entries:
                            error_entry = {
                                'severity': 'CRITICAL',
                                'message': f"[CRITICAL] Workflow failed: {workflow_status}",
                                'context_before': [],
                                'context_after': []
                            }
                            error_entries.append(error_entry)
                        
                        if error_entries:
                            error_logs[summary_path] = error_entries
                            
                    except Exception as e:
                        logger.warning(f"Could not read summary file {summary_path}: {e}")
    
    return error_logs

def print_error_report(error_logs, output_file=None):
    """
    Print a report of errors found in log files.
    
    Args:
        error_logs: Dictionary with log file paths as keys and lists of error messages as values
        output_file: If provided, write the report to this file
    """
    report = []
    report.append("=" * 80)
    report.append("LOG ERROR REPORT")
    report.append("=" * 80)
    
    if not error_logs:
        report.append("\nNo actual errors found in log files.")
        report.append("\nThe system is functioning correctly!")
    else:
        # Count errors by severity
        critical_count = 0
        error_count = 0
        total_errors = 0
        files_with_errors = len(error_logs)
        
        for log_file, errors in error_logs.items():
            for error in errors:
                total_errors += 1
                if error['severity'] == 'CRITICAL':
                    critical_count += 1
                else:
                    error_count += 1
        
        # Create error summary
        report.append("\nERROR SUMMARY:")
        report.append(f"Total error count: {total_errors} in {files_with_errors} files")
        report.append(f"Critical errors: {critical_count}")
        report.append(f"Regular errors: {error_count}\n")
        
        # List files with errors
        for log_file, errors in error_logs.items():
            # Get relative path from the base_dir if possible
            try:
                rel_path = os.path.relpath(log_file, str(path_manager.base_dir))
                display_path = rel_path
            except:
                display_path = log_file
                
            report.append(f"\nFile: {display_path}")
            report.append("-" * 80)
            
            # List errors with context
            for error in errors:
                # Add context before error
                if error['context_before']:
                    report.append("\nContext before error:")
                    for ctx_line in error['context_before']:
                        report.append(f"  {ctx_line}")
                
                # Add the error message (highlighted)
                report.append(f"\n>> {error['message']}")
                
                # Add context after error
                if error['context_after']:
                    report.append("\nContext after error:")
                    for ctx_line in error['context_after']:
                        report.append(f"  {ctx_line}")
                
                report.append("-" * 40)  # Separator between errors
            
            report.append("-" * 80)  # End of file section
    
    report_text = "\n".join(report)
    
    # Print to console
    print(report_text)
    
    # Write to file if requested
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Error report saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to write error report to {output_file}: {e}")
    
    return report_text 