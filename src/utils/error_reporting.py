#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error reporting module for all workflows.
Creates consolidated error reports across workflow runs.
"""
import os
import sys
import json
import re
import logging
import datetime
from typing import Dict, List, Any, Optional, Union

# Add the parent directory to the path
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
logger = logging_system.get_logger('error_reporting')

def generate_error_summary(output_dir: str, workflow_type: str, strategy_name: str, 
                          start_date: str, end_date: str, tickers: List[str], 
                          error_msg: str = None) -> str:
    """
    Generate a comprehensive error summary for a workflow run.
    
    Args:
        output_dir: Directory where workflow results are stored
        workflow_type: Type of workflow (simple, monte_carlo, etc.)
        strategy_name: Name of the strategy
        start_date: Start date for the backtest
        end_date: End date for the backtest
        tickers: List of ticker symbols
        error_msg: Optional specific error message to include
        
    Returns:
        Path to the generated error summary file
    """
    logger.info(f"Generating error summary for {workflow_type} workflow with {strategy_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to save the error summary
    summary_file = os.path.join(output_dir, "error_summary.txt")
    
    # Collect errors from logs
    from workflows.workflow_utils import check_logs_for_errors, print_error_report
    error_logs = check_logs_for_errors(output_dir)
    
    # Load config and parameters if available
    parameters = {}
    config_file = None
    
    # Look for a config.json file
    for filename in os.listdir(output_dir):
        if filename.endswith('.json') and 'config' in filename.lower():
            config_file = os.path.join(output_dir, filename)
            break
    
    # If we found a config file, load it
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            # Extract parameters for the strategy
            if 'strategies' in config and strategy_name in config['strategies']:
                parameters = config['strategies'][strategy_name].get('parameters', {})
            elif 'parameters' in config:
                parameters = config['parameters']
        except Exception as e:
            logger.warning(f"Error loading config file: {e}")
    
    # Write the summary file
    with open(summary_file, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write(f"{workflow_type.upper()} ERROR SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic test information
        f.write(f"Strategy: {strategy_name}\n")
        f.write(f"Period: {start_date} to {end_date}\n")
        if isinstance(tickers, list):
            f.write(f"Tickers: {', '.join(tickers)}\n")
        else:
            f.write(f"Tickers: {tickers}\n")
        
        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Error Report Generated: {timestamp}\n\n")
        
        # Parameters section
        f.write("-" * 80 + "\n")
        f.write("Parameters:\n")
        f.write("-" * 80 + "\n")
        if parameters:
            for param, value in parameters.items():
                f.write(f"{param}: {value}\n")
        else:
            f.write("No parameters found.\n")
        f.write("\n")
        
        # Error information
        f.write("-" * 80 + "\n")
        f.write("Error Information:\n")
        f.write("-" * 80 + "\n")
        
        # Include specific error message if provided
        if error_msg:
            f.write(f"Primary Error: {error_msg}\n\n")
        
        # Error summary statistics
        if not error_logs:
            f.write("No additional errors found in log files.\n")
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
            
            # Add error statistics
            f.write(f"Total errors found: {total_errors} in {files_with_errors} files\n")
            f.write(f"Critical errors: {critical_count}\n")
            f.write(f"Regular errors: {error_count}\n\n")
            
            # List errors by file
            for log_file, errors in error_logs.items():
                # Get relative path from output directory
                try:
                    rel_path = os.path.relpath(log_file, output_dir)
                    display_path = rel_path
                except:
                    display_path = os.path.basename(log_file)
                    
                f.write(f"File: {display_path}\n")
                f.write("-" * 40 + "\n")
                
                # List errors (simplified for summary)
                for i, error in enumerate(errors, 1):
                    f.write(f"{i}. {error['message']}\n")
                
                f.write("\n")
        
        # End of summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("WORKFLOW STATUS: ERROR\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Error summary saved to: {summary_file}")
    
    # Generate a more detailed error report with context
    error_report_path = os.path.join(output_dir, "error_report.txt")
    print_error_report(error_logs, error_report_path)
    
    return summary_file

def collect_workflow_errors(output_root: str = None, days: int = 7) -> Dict[str, Any]:
    """
    Collect errors from all workflow runs within a specified timeframe.
    
    Args:
        output_root: Root directory containing workflow outputs
        days: Only include errors from the last N days
        
    Returns:
        Dictionary with workflow statistics and error summaries
    """
    if output_root is None:
        output_root = os.path.join(project_root, "output")
    
    # Calculate cutoff date
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
    
    # Results dictionary
    results = {
        "total_workflow_runs": 0,
        "successful_runs": 0,
        "failed_runs": 0,
        "workflows_with_errors": [],
        "error_counts": {
            "critical": 0,
            "error": 0,
            "warning": 0
        },
        "errors_by_workflow": {},
        "errors_by_strategy": {},
        "most_common_errors": []
    }
    
    # Dictionary to track error occurrences
    error_occurrences = {}
    
    # Walk through output directories
    for root, dirs, files in os.walk(output_root):
        # Check if this is a workflow output directory
        if any(f.endswith('_summary.txt') for f in files) or any(f.endswith('.log') for f in files):
            # Get directory modification time
            try:
                mod_time = os.path.getmtime(root)
                dir_date = datetime.datetime.fromtimestamp(mod_time)
                if dir_date < cutoff_date:
                    continue  # Skip older directories
            except Exception:
                pass  # If we can't determine date, include it
            
            # Try to determine workflow type and strategy from directory name
            dir_name = os.path.basename(root)
            strategy_name = None
            workflow_type = None
            
            # Parse directory name (format: Strategy_workflow_timestamp_id)
            parts = dir_name.split('_')
            if len(parts) >= 2:
                strategy_name = parts[0]
                for workflow in ['simple', 'monte_carlo', 'optimization', 'walkforward', 'complete']:
                    if workflow in dir_name.lower():
                        workflow_type = workflow
                        break
            
            # Count this as a workflow run
            results["total_workflow_runs"] += 1
            
            # Check for error files or logs
            has_error = False
            error_files = []
            
            for filename in files:
                if filename == 'error_summary.txt' or filename == 'error_report.txt':
                    has_error = True
                    error_files.append(os.path.join(root, filename))
                elif '_summary.txt' in filename:
                    # Check if the summary file indicates an error
                    try:
                        with open(os.path.join(root, filename), 'r') as f:
                            content = f.read()
                            if "WORKFLOW STATUS: ERROR" in content or "Error:" in content:
                                has_error = True
                                error_files.append(os.path.join(root, filename))
                    except Exception:
                        pass
            
            # Update counts
            if has_error:
                results["failed_runs"] += 1
                
                # Create workflow entry
                workflow_entry = {
                    "path": root,
                    "strategy": strategy_name,
                    "workflow_type": workflow_type,
                    "timestamp": dir_date.strftime("%Y-%m-%d %H:%M:%S") if 'dir_date' in locals() else "Unknown",
                    "error_files": error_files
                }
                
                # Extract key errors
                key_errors = []
                for error_file in error_files:
                    try:
                        with open(error_file, 'r') as f:
                            content = f.read()
                            
                            # Extract main errors using regex
                            error_matches = re.findall(r'Error:.*?$(.*?$)?', content, re.MULTILINE)
                            key_errors.extend([e.strip() for e in error_matches if e.strip()])
                            
                            # Extract critical errors
                            critical_matches = re.findall(r'\[CRITICAL\].*?$', content, re.MULTILINE)
                            key_errors.extend([e.strip() for e in critical_matches if e.strip()])
                            
                            # Count errors by type
                            results["error_counts"]["critical"] += content.count("[CRITICAL]")
                            results["error_counts"]["error"] += content.count("[ERROR]")
                            results["error_counts"]["warning"] += content.count("[WARNING]")
                            
                            # Track error occurrences for most common errors
                            for error_line in content.split('\n'):
                                if "[CRITICAL]" in error_line or "[ERROR]" in error_line:
                                    # Normalize error message (remove timestamps, line numbers, variables)
                                    norm_error = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', error_line)
                                    norm_error = re.sub(r'line \d+', 'line X', norm_error)
                                    norm_error = re.sub(r':\d+:', ':X:', norm_error)
                                    
                                    if norm_error in error_occurrences:
                                        error_occurrences[norm_error] += 1
                                    else:
                                        error_occurrences[norm_error] = 1
                    except Exception as e:
                        key_errors.append(f"Error reading file: {e}")
                
                workflow_entry["key_errors"] = key_errors
                results["workflows_with_errors"].append(workflow_entry)
                
                # Update errors by workflow and strategy
                if workflow_type:
                    if workflow_type not in results["errors_by_workflow"]:
                        results["errors_by_workflow"][workflow_type] = 1
                    else:
                        results["errors_by_workflow"][workflow_type] += 1
                
                if strategy_name:
                    if strategy_name not in results["errors_by_strategy"]:
                        results["errors_by_strategy"][strategy_name] = 1
                    else:
                        results["errors_by_strategy"][strategy_name] += 1
            else:
                results["successful_runs"] += 1
    
    # Calculate most common errors
    most_common = sorted(error_occurrences.items(), key=lambda x: x[1], reverse=True)
    results["most_common_errors"] = [{"error": e, "count": c} for e, c in most_common[:10]]
    
    # Calculate success rate
    if results["total_workflow_runs"] > 0:
        results["success_rate"] = (results["successful_runs"] / results["total_workflow_runs"]) * 100
    else:
        results["success_rate"] = 0
    
    return results

def generate_consolidated_error_report(output_dir: str = None, days: int = 7) -> str:
    """
    Generate a consolidated error report for all workflow runs.
    
    Args:
        output_dir: Directory to save the report
        days: Only include errors from the last N days
        
    Returns:
        Path to the generated report file
    """
    # If no specific output directory is provided, use the current directory
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Path for the report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"consolidated_error_report_{timestamp}.txt")
    
    # Collect errors
    error_data = collect_workflow_errors(days=days)
    
    # Write the report
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CONSOLIDATED ERROR REPORT - Last {days} days\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total workflow runs: {error_data['total_workflow_runs']}\n")
        f.write(f"Successful runs: {error_data['successful_runs']}\n")
        f.write(f"Failed runs: {error_data['failed_runs']}\n")
        
        if error_data['total_workflow_runs'] > 0:
            success_rate = (error_data['successful_runs'] / error_data['total_workflow_runs']) * 100
            f.write(f"Success rate: {success_rate:.2f}%\n\n")
        
        # Error counts
        f.write("ERROR COUNTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Critical errors: {error_data['error_counts']['critical']}\n")
        f.write(f"Regular errors: {error_data['error_counts']['error']}\n")
        f.write(f"Warnings: {error_data['error_counts']['warning']}\n\n")
        
        # Errors by workflow type
        f.write("ERRORS BY WORKFLOW TYPE\n")
        f.write("-" * 80 + "\n")
        for workflow, count in error_data['errors_by_workflow'].items():
            f.write(f"{workflow}: {count}\n")
        f.write("\n")
        
        # Errors by strategy
        f.write("ERRORS BY STRATEGY\n")
        f.write("-" * 80 + "\n")
        for strategy, count in error_data['errors_by_strategy'].items():
            f.write(f"{strategy}: {count}\n")
        f.write("\n")
        
        # Most common errors
        f.write("MOST COMMON ERRORS\n")
        f.write("-" * 80 + "\n")
        for i, error_info in enumerate(error_data['most_common_errors'], 1):
            f.write(f"{i}. [{error_info['count']} occurrences] {error_info['error']}\n")
        f.write("\n")
        
        # Individual workflow errors
        f.write("INDIVIDUAL WORKFLOW ERRORS\n")
        f.write("-" * 80 + "\n")
        for i, workflow in enumerate(error_data['workflows_with_errors'], 1):
            f.write(f"Workflow #{i}: {workflow['strategy']} - {workflow['workflow_type']}\n")
            f.write(f"Path: {workflow['path']}\n")
            f.write(f"Timestamp: {workflow['timestamp']}\n")
            
            if workflow['key_errors']:
                f.write("Key errors:\n")
                for j, error in enumerate(workflow['key_errors'], 1):
                    f.write(f"  {j}. {error}\n")
            else:
                f.write("No key errors extracted.\n")
            
            f.write("-" * 40 + "\n\n")
        
        # End of report
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Consolidated error report saved to: {report_path}")
    return report_path

def main():
    """Command-line interface for error reporting."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate error reports for workflows")
    parser.add_argument("--days", type=int, default=7, help="Number of days to include in the report")
    parser.add_argument("--output-dir", type=str, help="Directory to save the report")
    parser.add_argument("--workflow-dir", type=str, help="Specific workflow directory to analyze")
    args = parser.parse_args()
    
    if args.workflow_dir:
        # Generate an error report for a specific workflow directory
        if not os.path.exists(args.workflow_dir):
            print(f"Error: Directory {args.workflow_dir} does not exist")
            return
            
        # Try to determine workflow type and strategy
        dir_name = os.path.basename(args.workflow_dir)
        parts = dir_name.split('_')
        
        strategy_name = parts[0] if len(parts) > 0 else "Unknown"
        
        workflow_type = "unknown"
        for wf in ['simple', 'monte_carlo', 'optimization', 'walkforward', 'complete']:
            if wf in dir_name.lower():
                workflow_type = wf
                break
        
        # Generate summary for the specific directory
        summary_file = generate_error_summary(
            args.workflow_dir, 
            workflow_type,
            strategy_name,
            "Unknown", "Unknown",
            ["Unknown"],
            "Manual error report generation"
        )
        
        print(f"Error summary generated: {summary_file}")
    else:
        # Generate a consolidated report
        report_path = generate_consolidated_error_report(
            args.output_dir,
            args.days
        )
        
        print(f"Consolidated error report generated: {report_path}")

if __name__ == "__main__":
    main()