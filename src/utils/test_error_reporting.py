#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the error reporting system.
This file creates simulated errors and tests the error reporting functionality.
"""
import os
import sys
import json
import argparse
import datetime
import random
import logging
from typing import List, Dict, Any, Optional

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
project_root = os.path.dirname(src_dir)  # Go up to project root
if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the error reporting module
from utils.error_reporting import (
    StageError, 
    StageErrorReport,
    extract_stage_errors,
    create_stage_error_report,
    detect_workflow_stage,
    WORKFLOW_STAGES
)

# Import the logging system
from engine.logging_system import LoggingSystem

# Initialize the logging system
logging_system = LoggingSystem(
    console_output=True,
    file_output=True,
    default_level='INFO',
    async_logging=False
)
logger = logging_system.get_logger('test_error_reporting')

def create_test_log_file(output_dir: str, workflow_type: str, 
                        error_count: int = 5,
                        critical_count: int = 2,
                        warning_count: int = 3) -> str:
    """
    Create a test log file with simulated errors.
    
    Args:
        output_dir: Directory to save the log file
        workflow_type: Type of workflow to simulate
        error_count: Number of regular errors to generate
        critical_count: Number of critical errors to generate
        warning_count: Number of warnings to generate
        
    Returns:
        Path to the created log file
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{workflow_type}_test.log")
    
    # Get stages for this workflow type
    stages = WORKFLOW_STAGES.get(workflow_type, [])
    if not stages:
        stages = ["initialization", "data_loading", "strategy_setup", "unknown"]
    
    # Common error messages
    error_messages = [
        "Failed to load data file",
        "Invalid parameter: {param}",
        "Unable to compute strategy metric",
        "Division by zero in performance calculation",
        "File not found: {file}",
        "Strategy returned invalid signal",
        "Cannot process empty dataframe",
        "Invalid date range",
        "Missing required parameter: {param}",
        "Index out of bounds"
    ]
    
    # Error contexts
    error_contexts = [
        ["Loading historical data for {ticker}", "Processing price data", "Calculating indicators"],
        ["Setting up strategy parameters", "Initializing {strategy} strategy", "Configuring backtester"],
        ["Running backtest iteration", "Processing day {date}", "Generating signals for {ticker}"],
        ["Calculating performance metrics", "Computing {metric}", "Finalizing results"]
    ]
    
    # Create random datetime objects for logging
    start_date = datetime.datetime.now() - datetime.timedelta(hours=2)
    
    # Open the log file for writing
    with open(log_file, 'w') as f:
        # Write normal log entries interspersed with errors
        
        # Initialization phase
        current_time = start_date
        f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [INFO] Starting {workflow_type} workflow\n")
        f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [INFO] Initializing workflow components\n")
        
        current_time += datetime.timedelta(seconds=5)
        f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [INFO] Loading configuration from config file\n")
        
        # Data loading phase
        current_time += datetime.timedelta(seconds=8)
        f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [INFO] Loading data for backtest\n")
        f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [INFO] Reading historical data for AAPL\n")
        
        # Generate warnings
        for i in range(warning_count):
            current_time += datetime.timedelta(seconds=random.randint(1, 5))
            stage = random.choice(stages)
            stage_msg = f"in {stage} stage"
            
            context = random.choice(error_contexts)
            context_msg = random.choice(context).format(
                ticker=random.choice(["AAPL", "MSFT", "GOOG"]),
                strategy=random.choice(["MACrossover", "RSI", "MACD"]),
                date=datetime.date.today().strftime("%Y-%m-%d"),
                metric=random.choice(["Sharpe", "Sortino", "Drawdown"])
            )
            
            warning_msg = f"Possible issue {stage_msg}: {context_msg}"
            f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [WARNING] {warning_msg}\n")
        
        # Strategy setup phase
        current_time += datetime.timedelta(seconds=7)
        f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [INFO] Setting up strategy for backtest\n")
        f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [INFO] Initializing strategy parameters\n")
        
        # Generate regular errors
        for i in range(error_count):
            current_time += datetime.timedelta(seconds=random.randint(1, 5))
            stage = random.choice(stages)
            stage_msg = f"in {stage} stage"
            
            error_msg = random.choice(error_messages).format(
                param=random.choice(["window_size", "threshold", "stop_loss"]),
                file=f"/path/to/data/{random.randint(1, 100)}.csv",
                ticker=random.choice(["AAPL", "MSFT", "GOOG"])
            )
            
            f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [ERROR] Error {stage_msg}: {error_msg}\n")
        
        # Backtest execution phase
        current_time += datetime.timedelta(seconds=15)
        f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [INFO] Executing backtest\n")
        f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [INFO] Running simulation for 500 days\n")
        
        # Generate critical errors
        for i in range(critical_count):
            current_time += datetime.timedelta(seconds=random.randint(1, 5))
            stage = random.choice(stages)
            stage_msg = f"in {stage} stage"
            
            error_msg = f"Critical failure {stage_msg}: "
            error_type = random.choice([
                "Exception", 
                "RuntimeError", 
                "ValueError", 
                "KeyError", 
                "IndexError"
            ])
            
            context = random.choice([
                "Failed to execute strategy",
                "Cannot process market data",
                "Invalid parameter configuration",
                "Exception during performance calculation"
            ])
            
            traceback = f"""Traceback (most recent call last):
  File "/path/to/source.py", line {random.randint(10, 500)}, in {random.choice(['run_backtest', 'calculate_metrics', 'process_data'])}
    {random.choice(['result = data[invalid_index]', 'value = 1/zero_value', 'obj.missing_method()'])}
{error_type}: {context}"""
            
            f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [CRITICAL] {error_msg}{error_type}: {context}\n")
            f.write(f"{traceback}\n\n")
        
        # Performance evaluation phase
        current_time += datetime.timedelta(seconds=12)
        f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [INFO] Evaluating performance\n")
        f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [INFO] Calculating metrics\n")
        
        # End of workflow
        current_time += datetime.timedelta(seconds=5)
        f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [INFO] Workflow completed with errors\n")
    
    logger.info(f"Created test log file with {error_count} errors, {critical_count} critical errors, and {warning_count} warnings: {log_file}")
    return log_file

def create_test_summary_file(output_dir: str, workflow_type: str, strategy_name: str) -> str:
    """
    Create a test summary file for a workflow.
    
    Args:
        output_dir: Directory to save the summary file
        workflow_type: Type of workflow
        strategy_name: Name of the strategy
        
    Returns:
        Path to the created summary file
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, f"{workflow_type}_summary.txt")
    
    with open(summary_file, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"{workflow_type.upper()} WORKFLOW SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic information
        f.write(f"Strategy: {strategy_name}\n")
        f.write(f"Date Range: 2022-01-01 to 2022-12-31\n")
        f.write(f"Tickers: AAPL, MSFT, GOOG\n")
        
        # Performance metrics
        f.write("\nPerformance Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Return: {random.uniform(-10, 50):.2f}%\n")
        f.write(f"Sharpe Ratio: {random.uniform(-1, 3):.2f}\n")
        f.write(f"Max Drawdown: {random.uniform(5, 25):.2f}%\n")
        
        # Add an error section
        f.write("\nError Information:\n")
        f.write("-" * 40 + "\n")
        f.write("The workflow encountered errors during execution.\n")
        f.write(f"Error in {random.choice(WORKFLOW_STAGES.get(workflow_type, ['unknown']))}: Failed to calculate some metrics.\n")
        
        # Workflow status
        f.write("\n" + "=" * 80 + "\n")
        f.write("WORKFLOW STATUS: ERROR\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Created test summary file: {summary_file}")
    return summary_file

def create_test_workflow_output(output_base: str, workflow_type: str, strategy_name: str) -> str:
    """
    Create a complete test workflow output directory with logs and summary.
    
    Args:
        output_base: Base directory for outputs
        workflow_type: Type of workflow to simulate
        strategy_name: Name of the strategy
        
    Returns:
        Path to the created output directory
    """
    # Create a timestamped directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, f"{strategy_name}_{workflow_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a logs subdirectory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log files
    create_test_log_file(logs_dir, workflow_type)
    
    # Create summary file
    create_test_summary_file(output_dir, workflow_type, strategy_name)
    
    logger.info(f"Created test workflow output at: {output_dir}")
    return output_dir

def test_stage_error_extraction(output_dir: str, workflow_type: str, strategy_name: str) -> None:
    """
    Test the stage error extraction functionality.
    
    Args:
        output_dir: Directory containing the workflow output
        workflow_type: Type of workflow
        strategy_name: Name of the strategy
    """
    logger.info(f"Testing stage error extraction for {workflow_type} workflow")
    
    # Extract errors
    report = extract_stage_errors(output_dir, workflow_type, strategy_name)
    
    # Print error statistics
    error_count = len(report.get_all_errors())
    logger.info(f"Extracted {error_count} errors from {output_dir}")
    
    # Print errors by stage
    logger.info("Errors by stage:")
    for stage, count in report.count_errors_by_stage().items():
        logger.info(f"  {stage}: {count}")
    
    # Print errors by severity
    logger.info("Errors by severity:")
    for severity, count in report.count_errors_by_severity().items():
        logger.info(f"  {severity}: {count}")
    
    # Generate report
    report_path = report.save_text_report()
    logger.info(f"Stage error report saved to: {report_path}")

def test_create_stage_error_report(output_dir: str, workflow_type: str, strategy_name: str) -> None:
    """
    Test the create_stage_error_report function.
    
    Args:
        output_dir: Directory containing the workflow output
        workflow_type: Type of workflow
        strategy_name: Name of the strategy
    """
    logger.info(f"Testing create_stage_error_report for {workflow_type} workflow")
    
    # Create report
    report_path = create_stage_error_report(output_dir, workflow_type, strategy_name)
    
    logger.info(f"Stage error report created at: {report_path}")
    
    # Verify file exists
    if os.path.exists(report_path):
        logger.info("Report file exists ✅")
    else:
        logger.error("Report file does not exist ❌")

def test_stage_error_class() -> None:
    """Test the StageError class."""
    logger.info("Testing StageError class")
    
    # Create a test error
    error = StageError(
        message="Test error message",
        stage="initialization",
        severity="ERROR",
        file_path="/path/to/file.py",
        line_number=42,
        context_before=["Line 1", "Line 2"],
        context_after=["Line 3", "Line 4"]
    )
    
    # Convert to dict and back
    error_dict = error.to_dict()
    restored_error = StageError.from_dict(error_dict)
    
    # Compare values
    logger.info(f"Original error: {error}")
    logger.info(f"Restored error: {restored_error}")
    
    if error.message == restored_error.message and error.stage == restored_error.stage:
        logger.info("Serialization test passed ✅")
    else:
        logger.error("Serialization test failed ❌")

def test_detect_workflow_stage() -> None:
    """Test the detect_workflow_stage function."""
    logger.info("Testing detect_workflow_stage function")
    
    # Test cases
    test_cases = [
        ("2023-01-01 12:00:00 [INFO] Initializing workflow components", "simple", "initialization"),
        ("2023-01-01 12:01:00 [INFO] Loading data for backtest", "simple", "data_loading"),
        ("2023-01-01 12:02:00 [INFO] Setting up strategy parameters", "simple", "strategy_setup"),
        ("2023-01-01 12:03:00 [INFO] Running backtest for strategy", "simple", "backtest_execution"),
        ("2023-01-01 12:04:00 [INFO] Calculating performance metrics", "simple", "performance_evaluation"),
        ("2023-01-01 12:05:00 [ERROR] Error in optimization: Parameter grid too large", "optimization", "parameter_grid_generation"),
        ("2023-01-01 12:06:00 [INFO] Generating Monte Carlo simulations", "monte_carlo", "simulation_generation"),
        ("2023-01-01 12:07:00 [INFO] Creating walkforward windows", "walkforward", "window_generation"),
        ("2023-01-01 12:08:00 [INFO] Some unrelated message", "simple", None)
    ]
    
    # Run tests
    passed = 0
    for log_line, workflow_type, expected_stage in test_cases:
        detected_stage = detect_workflow_stage(log_line, workflow_type)
        
        logger.info(f"Log: '{log_line[:30]}...'")
        logger.info(f"Expected: {expected_stage}, Detected: {detected_stage}")
        
        if detected_stage == expected_stage:
            logger.info("PASSED ✅")
            passed += 1
        else:
            logger.info("FAILED ❌")
    
    logger.info(f"Passed {passed}/{len(test_cases)} tests")

def run_all_tests(output_base: str) -> None:
    """Run all tests for the error reporting system."""
    logger.info("Running all error reporting tests")
    
    # Create a fresh test directory
    test_output = os.path.join(output_base, "test_error_reporting")
    os.makedirs(test_output, exist_ok=True)
    
    # Create test outputs for each workflow type
    workflow_dirs = {}
    
    for workflow_type in ["simple", "optimization", "monte_carlo", "walkforward", "complete"]:
        workflow_dirs[workflow_type] = create_test_workflow_output(
            test_output, workflow_type, "TestStrategy"
        )
    
    # Test stage error extraction
    for workflow_type, output_dir in workflow_dirs.items():
        test_stage_error_extraction(output_dir, workflow_type, "TestStrategy")
    
    # Test create_stage_error_report function
    for workflow_type, output_dir in workflow_dirs.items():
        test_create_stage_error_report(output_dir, workflow_type, "TestStrategy")
    
    # Test StageError class
    test_stage_error_class()
    
    # Test detect_workflow_stage function
    test_detect_workflow_stage()
    
    logger.info("All tests completed")

def main():
    """Main function for the test script."""
    parser = argparse.ArgumentParser(description="Test the error reporting system")
    parser.add_argument("--output-dir", type=str, default=os.path.join(project_root, "output"),
                       help="Directory for test outputs")
    parser.add_argument("--workflow-type", type=str, choices=WORKFLOW_STAGES.keys(),
                       default="simple", help="Workflow type to test")
    parser.add_argument("--test-all", action="store_true", 
                       help="Run all tests for all workflow types")
    
    args = parser.parse_args()
    
    if args.test_all:
        run_all_tests(args.output_dir)
    else:
        # Create a test output for the specified workflow type
        output_dir = create_test_workflow_output(
            args.output_dir, args.workflow_type, "TestStrategy"
        )
        
        # Test stage error extraction
        test_stage_error_extraction(output_dir, args.workflow_type, "TestStrategy")
        
        # Test create_stage_error_report function
        test_create_stage_error_report(output_dir, args.workflow_type, "TestStrategy")

if __name__ == "__main__":
    main()