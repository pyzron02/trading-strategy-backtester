#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface for the unified workflow.
"""
import os
import sys
import json
import argparse
from datetime import datetime
import uuid
import numpy as np
import pandas as pd

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
project_root = os.path.dirname(src_dir)  # Go up to project root
if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the utilities
from workflows.workflow_utils import logger, logging_system

# Import the unified workflow
from workflows.unified_workflow import run_unified_workflow, is_parameter_grid

def generate_monte_carlo_summary(result, output_dir):
    """
    This function is deprecated as summary generation is now handled in monte_carlo_workflow.py.
    This stub is maintained for backwards compatibility.
    
    Args:
        result (dict): The workflow result dictionary
        output_dir (str): The output directory path
    """
    logger.info(f"Monte Carlo summary generation is handled directly in monte_carlo_workflow.py")
    logger.info(f"Skipping duplicate summary generation for output directory: {output_dir}")
    
    # Check if summary file already exists to inform user
    summary_file = os.path.join(output_dir, "monte_carlo_summary.txt")
    if os.path.exists(summary_file):
        logger.info(f"Monte Carlo summary file already exists at: {summary_file}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Unified workflow for trading strategy backtesting")

    # Config file option
    parser.add_argument("--config", type=str, help="Path to JSON config file for the workflow")
    
    # Common arguments
    parser.add_argument("--workflow", type=str, choices=[
        "simple", "optimization", "monte_carlo", "walkforward", "complete"
    ], help="Type of workflow to run")
    parser.add_argument("--strategy", type=str, help="Name of trading strategy")
    parser.add_argument("--tickers", type=str, nargs='+', help="Ticker symbols (space or comma separated)")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, default="2025-01-01", help="End date in YYYY-MM-DD format")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--param-file", type=str, help="Parameter file for the strategy")
    parser.add_argument("--plot", action="store_true", help="Generate and save plots of backtest results (simple workflow only)")
    
    # Optimization parameters
    optimization_group = parser.add_argument_group("Optimization parameters")
    optimization_group.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials")
    optimization_group.add_argument("--optimization-metric", type=str, default="sharpe_ratio", 
                        help="Metric to optimize for (e.g., sharpe_ratio, total_return)")
    optimization_group.add_argument("--max-combinations", type=int, default=None, 
                        help="Maximum number of parameter combinations to test")
    
    # Monte Carlo parameters
    monte_carlo_group = parser.add_argument_group("Monte Carlo parameters")
    monte_carlo_group.add_argument("--n-simulations", type=int, default=100, 
                        help="Number of Monte Carlo simulations")
    monte_carlo_group.add_argument("--keep-permuted-data", action="store_true", 
                        help="Keep permuted data generated during Monte Carlo simulation")
    monte_carlo_group.add_argument("--enhanced-plots", action="store_true", 
                        help="Generate enhanced visualization dashboard for Monte Carlo simulations")
    
    # Walk-forward parameters
    walkforward_group = parser.add_argument_group("Walk-forward parameters")
    walkforward_group.add_argument("--window-size", type=int, default=252, 
                        help="Window size in trading days")
    walkforward_group.add_argument("--step-size", type=int, default=63, 
                        help="Step size in trading days")
    
    # Data parameters
    data_group = parser.add_argument_group("Data parameters")
    data_group.add_argument("--stock-csv", type=str, help="Path to CSV file with stock data")
    data_group.add_argument("--data-dir", type=str, default="input", help="Directory containing input data")
    
    # Capital parameters
    capital_group = parser.add_argument_group("Capital parameters")
    capital_group.add_argument("--initial-capital", type=float, default=100000.0, help="Initial capital")
    capital_group.add_argument("--commission", type=float, default=0.001, help="Commission rate")
    
    return parser.parse_args()

def process_tickers(ticker_args):
    """
    Process ticker arguments which could be space or comma separated.
    
    Args:
        ticker_args: List of ticker strings potentially containing commas
        
    Returns:
        List of individual tickers
    """
    if not ticker_args:
        return ["SPY"]  # Default ticker
    
    # Process each argument which could be a single ticker or comma-separated list
    tickers = []
    for arg in ticker_args:
        if ',' in arg:
            # Handle comma-separated tickers
            tickers.extend([t.strip() for t in arg.split(',') if t.strip()])
        else:
            # Handle single ticker
            if arg.strip():
                tickers.append(arg.strip())
    
    if not tickers:
        logger.warning("No valid tickers specified. Using default ticker 'SPY'")
        return ["SPY"]
    
    return tickers

def check_workflow_param_file_compatibility(workflow_type, param_file):
    """
    Check if the parameter file is compatible with the selected workflow.
    
    Args:
        workflow_type: Type of workflow to run
        param_file: Path to parameter file
        
    Returns:
        None, but logs a warning if there's a potential mismatch
    """
    if not param_file or not os.path.exists(param_file):
        return
        
    # Check if file contains parameter grids
    is_grid = is_parameter_grid(param_file)
    
    # Log appropriate warnings based on workflow type and file content
    if workflow_type == "simple" and is_grid:
        logger.warning(
            f"NOTE: Parameter file '{param_file}' contains grid values, which are typically used with optimization workflows."
            f" When used with simple workflow, only the first value from each parameter will be used."
            f" Consider using --workflow optimization if you want to test all parameter combinations."
        )
    elif workflow_type == "optimization" and not is_grid:
        logger.warning(
            f"NOTE: Parameter file '{param_file}' does not contain grid values. "
            f"For optimization, parameters should be lists of values to test. "
            f"This optimization will only test a single parameter combination."
        )

def run_cli():
    """Main function for the command-line interface."""
    args = parse_args()
    
    # Check if a config file is provided
    if args.config:
        # Import the config runner function here to avoid circular imports
        from workflows.unified_workflow import run_unified_workflow_from_config
        
        # Run with config file
        try:
            logger.info(f"Running with config file: {args.config}")
            result = run_unified_workflow_from_config(args.config)
            
            # Get workflow type
            workflow_type = result.get("workflow_type", "")
            
            # Print summary of results
            if result["status"] == "success":
                logger.info("\n" + "=" * 80)
                logger.info(f"Workflow completed successfully.")
                
                # Print summary of results from each strategy
                for strategy_name, strategy_result in result.get("results", {}).items():
                    status = strategy_result.get("status", "unknown")
                    output_dir = strategy_result.get("output_dir", "N/A")
                    logger.info(f"Strategy: {strategy_name} - Status: {status} - Output: {output_dir}")
            else:
                logger.error(f"\nError: {result.get('message', 'Unknown error')}")
            
            # No longer need to generate monte_carlo summary - handled in monte_carlo_workflow.py
            if workflow_type == "monte_carlo":
                for strategy_name, strategy_result in result.get("results", {}).items():
                    output_dir = strategy_result.get("output_dir", "N/A")
                    if os.path.exists(output_dir):
                        # Check for summary file existence only
                        summary_file = os.path.join(output_dir, "monte_carlo_summary.txt")
                        if os.path.exists(summary_file):
                            logger.info(f"Monte Carlo summary file exists at: {summary_file}")
                        else:
                            logger.warning(f"Monte Carlo summary file not found at: {summary_file}")
                    else:
                        logger.warning(f"Output directory doesn't exist: {output_dir}")
                
            # Exit with error if the workflow failed
            if result["status"] != "success":
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"\nError during workflow execution with config file: {str(e)}")
            if args.verbose:
                logger.exception("Full traceback:")
            sys.exit(1)
        
        return
    
    # If no config file is provided, ensure required args are present
    if not args.workflow:
        logger.error("Error: --workflow is required when not using a config file")
        sys.exit(1)
        
    if not args.strategy:
        logger.error("Error: --strategy is required when not using a config file")
        sys.exit(1)
    
    # Process tickers (handles both comma and space separated formats)
    tickers = process_tickers(args.tickers)
    
    # Set output directory if not specified
    output_dir = args.output_dir
    if output_dir is None:
        # Always create a unique output directory with timestamp and unique identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = str(uuid.uuid4())[:8]  # First 8 chars of a UUID for uniqueness
        output_dir = os.path.join(project_root, "output", f"{args.strategy}_{args.workflow}_{timestamp}_{run_id}")
        logger.info(f"Using output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set logging level if verbose
    if args.verbose:
        logging_system.set_level('DEBUG')
        logger.debug("Verbose logging enabled")
    
    # Check parameter file compatibility with selected workflow
    if args.param_file:
        check_workflow_param_file_compatibility(args.workflow, args.param_file)
    
    # Prepare parameters dictionary with common parameters
    workflow_params = {
        "strategy_name": args.strategy,
        "tickers": tickers,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "output_dir": output_dir,
        "verbose": args.verbose,
        "initial_capital": args.initial_capital,
        "commission": args.commission,
        "data_dir": args.data_dir,
        "param_file": args.param_file
    }
    
    # Add workflow-specific parameters based on workflow type
    if args.workflow == "simple":
        workflow_params.update({
            "plot": args.plot
        })
    elif args.workflow == "optimization":
        workflow_params.update({
            "n_trials": args.n_trials,
            "optimization_metric": args.optimization_metric
        })
        
        # Add max_combinations parameter if provided
        if args.max_combinations is not None:
            workflow_params["max_combinations"] = args.max_combinations
    elif args.workflow == "monte_carlo":
        workflow_params.update({
            "n_simulations": args.n_simulations,
            "keep_permuted_data": args.keep_permuted_data,
            "plot": args.plot,  # Monte Carlo also supports plotting
            "enhanced_plots": args.enhanced_plots
        })
    elif args.workflow == "walkforward":
        workflow_params.update({
            "window_size": args.window_size,
            "step_size": args.step_size,
            "n_trials": args.n_trials,
            "optimization_metric": args.optimization_metric,
            "plot": args.plot,  # Respect the user's plot setting
            "enhanced_plots": args.enhanced_plots if hasattr(args, 'enhanced_plots') else False  # Respect user's enhanced_plots setting
        })
    elif args.workflow == "complete":
        # For complete workflow, include all applicable parameters
        workflow_params.update({
            "n_trials": args.n_trials,
            "optimization_metric": args.optimization_metric,
            "n_simulations": args.n_simulations,
            "keep_permuted_data": args.keep_permuted_data,
            "plot": args.plot,
            "enhanced_plots": args.enhanced_plots
        })
        
        # Add max_combinations parameter if provided
        if args.max_combinations is not None:
            workflow_params["max_combinations"] = args.max_combinations
    
    # Run workflow with unified parameters
    try:
        result = run_unified_workflow(args.workflow, **workflow_params)
        
        # Display status message
        if result["status"] == "success":
            logger.info("\n" + "=" * 80)
            logger.info(f"Workflow completed successfully. Results saved to: {output_dir}")
        else:
            logger.error(f"\nError: {result.get('message', 'Unknown error')}")
        
        # For monte carlo workflow, just check if the summary file exists
        # No need to generate it here as it's done in monte_carlo_workflow.py
        if args.workflow == "monte_carlo":
            logger.info(f"Monte Carlo workflow completed. Checking for summary file...")
            
            # Check if output_dir exists
            if not os.path.exists(output_dir):
                logger.warning(f"Output directory doesn't exist: {output_dir}")
                # Try to find correct output dir from result if available
                for strategy_name, strategy_data in result.get("results", {}).items():
                    if "output_dir" in strategy_data:
                        output_dir = strategy_data["output_dir"]
                        logger.info(f"Using output directory from result: {output_dir}")
                        break
            
            # If we have a valid output_dir, check for summary file
            if os.path.exists(output_dir):
                summary_file = os.path.join(output_dir, "monte_carlo_summary.txt")
                if os.path.exists(summary_file):
                    logger.info(f"Monte Carlo summary file exists at: {summary_file}")
                    
                    # Show the first few lines of the summary
                    try:
                        with open(summary_file, 'r') as f:
                            first_lines = [next(f) for _ in range(5)]
                        logger.info(f"Summary file contents (first 5 lines):")
                        for line in first_lines:
                            logger.info(line.rstrip())
                    except Exception as read_err:
                        logger.warning(f"Could not read summary file: {str(read_err)}")
                else:
                    logger.warning(f"Monte Carlo summary file not found at: {summary_file}")
            else:
                logger.error(f"Cannot find valid output directory for summary file")
        
        # Exit with error if the workflow failed
        if result["status"] != "success":
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"\nError during workflow execution: {str(e)}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)
    
    # Reset logging level if it was changed
    if args.verbose:
        logging_system.set_level('INFO')

if __name__ == "__main__":
    run_cli() 