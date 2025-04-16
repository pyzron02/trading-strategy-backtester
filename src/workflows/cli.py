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

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Unified workflow for trading strategy backtesting")
    
    # Common arguments
    parser.add_argument("--workflow", type=str, choices=[
        "simple", "optimization", "monte_carlo", "walkforward", "complete"
    ], required=True, help="Type of workflow to run")
    parser.add_argument("--strategy", type=str, required=True, help="Name of trading strategy")
    parser.add_argument("--tickers", type=str, nargs='+', help="Ticker symbols (space or comma separated)")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, default="2025-01-01", help="End date in YYYY-MM-DD format")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--param-file", type=str, help="Parameter file for the strategy")
    parser.add_argument("--plot", action="store_true", help="Generate and save plots of backtest results (disabled by default)")
    parser.add_argument("--enhanced-plots", action="store_true", help="Generate enhanced visualization dashboard for Monte Carlo simulations (disabled by default)")
    
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
    
    # Process tickers (handles both comma and space separated formats)
    tickers = process_tickers(args.tickers)
    
    # Set output directory if not specified
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(project_root, "output", f"{args.strategy}_{args.workflow}_{timestamp}")
        logger.info(f"No output directory specified. Using: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set logging level if verbose
    if args.verbose:
        logging_system.set_level('DEBUG')
        logger.debug("Verbose logging enabled")
    
    # Check parameter file compatibility with selected workflow
    if args.param_file:
        check_workflow_param_file_compatibility(args.workflow, args.param_file)
    
    # Prepare parameters dictionary
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
    
    # Add workflow-specific parameters
    if args.workflow in ["optimization", "walkforward", "complete"]:
        workflow_params.update({
            "n_trials": args.n_trials,
            "optimization_metric": args.optimization_metric
        })
        
        # Add max_combinations parameter if provided
        if args.max_combinations is not None:
            workflow_params["max_combinations"] = args.max_combinations
    
    if args.workflow in ["monte_carlo", "complete"]:
        workflow_params.update({
            "n_simulations": args.n_simulations,
            "keep_permuted_data": args.keep_permuted_data
        })
    
    if args.workflow in ["walkforward"]:
        workflow_params.update({
            "window_size": args.window_size,
            "step_size": args.step_size
        })
    
    # Add plot parameter for all workflows that support plotting
    if args.workflow in ["simple", "monte_carlo", "complete"]:
        workflow_params.update({
            "plot": args.plot
        })
        
    # Add enhanced_plots parameter for monte_carlo workflows
    if args.workflow in ["monte_carlo", "complete"] and args.enhanced_plots:
        workflow_params.update({
            "enhanced_plots": args.enhanced_plots
        })
    
    # Run workflow with unified parameters
    try:
        result = run_unified_workflow(args.workflow, **workflow_params)
        
        if result["status"] == "success":
            logger.info("\n" + "=" * 80)
            logger.info(f"Workflow completed successfully. Results saved to: {output_dir}")
        else:
            logger.error(f"\nError: {result.get('message', 'Unknown error')}")
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