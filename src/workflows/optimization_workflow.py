#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization workflow module.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import datetime
import uuid
from tqdm import tqdm

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
    logger, logging_system, print_workflow_log, adapt_strategy_parameters,
    setup_output_dir_logging, remove_output_dir_logging,
    check_logs_for_errors, print_error_report
)

# Import engine components
from engine.run_backtest import run_backtest
from engine.testing.in_sample_excellence import InSampleExcellence
from workflows.simple_workflow import ensure_data_available
from utils.error_reporting import create_stage_error_report, StageError, StageErrorReport

@time_execution("optimization workflow")
def run_optimization_workflow(
    strategy=None,  # New parameter to support unified_workflow
    strategy_name=None,  # Original parameter
    tickers=None,
    start_date=None,
    end_date=None,
    output_dir=None,
    parameters=None,
    param_file=None,
    n_trials=50,
    optimization_metric="sharpe_ratio",
    max_combinations=None,
    verbose=False,
    initial_capital=100000.0,
    commission=0.001,
    data_dir="input",
    plot=False,
    progress_callback=None,
    progress_file=None,
    stock_csv=None,
    _temp_files_to_cleanup=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run an optimization workflow for a trading strategy.
    
    Args:
        strategy: Name of the strategy to optimize (alternative to strategy_name)
        strategy_name: Name of the strategy to optimize
        tickers: List of ticker symbols
        start_date: Start date for backtest
        end_date: End date for backtest
        output_dir: Directory for output files
        parameters: Dictionary of strategy parameters (overrides param_file)
        param_file: File with parameter grid definitions
        n_trials: Number of trials to run
        optimization_metric: Metric to optimize for
        max_combinations: Maximum parameter combinations to try
        verbose: Whether to print detailed logs
        initial_capital: Initial capital for backtest
        commission: Commission per trade
        data_dir: Directory with data files
        plot: Whether to plot results
        progress_callback: Callback for progress updates
        progress_file: File to write progress updates
        stock_csv: CSV file with stock data
        _temp_files_to_cleanup: List of temporary files to clean up
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with optimization results
    """
    # Use strategy if provided, otherwise use strategy_name
    if strategy is not None and strategy_name is None:
        strategy_name = strategy
    elif strategy is None and strategy_name is None:
        return {
            "status": "error",
            "message": "Either strategy or strategy_name must be provided"
        }
    
    # Track temporary files if not already tracking
    if _temp_files_to_cleanup is None:
        _temp_files_to_cleanup = []
    
    # Create a unique output directory if none is provided
    if not output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = str(uuid.uuid4())[:8]  # For uniqueness
        output_dir = os.path.join(project_root, "output", f"{strategy_name}_optimization_{timestamp}_{run_id}")
        logger.info(f"Creating unique output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set logging level based on verbose flag
    if verbose:
        logging_system.set_level('DEBUG', 'workflows')
    
    # Set up progress file if provided
    if progress_file:
        with open(progress_file, 'w') as f:
            json.dump({
                "progress": 0,
                "status": "Starting optimization",
                "current_step": "Initializing",
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=4)
    
    # Find parameter grid file if not provided
    if not param_file:
        # Look in several possible locations for parameter grid files
        # Also search for lowercase strategy name and snake_case variation
        strategy_snake_case = ''.join(['_'+c.lower() if c.isupper() else c.lower() for c in strategy_name]).lstrip('_')
        possible_locations = [
            os.path.join(project_root, "input", "parameter_grids", f"{strategy_name}_grid.json"),
            os.path.join(project_root, "input", "parameter_grids", f"{strategy_name.lower()}_grid.json"),
            os.path.join(project_root, "input", "parameter_grids", f"{strategy_snake_case}_grid.json"),
            os.path.join(project_root, "input", f"{strategy_name}_grid.json"),
            os.path.join(project_root, "input", "grids", f"{strategy_name}_grid.json"),
            os.path.join(project_root, "input", f"{strategy_name.lower()}_grid.json"),
        ]
        
        # Debug all locations we're checking
        logger.info(f"Searching for parameter grid file for {strategy_name} in:")
        for location in possible_locations:
            logger.info(f"  Checking: {location} (exists: {os.path.exists(location)})")
            if os.path.exists(location):
                param_file = location
                logger.info(f"  Found parameter grid file: {location}")
                break
        
        if not param_file:
            # If no grid file is found, create a basic one from the strategy parameters
            logger.warning(f"No parameter grid file found for {strategy_name}. Creating a basic grid.")
            
            # Get default parameters
            param_file = find_strategy_param_file(strategy_name)
            if param_file:
                try:
                    with open(param_file, 'r') as f:
                        params = json.load(f)
                    
                    # Create a simple grid with some variations
                    param_grid = {}
                    for key, value in params.items():
                        if isinstance(value, (int, float)) and key not in ['initial_capital', 'commission']:
                            # Create a range of values around the default
                            if value == 0:
                                param_grid[key] = [0, 1, 2, 5, 10]
                            else:
                                param_grid[key] = [
                                    value * 0.5,
                                    value * 0.75,
                                    value,
                                    value * 1.25,
                                    value * 1.5
                                ]
                    
                    # Save the grid to a temporary file
                    param_file = os.path.join(output_dir, f"{strategy_name}_grid.json")
                    with open(param_file, 'w') as f:
                        json.dump(param_grid, f, indent=4)
                    
                    logger.info(f"Created parameter grid file: {param_file}")
                    
                    # Track for cleanup
                    _temp_files_to_cleanup.append(param_file)
                except Exception as e:
                    logger.error(f"Error creating parameter grid: {e}")
                    
                    # Log workflow failure
                    print_workflow_log(
                        workflow_name="Optimization Workflow",
                        strategy_name=strategy_name,
                        tickers=tickers,
                        start_date=start_date,
                        end_date=end_date,
                        status="FAILED",
                        additional_info={"error": f"Parameter grid file not found and could not create one: {str(e)}"}
                    )
                    
                    

                    # Generate stage error report

                    try:

                        create_stage_error_report(output_dir, 'optimization', strategy_name)

                    except Exception as report_err:

                        logger.error(f"Error generating stage error report: {report_err}")
                        
                    return {"status": "error", "message": f"Parameter grid file not found and could not create one: {str(e)}"}
            else:
                logger.error(f"Error: Parameter grid file not found for {strategy_name} and no default parameters available")
                
                # Log workflow failure
                print_workflow_log(
                    workflow_name="Optimization Workflow",
                    strategy_name=strategy_name,
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date,
                    status="FAILED",
                    additional_info={"error": "Parameter grid file not found and no default parameters available"}
                )
                
                return {"status": "error", "message": "Parameter grid file not found and no default parameters available"}
    
    if not os.path.exists(param_file):
        logger.error(f"Error: Parameter grid file does not exist: {param_file}")
        
        # Log workflow failure
        print_workflow_log(
            workflow_name="Optimization Workflow",
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            status="FAILED",
            additional_info={"error": f"Parameter grid file does not exist: {param_file}"}
        )
        
        return {"status": "error", "message": f"Parameter grid file does not exist: {param_file}"}
    
    print_section("Running Optimization")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Tickers: {', '.join(tickers)}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Optimization metric: {optimization_metric}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Parameter grid file: {param_file}")
    
    try:
        # Extract keep_all_results from kwargs if available
        keep_all_results = kwargs.get("keep_all_results", False)
        logger.info(f"Keep all parameter set results: {keep_all_results}")
        
        # Initialize optimizer
        optimizer = InSampleExcellence(
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            param_grid_file=param_file,
            n_trials=n_trials,
            optimization_metric=optimization_metric,
            output_dir=output_dir,
            initial_capital=initial_capital,
            commission=commission,
            data_dir=data_dir,
            max_combinations=max_combinations,
            verbose=verbose,
            plot=plot,  # Pass the plot parameter to control chart generation
            keep_results=keep_all_results  # Pass keep_all_results as keep_results
        )
        
        # Run optimization
        result = optimizer.run()
        
        # Handle different return types from optimizer.run()
        if isinstance(result, tuple) and len(result) == 2:
            best_params, trials_df = result
        elif isinstance(result, dict) and 'parameters' in result:
            best_params = result.get('parameters', {})
            trials_df = result.get('all_results', pd.DataFrame())
        else:
            best_params = None
            trials_df = None
            logger.error(f"Unexpected result type from optimizer.run(): {type(result)}")
        
        # Check if we have valid results
        if best_params is None or trials_df is None:
            valid_results = False
        elif isinstance(trials_df, pd.DataFrame):
            valid_results = not trials_df.empty
        else:
            valid_results = False
            
        if not valid_results:
            logger.error("Optimization failed to produce valid results")
            
            # Log workflow failure
            print_workflow_log(
                workflow_name="Optimization Workflow",
                strategy_name=strategy_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                status="FAILED",
                additional_info={"error": "Optimization failed to produce valid results"}
            )
            
            return {
                "status": "error", 
                "message": "Optimization failed to produce valid results",
                "output_dir": output_dir
            }
        
        # Check if optimization metric is present in trials_df
        if optimization_metric not in trials_df.columns:
            logger.error(f"Optimization metric '{optimization_metric}' not found in results")
            
            # Log workflow failure
            print_workflow_log(
                workflow_name="Optimization Workflow",
                strategy_name=strategy_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                status="FAILED",
                additional_info={"error": f"Optimization metric '{optimization_metric}' not found in results"}
            )
            
            return {
                "status": "error",
                "message": f"Optimization metric '{optimization_metric}' not found in results",
                "output_dir": output_dir
            }
            
        # Ensure we have valid values in the optimization metric
        if not isinstance(trials_df, pd.DataFrame):
            logger.error(f"trials_df is not a DataFrame, it's a {type(trials_df)}")
            
            # Log workflow failure
            print_workflow_log(
                workflow_name="Optimization Workflow",
                strategy_name=strategy_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                status="FAILED",
                additional_info={"error": f"trials_df is not a DataFrame, it's a {type(trials_df)}"}
            )
            
            return {
                "status": "error",
                "message": f"trials_df is not a DataFrame, it's a {type(trials_df)}",
                "output_dir": output_dir
            }
            
        if optimization_metric not in trials_df.columns:
            logger.error(f"Optimization metric '{optimization_metric}' not found in results columns: {list(trials_df.columns)}")
            
            # Log workflow failure
            print_workflow_log(
                workflow_name="Optimization Workflow",
                strategy_name=strategy_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                status="FAILED",
                additional_info={"error": f"Optimization metric '{optimization_metric}' not found in results columns: {list(trials_df.columns)}"}
            )
            
            return {
                "status": "error",
                "message": f"Optimization metric '{optimization_metric}' not found in results columns: {list(trials_df.columns)}",
                "output_dir": output_dir
            }
            
        if trials_df[optimization_metric].isna().all():
            logger.error(f"All values for optimization metric '{optimization_metric}' are NaN")
            
            # Log workflow failure
            print_workflow_log(
                workflow_name="Optimization Workflow",
                strategy_name=strategy_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                status="FAILED",
                additional_info={"error": f"All values for optimization metric '{optimization_metric}' are NaN"}
            )
            
            return {
                "status": "error",
                "message": f"All values for optimization metric '{optimization_metric}' are NaN",
                "output_dir": output_dir
            }
        
        # Run backtest with best parameters
        print_section("Running Backtest with Optimized Parameters")
        print_parameters(best_params)
        
        # Save best parameters to a file
        best_params_file = os.path.join(output_dir, f"{strategy_name}_best_params.json")
        with open(best_params_file, "w") as f:
            json.dump(best_params, f, indent=4)
        
        # Convert numerical parameters from numpy types to Python types and integers where needed
        converted_params = {}
        for key, value in best_params.items():
            # Convert float parameters to int for parameters that likely need integers
            if isinstance(value, (np.float64, np.float32, float)) and key in ['sma_period', 'max_positions']:
                converted_params[key] = int(value)
            elif isinstance(value, (np.float64, np.float32, np.int64, np.int32)):
                # Convert numpy types to Python native types
                converted_params[key] = value.item()
            else:
                converted_params[key] = value
                
        logger.info(f"Converting parameters for backtest: {best_params} -> {converted_params}")
        
        # Run backtest with converted best parameters
        backtest_result = run_backtest(
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            parameters=converted_params,
            plot=plot,  # Use the plot parameter passed to the function
            initial_capital=initial_capital,
            commission=commission,
            data_dir=data_dir,
            verbose=verbose
        )
        
        if not backtest_result:
            logger.error("Backtest with optimized parameters failed")
            
            # Log workflow failure
            print_workflow_log(
                workflow_name="Optimization Workflow",
                strategy_name=strategy_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                status="FAILED",
                additional_info={"error": "Backtest with optimized parameters failed"}
            )
            
            return {
                "status": "error", 
                "message": "Backtest with optimized parameters failed",
                "output_dir": output_dir
            }
        
        # Extract and display results
        results = {
            "strategy_name": strategy_name,
            "dates": {
                "start_date": start_date,
                "end_date": end_date
            },
            "best_parameters": best_params,
            "metrics": backtest_result.get("metrics", {}),
            "trials_summary": {
                "n_trials": n_trials,
                "optimization_metric": optimization_metric,
                "best_value": None  # Initialize to None
            }
        }
        
        # Safely calculate best value
        try:
            # Make sure we have a valid DataFrame and column
            if isinstance(trials_df, pd.DataFrame) and optimization_metric in trials_df.columns:
                if optimization_metric.startswith("max_drawdown"):
                    best_value = trials_df[optimization_metric].min()
                else:
                    best_value = trials_df[optimization_metric].max()
                    
                # Check if best_value is valid
                if not pd.isna(best_value) and np.isfinite(best_value):
                    results["trials_summary"]["best_value"] = best_value
                else:
                    logger.warning(f"Best value for {optimization_metric} is not valid: {best_value}")
                    results["trials_summary"]["best_value"] = "N/A"
            else:
                logger.warning(f"Cannot calculate best value: trials_df is not a valid DataFrame or missing column {optimization_metric}")
                results["trials_summary"]["best_value"] = "N/A"
        except Exception as e:
            logger.warning(f"Could not calculate best value for {optimization_metric}: {str(e)}")
            results["trials_summary"]["best_value"] = "N/A"
        
        # Save trials dataframe
        trials_file = os.path.join(output_dir, "optimization_trials.csv")
        if isinstance(trials_df, pd.DataFrame):
            trials_df.to_csv(trials_file, index=False)
        else:
            logger.warning(f"Cannot save trials data to CSV: trials_df is not a DataFrame, it's a {type(trials_df)}")
            # Create an empty CSV with column headers so downstream code doesn't break
            pd.DataFrame(columns=['param_' + k for k in best_params.keys()] + [optimization_metric]).to_csv(trials_file, index=False)
        
        print_section("Optimization Results")
        logger.info(f"Best {optimization_metric}: {results['trials_summary']['best_value']}")
        
        logger.info("\nOptimized Parameters:")
        print_parameters(results["best_parameters"])
        
        logger.info("\nPerformance Metrics with Best Parameters:")
        print_metrics(results["metrics"])
        
        # Save summary report
        summary_file = os.path.join(output_dir, "optimization_summary.txt")
        save_results_summary(results, summary_file, "Optimization Results")
        logger.info(f"\nDetailed results saved to: {output_dir}")
    
    except Exception as e:
        logger.error(f"Optimization workflow failed: {str(e)}")
        if verbose:
            logger.exception("Full error traceback:")
        
        # Log workflow failure
        print_workflow_log(
            workflow_name="Optimization Workflow",
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            status="FAILED",
            additional_info={"error": f"Optimization workflow failed: {str(e)}"}
        )
        
        # Check logs for errors
        logger.info("Checking logs for errors...")
        error_logs = check_logs_for_errors(output_dir)
        
        if error_logs:
            # Generate error report and save to file
            error_report_path = os.path.join(output_dir, "error_report.txt")
            print_error_report(error_logs, error_report_path)
            logger.warning(f"Found errors in logs. Error report saved to: {error_report_path}")
        
        return {
            "status": "error",
            "message": f"Optimization workflow failed: {str(e)}",
            "output_dir": output_dir
        }
    
    # Reset logging level if it was changed
    if verbose:
        logging_system.set_level('INFO', 'workflows')
    
    # Log workflow completion
    completion_info = {
        "best_value": results["trials_summary"]["best_value"],
        "total_trials": n_trials,
        "output_dir": output_dir
    }
    print_workflow_log(
        workflow_name="Optimization Workflow",
        strategy_name=strategy_name,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        status="COMPLETED",
        additional_info=completion_info
    )
    
    # Clean up temporary files
    files_to_delete = []
    files_skipped = []
    
    for temp_file in _temp_files_to_cleanup:
        if os.path.exists(temp_file):
            # Skip files in the workflow_configs directory
            if "workflow_configs" in temp_file:
                files_skipped.append(temp_file)
            else:
                files_to_delete.append(temp_file)
    
    if files_skipped:
        logger.info(f"Skipping cleanup of {len(files_skipped)} workflow config files")
        for file_path in files_skipped:
            logger.debug(f"Preserved file: {file_path}")
    
    for temp_file in files_to_delete:
        try:
            os.remove(temp_file)
            logger.info(f"Cleaned up temporary file: {temp_file}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary file: {str(e)}")
    
    # Check logs for errors
    logger.info("Checking logs for errors...")
    error_logs = check_logs_for_errors(output_dir)
    
    if error_logs:
        # Add log errors to the results
        results["log_errors"] = {
            "count": sum(len(errors) for errors in error_logs.values()),
            "files": len(error_logs)
        }
        
        # Generate error report and save to file
        error_report_path = os.path.join(output_dir, "error_report.txt")
        print_error_report(error_logs, error_report_path)
        logger.warning(f"Found errors in logs. Error report saved to: {error_report_path}")
    else:
        logger.info("No errors found in logs.")
        results["log_errors"] = {"count": 0, "files": 0}
    
    return {
        "status": "success",
        "results": results,
        "output_dir": output_dir
    } 