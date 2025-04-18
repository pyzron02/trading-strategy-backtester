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
from typing import Dict, List, Any, Optional

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
    logger, logging_system, print_workflow_log
)

# Import engine components
from engine.run_backtest import run_backtest
from engine.testing.in_sample_excellence import InSampleExcellence
from workflows.simple_workflow import ensure_data_available

@time_execution("optimization workflow")
def run_optimization_workflow(
    strategy_name: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_dir: str,
    parameters: Optional[Dict[str, Any]] = None,
    param_file: Optional[str] = None,
    n_trials: int = 50,
    optimization_metric: str = "sharpe_ratio",
    max_combinations: Optional[int] = None,
    verbose: bool = False,
    initial_capital: float = 100000.0,
    commission: float = 0.001,
    data_dir: str = "input",
    plot: bool = False
) -> Dict[str, Any]:
    """
    Run an optimization workflow for the given strategy.
    
    Args:
        strategy_name: Name of the strategy to run
        tickers: List of ticker symbols
        start_date: Start date for backtest in YYYY-MM-DD format
        end_date: End date for backtest in YYYY-MM-DD format
        output_dir: Directory to save results
        parameters: Dictionary of strategy parameters (overrides param_file)
        param_file: File with parameter definitions
        n_trials: Number of optimization trials
        optimization_metric: Metric to optimize for
        max_combinations: Maximum number of parameter combinations to test (grid search)
        verbose: Whether to print detailed output
        initial_capital: Initial capital for backtest
        commission: Commission rate for trades
        data_dir: Directory containing input data
    
    Returns:
        Dict containing the workflow results
    """
    # For backward compatibility
    param_grid_file = param_file
    
    # Log workflow start
    additional_info = {
        "n_trials": n_trials,
        "optimization_metric": optimization_metric,
        "output_dir": output_dir,
        "initial_capital": initial_capital,
        "commission": commission,
        "data_dir": data_dir,
        "plot": plot
    }
    if param_grid_file:
        additional_info["param_file"] = param_grid_file
    if max_combinations:
        additional_info["max_combinations"] = max_combinations
    
    print_workflow_log(
        workflow_name="Optimization Workflow",
        strategy_name=strategy_name,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        status="STARTED",
        additional_info=additional_info
    )
    
    print_header(f"Optimization Workflow: {strategy_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set logging level based on verbose flag
    if verbose:
        logging_system.set_level('DEBUG', 'workflows')
    
    # Find parameter grid file if not provided
    if not param_grid_file:
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
                param_grid_file = location
                logger.info(f"  Found parameter grid file: {location}")
                break
        
        if not param_grid_file:
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
                    param_grid_file = os.path.join(output_dir, f"{strategy_name}_grid.json")
                    with open(param_grid_file, 'w') as f:
                        json.dump(param_grid, f, indent=4)
                    
                    logger.info(f"Created parameter grid file: {param_grid_file}")
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
    
    if not os.path.exists(param_grid_file):
        logger.error(f"Error: Parameter grid file does not exist: {param_grid_file}")
        
        # Log workflow failure
        print_workflow_log(
            workflow_name="Optimization Workflow",
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            status="FAILED",
            additional_info={"error": f"Parameter grid file does not exist: {param_grid_file}"}
        )
        
        return {"status": "error", "message": f"Parameter grid file does not exist: {param_grid_file}"}
    
    print_section("Running Optimization")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Tickers: {', '.join(tickers)}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Optimization metric: {optimization_metric}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Parameter grid file: {param_grid_file}")
    
    try:
        # Initialize optimizer
        optimizer = InSampleExcellence(
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            param_grid_file=param_grid_file,
            n_trials=n_trials,
            optimization_metric=optimization_metric,
            output_dir=output_dir,
            initial_capital=initial_capital,
            commission=commission,
            data_dir=data_dir,
            max_combinations=max_combinations,
            verbose=verbose,
            plot=plot  # Pass the plot parameter to control chart generation
        )
        
        # Run optimization
        best_params, trials_df = optimizer.run()
        
        # Check if we have valid results
        if best_params is None or trials_df is None or (isinstance(trials_df, pd.DataFrame) and trials_df.empty):
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
        
        # Run backtest with best parameters
        backtest_result = run_backtest(
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            parameters=best_params,
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
        except Exception as e:
            logger.warning(f"Could not calculate best value for {optimization_metric}: {str(e)}")
            results["trials_summary"]["best_value"] = "N/A"
        
        # Save trials dataframe
        trials_file = os.path.join(output_dir, "optimization_trials.csv")
        trials_df.to_csv(trials_file, index=False)
        
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
    
    return {
        "status": "success",
        "results": results,
        "output_dir": output_dir
    } 