#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete workflow module that combines multiple workflow types.
"""
import os
import sys
import json
import datetime
import uuid
import time
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
    print_header, print_section, time_execution,
    logger, logging_system, adapt_strategy_parameters,
    setup_output_dir_logging, print_workflow_log,
    check_logs_for_errors, print_error_report
)

# Import individual workflow modules
from workflows.simple_workflow import run_simple_workflow, ensure_data_available
from workflows.optimization_workflow import run_optimization_workflow
from workflows.monte_carlo_workflow import run_monte_carlo_workflow
from workflows.walkforward_workflow import run_walkforward_workflow

@time_execution("complete workflow")
def run_complete_workflow(
    strategy=None,  # New parameter to support unified_workflow
    strategy_name=None,  # Original parameter
    tickers=None,
    start_date=None,
    end_date=None,
    output_dir=None,
    parameters=None,
    param_file=None,
    plot=True,
    n_trials=50,
    n_simulations=100,
    optimization_metric="sharpe_ratio",
    keep_permuted_data=False,
    verbose=False,
    initial_capital=100000.0,
    commission=0.001,
    data_dir="input",
    enhanced_plots=False,
    _temp_files_to_cleanup=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a complete workflow including backtest, optimization, and Monte Carlo simulation.
    
    Args:
        strategy: Name of the strategy to run (alternative to strategy_name)
        strategy_name: Name of the strategy to run
        tickers: List of ticker symbols
        start_date: Start date for backtest in YYYY-MM-DD format
        end_date: End date for backtest in YYYY-MM-DD format
        output_dir: Directory to save results
        parameters: Dictionary of strategy parameters (overrides param_file)
        param_file: File with parameter definitions
        plot: Whether to generate plots
        n_trials: Number of optimization trials
        n_simulations: Number of Monte Carlo simulations
        optimization_metric: Metric to optimize for
        keep_permuted_data: Whether to keep permuted data files from Monte Carlo
        verbose: Whether to print detailed output
        initial_capital: Initial capital for backtest
        commission: Commission rate for trades
        data_dir: Directory containing input data
        enhanced_plots: Whether to generate enhanced visualization dashboard for Monte Carlo
        _temp_files_to_cleanup: List of temporary files to clean up
        **kwargs: Additional arguments
    
    Returns:
        Dict containing the results from all workflow steps
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
    
    # Create a unique output directory for this run
    if not output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = str(uuid.uuid4())[:8]  # For uniqueness
        output_dir = os.path.join(project_root, "output", f"{strategy_name}_complete_{timestamp}_{run_id}")
        logger.info(f"Creating unique output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging for this run
    setup_output_dir_logging(output_dir, strategy_name, "complete")
    
    # Log the start of the workflow
    print_header("COMPLETE WORKFLOW")
    print_workflow_log("Complete", strategy_name, tickers, start_date, end_date, 
                      additional_info={"output_dir": output_dir})
    
    logger.info(f"Complete workflow started for strategy: {strategy_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Set up progress file if provided
    progress_file = None
    if kwargs.get('progress_file'):
        progress_file = kwargs['progress_file']
        with open(progress_file, 'w') as f:
            json.dump({
                "progress": 0,
                "status": "Starting complete workflow",
                "current_step": "Initializing",
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=4)

    # Create parameter file from parameters if provided
    if parameters and not param_file:
        # Create a temporary parameter file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        param_dir = os.path.join(project_root, "input", "parameters")
        os.makedirs(param_dir, exist_ok=True)
        param_file = os.path.join(param_dir, f"{strategy_name.lower()}_params_{timestamp}.json")
        
        try:
            with open(param_file, 'w') as f:
                json.dump(parameters, f, indent=2)
            logger.info(f"Created temporary parameter file from provided parameters: {param_file}")
            # Track for cleanup
            _temp_files_to_cleanup.append(param_file)
        except Exception as e:
            logger.error(f"Error creating temporary parameter file: {str(e)}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up standardized subdirectories for each part of the workflow
    # These directory names will be used consistently throughout the workflow
    _workflow_dirs = {
        'simple_backtest': os.path.join(output_dir, "01_simple_backtest"),
        'optimization': os.path.join(output_dir, "02_optimization"),
        'walkforward': os.path.join(output_dir, "03_walkforward"),
        'monte_carlo': os.path.join(output_dir, "04_monte_carlo"),
        'optimized_backtest': os.path.join(output_dir, "05_optimized_backtest")
    }
    
    # Create all subdirectories
    for dir_path in _workflow_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    # Store in variables for easy reference
    simple_dir = _workflow_dirs['simple_backtest']
    optimization_dir = _workflow_dirs['optimization']
    walkforward_dir = _workflow_dirs['walkforward']
    montecarlo_dir = _workflow_dirs['monte_carlo']
    optimized_backtest_dir = _workflow_dirs['optimized_backtest']
    
    # Set logging level based on verbose flag
    if verbose:
        logging_system.set_level('DEBUG', 'workflows')
    
    # Create the complete results structure
    combined_results = {
        "status": "incomplete",
        "strategy_name": strategy_name,
        "dates": {
            "start_date": start_date,
            "end_date": end_date
        },
        "output_dir": output_dir
    }
    
    try:
        # Check if stock_data.csv exists but don't regenerate it
        stock_data_path = os.path.join(data_dir, "stock_data.csv")
        if os.path.exists(stock_data_path):
            stock_csv = stock_data_path
            logger.info(f"Using existing stock data from: {stock_csv}")
        else:
            logger.error(f"Stock data file not found at: {stock_data_path}. Please run data_setup.py first.")
            return {
                "status": "error",
                "message": f"Stock data file not found. Please run data_setup.py first.",
                "output_dir": output_dir
            }
        
        # Find progress file if it exists in the workflow configuration
        progress_file = None
        workflow_config_path = os.path.join(output_dir, 'workflow_config.json')
        if os.path.exists(workflow_config_path):
            try:
                with open(workflow_config_path, 'r') as f:
                    workflow_config = json.load(f)
                if 'frontend' in workflow_config and 'progress_file' in workflow_config['frontend']:
                    progress_file = workflow_config['frontend']['progress_file']
                    logger.info(f"Found progress file for frontend updates: {progress_file}")
                    
                    # Update initial progress
                    if os.path.exists(progress_file):
                        import json
                        with open(progress_file, 'r') as f:
                            progress_data = json.load(f)
                        
                        progress_data.update({
                            'current_step': "Starting complete workflow",
                            'progress': 5,
                            'current_step_progress': 100,
                            'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        
                        with open(progress_file, 'w') as f:
                            json.dump(progress_data, f, indent=4)
            except Exception as e:
                logger.warning(f"Error reading workflow config for progress updates: {e}")

        # Reorder workflow steps to run Simple Backtest first 
        # Step 1: Run Simple Backtest as a baseline
        print_section("Step 1: Simple Backtest (Baseline)")
        
        # Use the standardized directory for simple backtest
        simple_output_dir = _workflow_dirs['simple_backtest']
        
        # Update progress for simple backtest step
        if progress_file and os.path.exists(progress_file):
            try:
                import json
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                progress_data.update({
                    'current_step': "Step 1: Simple Backtest (Baseline)",
                    'progress': 10,
                    'current_step_progress': 0,
                    'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=4)
            except Exception as e:
                logger.warning(f"Error updating progress file: {e}")
                
        simple_output_dir = os.path.join(output_dir, "01_simple_backtest")
        if not os.path.exists(simple_output_dir):
            os.makedirs(simple_output_dir)
        
        # Prepare simple backtest kwargs
        simple_backtest_kwargs = {
            "strategy": strategy_name,
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "output_dir": simple_output_dir,
            "parameters": parameters,
            "plot": plot,
            "verbose": verbose,
            "initial_capital": initial_capital,
            "commission": commission,
            "data_dir": data_dir
        }
        
        simple_result = run_simple_workflow(**simple_backtest_kwargs)
        
        # Store simple backtest results in combined_results
        if simple_result and simple_result.get("status") == "success":
            # Debug the structure of simple_result
            logger.debug(f"Simple backtest result keys: {simple_result.keys() if isinstance(simple_result, dict) else 'Not a dict'}")
            
            # Try to extract the metrics from the right location
            simple_data = {}
            
            # First look for results key
            if "results" in simple_result:
                simple_data = simple_result["results"]
                logger.debug(f"Found results key with sub-keys: {simple_data.keys() if isinstance(simple_data, dict) else 'Not a dict'}")
            
            # If no results found or results is empty, look for equity_curve or metrics directly
            if not simple_data and "metrics" in simple_result:
                simple_data = {"metrics": simple_result["metrics"]}
                logger.debug("Found metrics directly in simple_result")
                
            if "equity_curve" in simple_result:
                if not simple_data:
                    simple_data = {}
                simple_data["equity_curve"] = simple_result["equity_curve"]
                logger.debug("Found equity_curve directly in simple_result")
            
            combined_results["simple_backtest"] = simple_data
            logger.info("Simple backtest completed successfully")
        else:
            logger.warning(f"Simple backtest failed: {simple_result.get('message', 'Unknown error')}")
            combined_results["simple_backtest"] = {"status": "error", "message": simple_result.get('message', 'Unknown error')}
            combined_results["status"] = "partial"

        # Step 2: Run optimization to find the best parameters
        print_section("Step 2: Parameter Optimization")
        
        # Update progress for optimization step
        if progress_file and os.path.exists(progress_file):
            try:
                import json
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                progress_data.update({
                    'current_step': "Step 2: Parameter Optimization",
                    'progress': 30,
                    'current_step_progress': 0,
                    'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=4)
            except Exception as e:
                logger.warning(f"Error updating progress file: {e}")
                
        # Use the standardized directory path
        optimization_output_dir = _workflow_dirs['optimization']
            
        # Prepare optimization kwargs
        # Adapt strategy parameters if needed
        adapted_parameters = adapt_strategy_parameters(strategy_name, parameters)
        
        # If parameters were adapted, log this
        if adapted_parameters != parameters and adapted_parameters:
            logger.info(f"Using adapted parameters for {strategy_name} optimization")
            logger.debug(f"Original parameters: {parameters}")
            logger.debug(f"Adapted parameters: {adapted_parameters}")
        
        optimization_kwargs = {
            "strategy": strategy_name,
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "output_dir": optimization_output_dir,
            "parameters": adapted_parameters,
            "param_file": param_file,
            "n_trials": n_trials,
            "optimization_metric": optimization_metric,
            "verbose": verbose,
            "initial_capital": initial_capital,
            "commission": commission,
            "data_dir": data_dir,
            "plot": plot,
            "keep_all_results": kwargs.get("optimization", {}).get("keep_all_results", False)
        }
        
        optimization_result = run_optimization_workflow(**optimization_kwargs)
        
        if optimization_result["status"] == "success":
            # Extract best parameters for Monte Carlo
            best_params = optimization_result["results"].get("best_parameters", {})
            combined_results["optimization"] = optimization_result.get("results", {})
            logger.info("Optimization completed successfully")
        else:
            logger.warning(f"Optimization failed: {optimization_result.get('message', 'Unknown error')}")
            combined_results["optimization"] = {"status": "error", "message": optimization_result.get('message', 'Unknown error')}
            # Fall back to simple backtest parameters if available
            best_params = combined_results.get("simple_backtest", {}).get("parameters", {})
            combined_results["status"] = "partial"

        # Step 3: Run Walk Forward Analysis with the best parameters
        print_section("Step 3: Walk Forward Analysis")
        
        # Update progress for walk forward step
        if progress_file and os.path.exists(progress_file):
            try:
                import json
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                progress_data.update({
                    'current_step': "Step 3: Walk Forward Analysis",
                    'progress': 60,
                    'current_step_progress': 0,
                    'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=4)
            except Exception as e:
                logger.warning(f"Error updating progress file: {e}")
        
        # Use the standardized directory path
        walkforward_dir = _workflow_dirs['walkforward']
        
        # Prepare walk forward kwargs using the best parameters from optimization
        walkforward_kwargs = {
            "strategy": strategy_name,
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "output_dir": walkforward_dir,
            "parameters": best_params,  # Use the best parameters from optimization
            "n_trials": n_trials,  # For optimization within walk forward 
            "optimization_metric": optimization_metric,
            "verbose": verbose,
            "initial_capital": initial_capital,
            "commission": commission,
            "data_dir": data_dir,
            "plot": plot
        }
        
        # Add walk forward specific parameters from kwargs if available
        if "walk_forward" in kwargs:
            wf_params = kwargs["walk_forward"]
            # Add number of windows if specified
            if "n_windows" in wf_params:
                walkforward_kwargs["window_size"] = int((datetime.datetime.strptime(end_date, '%Y-%m-%d') - 
                                                         datetime.datetime.strptime(start_date, '%Y-%m-%d')).days / wf_params["n_windows"])
            # Add in-sample percentage if specified
            if "in_sample_pct" in wf_params:
                walkforward_kwargs["in_sample_pct"] = wf_params["in_sample_pct"]
            
        walkforward_result = run_walkforward_workflow(**walkforward_kwargs)
        
        # Store walk forward results in combined_results
        if walkforward_result and walkforward_result.get("status") == "success":
            # Extract actual results from the returned object
            walkforward_data = {}
            
            # Try different possible locations for the walk forward results
            if "walkforward_results" in walkforward_result:
                walkforward_data = walkforward_result["walkforward_results"]
            elif "results" in walkforward_result:
                walkforward_data = walkforward_result["results"]
            
            # Check if results actually contains data
            if not walkforward_data:
                logger.warning("Walk Forward results are empty, checking for specific object structure")
                # Try to extract key files and data structures that should be present
                if "output_dir" in walkforward_result:
                    wf_dir = walkforward_result["output_dir"]
                    # Check for summary file
                    summary_file = os.path.join(wf_dir, "walkforward_summary.txt")
                    if os.path.exists(summary_file):
                        logger.info(f"Found walkforward summary file: {summary_file}")
                        # Read the summary content
                        with open(summary_file, 'r') as f:
                            summary_content = f.read()
                        # Add it to the data
                        walkforward_data["summary"] = summary_content
                        walkforward_data["summary_file"] = summary_file
                        
                    # Check for comparison data
                    comparison_file = os.path.join(wf_dir, "performance_comparison.csv")
                    if os.path.exists(comparison_file):
                        logger.info(f"Found walkforward comparison file: {comparison_file}")
                        try:
                            comparison_data = pd.read_csv(comparison_file, index_col=0)
                            walkforward_data["comparison"] = comparison_data
                        except Exception as e:
                            logger.warning(f"Error reading comparison file: {e}")
            
            # Assign to combined_results
            combined_results["walkforward"] = walkforward_data
            logger.info("Walk Forward Analysis completed successfully")
        else:
            logger.warning(f"Walk Forward Analysis failed: {walkforward_result.get('message', 'Unknown error')}")
            combined_results["walkforward"] = {"status": "error", "message": walkforward_result.get('message', 'Unknown error')}
            combined_results["status"] = "partial"

        # Step 4: Run optimized backtest with the best parameters from optimization
        print_section("Step 4: Backtesting with Optimized Parameters")
        
        # Update progress for optimized backtest step
        if progress_file and os.path.exists(progress_file):
            try:
                import json
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                progress_data.update({
                    'current_step': "Step 4: Backtesting with Optimized Parameters",
                    'progress': 60,
                    'current_step_progress': 0,
                    'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=4)
            except Exception as e:
                logger.warning(f"Error updating progress file: {e}")
        
        # Use the standardized directory path
        optimized_backtest_dir = _workflow_dirs['optimized_backtest']
        
        # Get the best parameters from optimization
        if (optimization_result and 
            optimization_result.get("status") == "success" and 
            "best_params" in optimization_result):
            best_params = optimization_result["best_params"]
            logger.info(f"Using optimized parameters: {best_params}")
        else:
            # If optimization failed, use original parameters
            logger.warning("Optimization failed or did not produce best parameters. Using original parameters.")
            best_params = parameters
        
        # Adapt best parameters for the strategy
        adapted_best_params = adapt_strategy_parameters(strategy_name, best_params)
        
        # If parameters were adapted, log this
        if adapted_best_params != best_params and adapted_best_params:
            logger.info(f"Using adapted optimized parameters for {strategy_name} backtest")
            logger.debug(f"Original best parameters: {best_params}")
            logger.debug(f"Adapted best parameters: {adapted_best_params}")
        
        # Prepare optimized backtest kwargs
        optimized_backtest_kwargs = {
            "strategy": strategy_name,
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "output_dir": optimized_backtest_dir,
            "parameters": adapted_best_params,
            "plot": plot,
            "verbose": verbose,
            "initial_capital": initial_capital,
            "commission": commission,
            "data_dir": data_dir
        }
        
        optimized_backtest_result = run_simple_workflow(**optimized_backtest_kwargs)
        
        # Store optimized backtest results in combined_results
        if optimized_backtest_result and optimized_backtest_result.get("status") == "success":
            # Debug the structure of optimized_backtest_result
            logger.debug(f"Optimized backtest result keys: {optimized_backtest_result.keys() if isinstance(optimized_backtest_result, dict) else 'Not a dict'}")
            
            # Try to extract the metrics from the right location
            optimized_backtest_data = {}
            
            # First look for results key
            if "results" in optimized_backtest_result:
                optimized_backtest_data = optimized_backtest_result["results"]
                logger.debug(f"Found results key with sub-keys: {optimized_backtest_data.keys() if isinstance(optimized_backtest_data, dict) else 'Not a dict'}")
            
            # If no results found or results is empty, look for equity_curve or metrics directly
            if not optimized_backtest_data and "metrics" in optimized_backtest_result:
                optimized_backtest_data = {"metrics": optimized_backtest_result["metrics"]}
                logger.debug("Found metrics directly in optimized_backtest_result")
                
            if "equity_curve" in optimized_backtest_result:
                if not optimized_backtest_data:
                    optimized_backtest_data = {}
                optimized_backtest_data["equity_curve"] = optimized_backtest_result["equity_curve"]
                logger.debug("Found equity_curve directly in optimized_backtest_result")
            
            combined_results["optimized_backtest"] = optimized_backtest_data
            logger.info("Optimized backtest completed successfully")
        else:
            logger.warning(f"Optimized backtest failed: {optimized_backtest_result.get('message', 'Unknown error')}")
            combined_results["optimized_backtest"] = {"status": "error", "message": optimized_backtest_result.get('message', 'Unknown error')}
        
        # Step 5: Run Monte Carlo simulation
        print_section("Step 5: Monte Carlo Simulation")
        
        # Update progress for Monte Carlo step
        if progress_file and os.path.exists(progress_file):
            try:
                import json
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                progress_data.update({
                    'current_step': "Step 5: Monte Carlo Simulation",
                    'progress': 80,
                    'current_step_progress': 0,
                    'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=4)
            except Exception as e:
                logger.warning(f"Error updating progress file: {e}")
        
        # Use the standardized directory path
        monte_carlo_output_dir = _workflow_dirs['monte_carlo']
        
        # We already adapted the best_params earlier, but to be safe
        # ensure we have adapted parameters for the Monte Carlo step too
        adapted_monte_carlo_params = adapt_strategy_parameters(strategy_name, best_params)
        
        # If parameters were adapted, log this
        if adapted_monte_carlo_params != best_params and adapted_monte_carlo_params:
            logger.info(f"Using adapted parameters for {strategy_name} Monte Carlo simulation")
            logger.debug(f"Original parameters: {best_params}")
            logger.debug(f"Adapted parameters: {adapted_monte_carlo_params}")
        
        # Prepare monte carlo kwargs
        monte_carlo_kwargs = {
            "strategy": strategy_name,
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "output_dir": monte_carlo_output_dir,
            "parameters": adapted_monte_carlo_params,
            "n_simulations": n_simulations,
            "keep_permuted_data": keep_permuted_data,
            "verbose": verbose,
            "initial_capital": initial_capital,
            "commission": commission,
            "data_dir": data_dir,
            "plot": plot,
            "enhanced_plots": enhanced_plots,
            "workflow_type": "complete"
        }
        
        monte_carlo_result = run_monte_carlo_workflow(**monte_carlo_kwargs)
        
        # Store Monte Carlo results in combined_results
        if monte_carlo_result and monte_carlo_result.get("status") == "success":
            # Print the monte_carlo_result keys for debugging
            logger.debug(f"Monte Carlo result keys: {monte_carlo_result.keys() if isinstance(monte_carlo_result, dict) else 'Not a dict'}")
            
            # Extract actual results from the returned object
            monte_carlo_data = {}
            
            # Try different possible locations for the Monte Carlo results
            if "monte_carlo_results" in monte_carlo_result:
                monte_carlo_data = monte_carlo_result["monte_carlo_results"]
            elif "results" in monte_carlo_result:
                if "monte_carlo_results" in monte_carlo_result["results"]:
                    monte_carlo_data = monte_carlo_result["results"]["monte_carlo_results"]
                else:
                    monte_carlo_data = monte_carlo_result["results"]
            
            # Assign to combined_results
            combined_results["monte_carlo"] = monte_carlo_data
            logger.info("Monte Carlo simulation completed successfully")
            
            # If all parts have run, mark as complete
            if combined_results.get("simple_backtest") and combined_results.get("optimization"):
                combined_results["status"] = "success"
            else:
                combined_results["status"] = "partial"
        else:
            logger.warning(f"Monte Carlo failed: {monte_carlo_result.get('message', 'Unknown error')}")
            combined_results["monte_carlo"] = {"status": "error", "message": monte_carlo_result.get('message', 'Unknown error')}
            combined_results["status"] = "partial"
        
        # Update progress for completion
        if progress_file and os.path.exists(progress_file):
            try:
                import json
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                progress_data.update({
                    'current_step': "Workflow Completed - Saving Results",
                    'progress': 95,
                    'current_step_progress': 100,
                    'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=4)
            except Exception as e:
                logger.warning(f"Error updating progress file: {e}")
        
        # Save a comprehensive summary of all results
        summary_file = os.path.join(output_dir, "complete_workflow_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"COMPLETE WORKFLOW SUMMARY: {strategy_name}\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic test information
            f.write(f"Strategy: {strategy_name}\n")
            f.write(f"Period: {start_date} to {end_date}\n")
            f.write(f"Tickers: {', '.join(tickers)}\n")
            f.write(f"Initial Capital: ${initial_capital:.2f}\n")
            f.write(f"Commission Rate: {commission:.4f}\n\n")
            
            # Simple Backtest Results
            f.write("\n" + "=" * 80 + "\n")
            f.write("SIMPLE BACKTEST RESULTS\n")
            f.write("=" * 80 + "\n")
            
            # Look for backtest results in the standard location
            if "simple_backtest" in combined_results and isinstance(combined_results["simple_backtest"], dict):
                backtest_data = combined_results["simple_backtest"]
                # First try to get metrics directly
                metrics = backtest_data.get("metrics", {})
                
                # If empty, check if metrics are nested in the result structure
                if not metrics and "results" in backtest_data:
                    metrics = backtest_data["results"].get("metrics", {})
                    
                # If still empty, try other possible locations
                if not metrics and "backtest_result" in backtest_data:
                    metrics = backtest_data["backtest_result"].get("metrics", {})
                
                logger.debug(f"Backtest metrics found: {bool(metrics)}")
                
                if metrics:
                    # Performance metrics
                    f.write("\n----- Performance Metrics -----\n")
                    f.write(f"Initial Portfolio Value: ${metrics.get('initial_value', initial_capital):.2f}\n")
                    f.write(f"Final Portfolio Value: ${metrics.get('final_value', initial_capital):.2f}\n")
                    f.write(f"Absolute Return: ${metrics.get('final_value', initial_capital) - metrics.get('initial_value', initial_capital):.2f}\n")
                    f.write(f"Total Return: {metrics.get('total_return', 0.0):.2%}\n")
                    f.write(f"Annual Return: {metrics.get('annual_return', 0.0):.2%}\n")
                    f.write(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.4f}\n")
                    f.write(f"Benchmark Return: {metrics.get('benchmark_return', 0.0):.2%}\n")
                    f.write(f"Alpha: {metrics.get('alpha', 0.0):.2%}\n")
                    
                    # Risk metrics
                    f.write("\n----- Risk Metrics -----\n")
                    f.write(f"Maximum Drawdown: {metrics.get('max_drawdown_pct', 0.0):.2f}%\n")
                    f.write(f"Maximum Drawdown (Money): ${metrics.get('max_drawdown_money', 0.0):.2f}\n")
                    
                    # Trade statistics
                    f.write("\n----- Trade Statistics -----\n")
                    f.write(f"Total Trades: {metrics.get('total_trades', 0)}\n")
                    f.write(f"Winning Trades: {metrics.get('winning_trades', 0)}\n")
                    f.write(f"Losing Trades: {metrics.get('losing_trades', 0)}\n")
                    f.write(f"Win Rate: {metrics.get('win_rate_pct', 0.0):.2f}%\n")
                    f.write(f"Profit Factor: {metrics.get('profit_factor', 0.0):.4f}\n")
                    f.write(f"Gross Profit: ${metrics.get('gross_profit', 0.0):.2f}\n")
                    f.write(f"Gross Loss: ${metrics.get('gross_loss', 0.0):.2f}\n")
                    f.write(f"Net Profit: ${metrics.get('net_profit', 0.0):.2f}\n")
                    f.write(f"Average Trade P&L: ${metrics.get('avg_trade_pnl', 0.0):.2f}\n")
                    f.write(f"Max Consecutive Wins: {metrics.get('max_consecutive_wins', 0)}\n")
                    f.write(f"Max Consecutive Losses: {metrics.get('max_consecutive_losses', 0)}\n")
                else:
                    f.write("No metrics available for simple backtest\n")
                    # Log debug info about the combined_results structure
                    logger.debug(f"Backtest data structure: {backtest_data.keys() if isinstance(backtest_data, dict) else 'Not a dict'}")
            else:
                f.write("Simple backtest failed or was skipped\n")
                # Log debug info about the combined_results structure
                logger.debug(f"Simple backtest data missing. Available keys: {combined_results.keys()}")
            
            # Optimization Results
            if "optimization" in combined_results and combined_results["optimization"].get("status") != "skipped":
                f.write("\n" + "=" * 80 + "\n")
                f.write("OPTIMIZATION RESULTS\n")
                f.write("=" * 80 + "\n")
                
                # Information about the optimization
                f.write(f"\nNumber of Trials: {n_trials}\n")
                f.write(f"Optimization Metric: {optimization_metric}\n\n")
                
                if "best_parameters" in combined_results.get("optimization", {}):
                    # Best parameters found
                    f.write("----- Best Parameters -----\n")
                    for key, value in combined_results["optimization"]["best_parameters"].items():
                        f.write(f"{key}: {value}\n")
                    
                    # Metrics with best parameters
                    metrics = combined_results["optimization"].get("metrics", {})
                    if metrics:
                        f.write("\n----- Performance with Best Parameters -----\n")
                        # Print comprehensive metrics for best parameters if available
                        if metrics:
                            f.write(f"Total Return: {metrics.get('total_return', 0):.2%}\n")
                            f.write(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}\n")
                            
                            if 'win_rate' in metrics and metrics['win_rate'] > 0:
                                f.write(f"Win Rate: {metrics.get('win_rate', 0):.2%}\n")
                                f.write(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n")
                                f.write(f"Total Trades: {metrics.get('total_trades', 0)}\n")
                            
                            if 'max_drawdown' in metrics:
                                # Use max_drawdown_pct if available, otherwise convert max_drawdown to percentage
                                if 'max_drawdown_pct' in metrics:
                                    # Backtrader returns the drawdown as a percentage already
                                    # Just ensure it's not unrealistically high
                                    max_dd = min(metrics.get('max_drawdown_pct', 0), 100)
                                    f.write(f"Max Drawdown: {max_dd:.2f}%\n")
                                else:
                                    # Convert from decimal to percentage display (max 100%)
                                    max_dd = min(metrics.get('max_drawdown', 0) * 100, 100)
                                    f.write(f"Max Drawdown: {max_dd:.2f}%\n")
                            
                            # Add advanced metrics if available
                            if 'annual_return' in metrics:
                                f.write(f"Annual Return: {metrics.get('annual_return', 0):.2%}\n")
                            
                            if 'alpha' in metrics:
                                f.write(f"Alpha: {metrics.get('alpha', 0):.2%}\n")
                    else:
                        f.write("\nNo metrics available for optimized parameters\n")
                else:
                    f.write("Optimization failed or no best parameters found\n")
            
            # Walk Forward Results
            if "walkforward" in combined_results:
                f.write("\n" + "=" * 80 + "\n")
                f.write("WALK FORWARD ANALYSIS RESULTS\n")
                f.write("=" * 80 + "\n")
                
                # Try to get walk forward results with fallbacks for different possible structures
                walkforward_data = combined_results.get("walkforward", {})
                walkforward_results = {}
                
                # First try direct access
                if isinstance(walkforward_data, dict):
                    # Try different possible keys where results might be stored
                    if "walkforward_results" in walkforward_data:
                        walkforward_results = walkforward_data["walkforward_results"]
                    elif "results" in walkforward_data:
                        walkforward_results = walkforward_data["results"]
                    else:
                        # If no nested results, maybe the data is directly in walkforward_data
                        # Check for expected walk forward fields
                        if any(key in walkforward_data for key in ['mean_return', 'probability_of_profit', 'initial_equity']):
                            walkforward_results = walkforward_data
                
                logger.debug(f"Walk Forward results found: {bool(walkforward_results)}")
                
                # Check for summary content directly
                if "summary" in walkforward_data and isinstance(walkforward_data["summary"], str):
                    # We have a summary text directly from the walkforward analysis
                    f.write("\n")
                    f.write(walkforward_data["summary"])
                    f.write("\n")
                elif "comparison" in walkforward_data and isinstance(walkforward_data["comparison"], pd.DataFrame):
                    # We have a comparison DataFrame, use it to create a summary
                    comparison_df = walkforward_data["comparison"]
                    f.write("\n----- Walk Forward Performance Comparison -----\n\n")
                    
                    # Format the comparison data for display
                    for idx, row in comparison_df.iterrows():
                        metric_name = idx.replace('_', ' ').title()
                        in_sample = row.get('In-Sample', 0)
                        out_sample = row.get('Out-of-Sample', 0)
                        difference = row.get('Difference', 0)
                        
                        # Format as percentages for return metrics
                        if 'return' in idx.lower() or 'alpha' in idx.lower():
                            f.write(f"{metric_name}: In-Sample: {in_sample*100:.2f}%, Out-of-Sample: {out_sample*100:.2f}%, Difference: {difference*100:.2f}%\n")
                        else:
                            f.write(f"{metric_name}: In-Sample: {in_sample:.4f}, Out-of-Sample: {out_sample:.4f}, Difference: {difference:.4f}\n")
                    
                    f.write("\n")
                elif walkforward_results:
                    # Summary stats from the results dict
                    f.write(f"\nNumber of Simulations: {walkforward_results.get('num_simulations', n_trials)}\n")
                    f.write(f"Confidence Level: {walkforward_results.get('confidence_level', 0.95):.0%}\n\n")
                    
                    # Basic statistics
                    f.write("----- Portfolio Statistics -----\n")
                    f.write(f"Initial Equity: ${walkforward_results.get('initial_equity', initial_capital):.2f}\n")
                    f.write(f"Final Equity (Original): ${walkforward_results.get('final_equity_original', initial_capital):.2f}\n")
                    f.write(f"Original Return: {walkforward_results.get('return_original', 0.0):.2%}\n\n")
                    
                    # Simulation results
                    f.write("----- Simulation Results -----\n")
                    f.write(f"Mean Final Equity: ${walkforward_results.get('mean_final_equity', initial_capital):.2f}\n")
                    f.write(f"Median Final Equity: ${walkforward_results.get('median_final_equity', initial_capital):.2f}\n")
                    f.write(f"Mean Return: {walkforward_results.get('mean_return', 0.0):.2%}\n\n")
                    
                    # Confidence intervals
                    confidence_level = walkforward_results.get('confidence_level', 0.95)
                    f.write(f"----- Confidence Intervals ({confidence_level:.0%}) -----\n")
                    f.write(f"Final Equity Range: ${walkforward_results.get('ci_lower_final_equity', 0.0):.2f} to " +
                            f"${walkforward_results.get('ci_upper_final_equity', 0.0):.2f}\n")
                    f.write(f"Return Range: {walkforward_results.get('ci_lower_return', 0.0):.2%} to " +
                            f"{walkforward_results.get('ci_upper_return', 0.0):.2%}\n\n")
                    
                    # Risk metrics
                    f.write("----- Risk Metrics -----\n")
                    f.write(f"Value at Risk (VaR {confidence_level:.0%}): " +
                            f"{walkforward_results.get('var_pct', 0.0):.2%}\n")
                    f.write(f"Conditional VaR (CVaR {confidence_level:.0%}): " +
                            f"{walkforward_results.get('cvar_pct', 0.0):.2%}\n")
                    f.write(f"Worst Return: {walkforward_results.get('worst_return', 0.0):.2%}\n")
                    f.write(f"Best Return: {walkforward_results.get('best_return', 0.0):.2%}\n\n")
                    
                    # Probability metrics
                    f.write("----- Probability Metrics -----\n")
                    f.write(f"Probability of Profit: {walkforward_results.get('probability_of_profit', 0.0):.2%}\n")
                else:
                    # Check if there's any output directory info we can use to search for walk-forward results
                    walkforward_output_dir = None
                    if "output_dir" in walkforward_data:
                        walkforward_output_dir = walkforward_data["output_dir"]
                    elif _workflow_dirs and "walkforward" in _workflow_dirs:
                        walkforward_output_dir = _workflow_dirs["walkforward"]
                    
                    if walkforward_output_dir and os.path.exists(walkforward_output_dir):
                        # Check for summary files in the output directory
                        potential_summary_files = [
                            os.path.join(walkforward_output_dir, "walkforward_summary.txt"),
                            os.path.join(walkforward_output_dir, "summary.txt"),
                            os.path.join(walkforward_output_dir, "results.txt")
                        ]
                        
                        for summary_file in potential_summary_files:
                            if os.path.exists(summary_file):
                                try:
                                    with open(summary_file, 'r') as sf:
                                        summary_content = sf.read()
                                    f.write("\n")
                                    f.write(summary_content)
                                    f.write("\n")
                                    logger.info(f"Added walkforward summary from {summary_file}")
                                    break
                                except Exception as e:
                                    logger.warning(f"Error reading summary file {summary_file}: {e}")
                        else:
                            f.write("Walk Forward Analysis completed but no detailed results available\n")
                    else:
                        f.write("Walk Forward Analysis failed or no results available\n")
                    # Log debug info about the combined_results structure
                    logger.debug(f"Walk Forward data structure: {walkforward_data.keys() if isinstance(walkforward_data, dict) else 'Not a dict'}")
            else:
                # No walk forward section found
                logger.debug(f"Walk Forward data missing. Available keys: {combined_results.keys()}")
                f.write("\n" + "=" * 80 + "\n")
                f.write("WALK FORWARD ANALYSIS RESULTS\n")
                f.write("=" * 80 + "\n")
                f.write("Walk Forward Analysis was not performed or failed.\n")
            
            # Overall workflow status
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"WORKFLOW STATUS: {combined_results['status'].upper()}\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Complete workflow finished. Results saved to: {output_dir}")
        
        # Reset logging level if it was changed
        if verbose:
            logging_system.set_level('INFO', 'workflows')
        
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
            combined_results["log_errors"] = {
                "count": sum(len(errors) for errors in error_logs.values()),
                "files": len(error_logs)
            }
            
            # Generate error report and save to file
            error_report_path = os.path.join(output_dir, "error_report.txt")
            print_error_report(error_logs, error_report_path)
            logger.warning(f"Found errors in logs. Error report saved to: {error_report_path}")
        else:
            logger.info("No errors found in logs.")
            combined_results["log_errors"] = {"count": 0, "files": 0}
        
        # Make sure to return in the format the cli.py expects
        return {
            "status": "success",
            "results": combined_results,
            "output_dir": output_dir
        }
    except Exception as e:
        logger.error(f"Complete workflow failed with exception: {str(e)}")
        if verbose:
            logger.exception("Full error traceback:")
        combined_results["status"] = "error"
        combined_results["message"] = str(e)
        logger.info(f"Complete workflow finished. Results saved to: {output_dir}")
        
        # Reset logging level if it was changed
        if verbose:
            logging_system.set_level('INFO', 'workflows')
        
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
            # Generate error report and save to file
            error_report_path = os.path.join(output_dir, "error_report.txt")
            print_error_report(error_logs, error_report_path)
            logger.warning(f"Found errors in logs. Error report saved to: {error_report_path}")
        
        # Make sure to return in the format the cli.py expects
        return {
            "status": "error",
            "message": str(e),
            "output_dir": output_dir
        } 