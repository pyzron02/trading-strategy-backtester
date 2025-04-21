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
    setup_output_dir_logging, print_workflow_log
)

# Import individual workflow modules
from workflows.simple_workflow import run_simple_workflow, ensure_data_available
from workflows.optimization_workflow import run_optimization_workflow
from workflows.monte_carlo_workflow import run_monte_carlo_workflow

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
    
    # Set up subdirectories for each part of the workflow
    simple_dir = os.path.join(output_dir, "01_simple_backtest")
    optimization_dir = os.path.join(output_dir, "02_optimization")
    montecarlo_dir = os.path.join(output_dir, "03_monte_carlo")
    
    # Create subdirectories
    os.makedirs(simple_dir, exist_ok=True)
    os.makedirs(optimization_dir, exist_ok=True)
    os.makedirs(montecarlo_dir, exist_ok=True)
    
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

        # Step 1: Run optimization to find the best parameters
        print_section("Step 1: Parameter Optimization")
        
        # Update progress for optimization step
        if progress_file and os.path.exists(progress_file):
            try:
                import json
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                progress_data.update({
                    'current_step': "Step 1: Parameter Optimization",
                    'progress': 10,
                    'current_step_progress': 0,
                    'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=4)
            except Exception as e:
                logger.warning(f"Error updating progress file: {e}")
                
        optimization_output_dir = os.path.join(output_dir, "1_optimization")
        if not os.path.exists(optimization_output_dir):
            os.makedirs(optimization_output_dir)
            
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
            "plot": plot
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
    except Exception as e:
        logger.error(f"Optimization failed with exception: {str(e)}")
        if verbose:
            logger.exception("Full error traceback:")
        combined_results["optimization"] = {"status": "error", "message": str(e)}
        # Fall back to simple backtest parameters if available
        best_params = combined_results.get("simple_backtest", {}).get("parameters", {})
        combined_results["status"] = "partial"
    
    # Step 2: Run backtest with optimized parameters
    print_section("Step 2: Backtesting with Optimized Parameters")
    
    # Update progress for backtest step
    if progress_file and os.path.exists(progress_file):
        try:
            import json
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            progress_data.update({
                'current_step': "Step 2: Backtesting with Optimized Parameters",
                'progress': 40,
                'current_step_progress': 0,
                'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=4)
        except Exception as e:
            logger.warning(f"Error updating progress file: {e}")
    
    backtest_output_dir = os.path.join(output_dir, "2_backtest")
    if not os.path.exists(backtest_output_dir):
        os.makedirs(backtest_output_dir)
    
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
    
    # Prepare backtest kwargs
    backtest_kwargs = {
        "strategy": strategy_name,
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "output_dir": backtest_output_dir,
        "parameters": adapted_best_params,
        "plot": plot,
        "verbose": verbose,
        "initial_capital": initial_capital,
        "commission": commission,
        "data_dir": data_dir
    }
    
    backtest_result = run_simple_workflow(**backtest_kwargs)
    
    # Store backtest results in combined_results
    if backtest_result and backtest_result.get("status") == "success":
        # Debug the structure of backtest_result
        logger.debug(f"Backtest result keys: {backtest_result.keys() if isinstance(backtest_result, dict) else 'Not a dict'}")
        
        # Try to extract the metrics from the right location
        backtest_data = {}
        
        # First look for results key
        if "results" in backtest_result:
            backtest_data = backtest_result["results"]
            logger.debug(f"Found results key with sub-keys: {backtest_data.keys() if isinstance(backtest_data, dict) else 'Not a dict'}")
        
        # If no results found or results is empty, look for equity_curve or metrics directly
        if not backtest_data and "metrics" in backtest_result:
            backtest_data = {"metrics": backtest_result["metrics"]}
            logger.debug("Found metrics directly in backtest_result")
            
        if "equity_curve" in backtest_result:
            if not backtest_data:
                backtest_data = {}
            backtest_data["equity_curve"] = backtest_result["equity_curve"]
            logger.debug("Found equity_curve directly in backtest_result")
        
        combined_results["simple_backtest"] = backtest_data
        logger.info("Simple backtest completed successfully")
    else:
        logger.warning(f"Simple backtest failed: {backtest_result.get('message', 'Unknown error')}")
        combined_results["simple_backtest"] = {"status": "error", "message": backtest_result.get('message', 'Unknown error')}
    
    # Step 3: Run Monte Carlo simulation
    print_section("Step 3: Monte Carlo Simulation")
    
    # Update progress for Monte Carlo step
    if progress_file and os.path.exists(progress_file):
        try:
            import json
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            progress_data.update({
                'current_step': "Step 3: Monte Carlo Simulation",
                'progress': 60,
                'current_step_progress': 0,
                'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=4)
        except Exception as e:
            logger.warning(f"Error updating progress file: {e}")
    
    monte_carlo_output_dir = os.path.join(output_dir, "3_monte_carlo")
    if not os.path.exists(monte_carlo_output_dir):
        os.makedirs(monte_carlo_output_dir)
    
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
        "enhanced_plots": enhanced_plots
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
                            f.write(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n")
                        
                        # Add advanced metrics if available
                        if 'annual_return' in metrics:
                            f.write(f"Annual Return: {metrics.get('annual_return', 0):.2%}\n")
                        
                        if 'alpha' in metrics:
                            f.write(f"Alpha: {metrics.get('alpha', 0):.2%}\n")
                else:
                    f.write("\nNo metrics available for optimized parameters\n")
            else:
                f.write("Optimization failed or no best parameters found\n")
        
        # Monte Carlo Results
        if "monte_carlo" in combined_results:
            f.write("\n" + "=" * 80 + "\n")
            f.write("MONTE CARLO SIMULATION RESULTS\n")
            f.write("=" * 80 + "\n")
            
            # Try to get monte carlo results with fallbacks for different possible structures
            monte_carlo_data = combined_results.get("monte_carlo", {})
            monte_carlo_results = {}
            
            # First try direct access
            if isinstance(monte_carlo_data, dict):
                # Try different possible keys where results might be stored
                if "monte_carlo_results" in monte_carlo_data:
                    monte_carlo_results = monte_carlo_data["monte_carlo_results"]
                elif "results" in monte_carlo_data:
                    monte_carlo_results = monte_carlo_data["results"]
                else:
                    # If no nested results, maybe the data is directly in monte_carlo_data
                    # Check for expected Monte Carlo fields
                    if any(key in monte_carlo_data for key in ['mean_return', 'probability_of_profit', 'initial_equity']):
                        monte_carlo_results = monte_carlo_data
            
            logger.debug(f"Monte Carlo results found: {bool(monte_carlo_results)}")
            
            if monte_carlo_results:
                # Summary stats
                f.write(f"\nNumber of Simulations: {monte_carlo_results.get('num_simulations', n_simulations)}\n")
                f.write(f"Confidence Level: {monte_carlo_results.get('confidence_level', 0.95):.0%}\n\n")
                
                # Basic statistics
                f.write("----- Portfolio Statistics -----\n")
                f.write(f"Initial Equity: ${monte_carlo_results.get('initial_equity', initial_capital):.2f}\n")
                f.write(f"Final Equity (Original): ${monte_carlo_results.get('final_equity_original', initial_capital):.2f}\n")
                f.write(f"Original Return: {monte_carlo_results.get('return_original', 0.0):.2%}\n\n")
                
                # Simulation results
                f.write("----- Simulation Results -----\n")
                f.write(f"Mean Final Equity: ${monte_carlo_results.get('mean_final_equity', initial_capital):.2f}\n")
                f.write(f"Median Final Equity: ${monte_carlo_results.get('median_final_equity', initial_capital):.2f}\n")
                f.write(f"Mean Return: {monte_carlo_results.get('mean_return', 0.0):.2%}\n\n")
                
                # Confidence intervals
                confidence_level = monte_carlo_results.get('confidence_level', 0.95)
                f.write(f"----- Confidence Intervals ({confidence_level:.0%}) -----\n")
                f.write(f"Final Equity Range: ${monte_carlo_results.get('ci_lower_final_equity', 0.0):.2f} to " +
                        f"${monte_carlo_results.get('ci_upper_final_equity', 0.0):.2f}\n")
                f.write(f"Return Range: {monte_carlo_results.get('ci_lower_return', 0.0):.2%} to " +
                        f"{monte_carlo_results.get('ci_upper_return', 0.0):.2%}\n\n")
                
                # Risk metrics
                f.write("----- Risk Metrics -----\n")
                f.write(f"Value at Risk (VaR {confidence_level:.0%}): " +
                        f"{monte_carlo_results.get('var_pct', 0.0):.2%}\n")
                f.write(f"Conditional VaR (CVaR {confidence_level:.0%}): " +
                        f"{monte_carlo_results.get('cvar_pct', 0.0):.2%}\n")
                f.write(f"Worst Return: {monte_carlo_results.get('worst_return', 0.0):.2%}\n")
                f.write(f"Best Return: {monte_carlo_results.get('best_return', 0.0):.2%}\n\n")
                
                # Probability metrics
                f.write("----- Probability Metrics -----\n")
                f.write(f"Probability of Profit: {monte_carlo_results.get('probability_of_profit', 0.0):.2%}\n")
            else:
                f.write("Monte Carlo simulation failed or no results available\n")
                # Log debug info about the combined_results structure
                logger.debug(f"Monte Carlo data structure: {monte_carlo_data.keys() if isinstance(monte_carlo_data, dict) else 'Not a dict'}")
        else:
            # No monte carlo section found
            logger.debug(f"Monte Carlo data missing. Available keys: {combined_results.keys()}")
        
        # Overall workflow status
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"WORKFLOW STATUS: {combined_results['status'].upper()}\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Complete workflow finished. Results saved to: {output_dir}")
    
    # Reset logging level if it was changed
    if verbose:
        logging_system.set_level('INFO', 'workflows')
    
    # Clean up temporary files
    for temp_file in _temp_files_to_cleanup:
        try:
            os.remove(temp_file)
            logger.info(f"Cleaned up temporary file: {temp_file}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary file: {str(e)}")
    
    # Make sure to return in the format the cli.py expects
    return {
        "status": "success",
        "results": combined_results,
        "output_dir": output_dir
    } 