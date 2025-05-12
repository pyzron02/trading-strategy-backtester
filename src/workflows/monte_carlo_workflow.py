#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo workflow module.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import uuid
import seaborn as sns
from dateutil.relativedelta import relativedelta
import logging
import random

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
    time_execution, find_strategy_param_file, logger, logging_system,
    print_workflow_log, adapt_strategy_parameters,
    save_results_summary, setup_output_dir_logging, remove_output_dir_logging,
    check_logs_for_errors, print_error_report
)

# Import engine components
from engine.run_backtest import run_backtest
from monte_carlo.monte_carlo_analysis import MonteCarloAnalysis
from monte_carlo.visualizations import MonteCarloVisualizer
from workflows.simple_workflow import ensure_data_available, convert_grid_to_single_values

def calculate_trading_days(start_date_str: str, end_date_str: str) -> int:
    """
    Calculate the number of trading days between two dates.
    
    Args:
        start_date_str: Start date in 'YYYY-MM-DD' format
        end_date_str: End date in 'YYYY-MM-DD' format
        
    Returns:
        int: Estimated number of trading days (roughly 252 per year)
    """
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        
        # Calculate the difference in days
        delta = end_date - start_date
        total_days = delta.days
        
        # Ensure total_days is at least 1 to avoid division by zero
        if total_days <= 0:
            logger.warning(f"Start date {start_date} is on or after end date {end_date}. Setting to 1 trading day.")
            return 1
        
        # Estimate trading days (roughly 252 trading days per year)
        years = total_days / 365.25
        trading_days = int(years * 252)
        
        # Ensure at least 1 trading day
        return max(1, trading_days)
    except Exception as e:
        logger.warning(f"Error calculating trading days: {e}")
        # Default to 252 (1 year) if calculation fails
        return 252

@time_execution("monte carlo workflow")
def run_monte_carlo_workflow(
    strategy=None,  # New parameter to support unified_workflow
    strategy_name=None,  # Original parameter
    tickers=None,
    start_date=None,
    end_date=None,
    output_dir=None,
    n_simulations=100,
    parameters=None,
    param_file=None,
    keep_permuted_data=False,
    verbose=False,
    initial_capital=100000.0,
    commission=0.001,
    data_dir="input",
    analyze_only=False,
    backtest_result=None,
    confidence_level=0.95,
    bootstrap_pct=0.5,
    random_seed=None,
    plot=False,
    enhanced_plots=False,
    workflow_type="monte_carlo",  # Add workflow_type parameter
    _temp_files_to_cleanup=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a Monte Carlo simulation workflow for the given strategy.
    
    Args:
        strategy: Name of the strategy to use (alternative to strategy_name)
        strategy_name: Name of the strategy to use
        tickers: List of ticker symbols
        start_date: Start date for backtest
        end_date: End date for backtest
        output_dir: Directory for output files
        n_simulations: Number of Monte Carlo simulations to run
        parameters: Dictionary of strategy parameters (overrides param_file)
        param_file: File with parameter definitions
        keep_permuted_data: Whether to keep permuted data files
        verbose: Whether to print detailed logs
        initial_capital: Initial capital for backtest
        commission: Commission per trade
        data_dir: Directory with data files
        analyze_only: Whether to only run analysis on existing backtest
        backtest_result: Existing backtest result to analyze
        confidence_level: Confidence level for Monte Carlo analysis
        bootstrap_pct: Percentage of data to bootstrap
        random_seed: Random seed for reproducibility
        plot: Whether to plot results
        enhanced_plots: Whether to create enhanced plots
        workflow_type: Type of workflow
        _temp_files_to_cleanup: List of temporary files to clean up
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with Monte Carlo simulation results
    """
    # Validate date inputs to prevent division by zero issues
    try:
        if start_date and end_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
            
            # Check if dates make sense
            if start_date_obj >= end_date_obj:
                error_msg = f"Start date {start_date} must be before end date {end_date}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "message": error_msg
                }
            
            # Ensure reasonable date range - at least 5 days difference
            min_days = 5
            if (end_date_obj - start_date_obj).days < min_days:
                error_msg = f"Date range too small ({(end_date_obj - start_date_obj).days} days). Need at least {min_days} days for Monte Carlo analysis."
                logger.error(error_msg)
                return {
                    "status": "error",
                    "message": error_msg
                }
    except Exception as e:
        error_msg = f"Error validating dates: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg
        }
    
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
        output_dir = os.path.join(project_root, "output", f"{strategy_name}_monte_carlo_{timestamp}_{run_id}")
        logger.info(f"Creating unique output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging for this workflow
    setup_output_dir_logging(output_dir, strategy_name, "monte_carlo")
    
    # Set logging level based on verbose flag
    if verbose:
        logging_system.set_level('DEBUG', 'workflows')
    
    # If parameters not provided, load from file
    if not parameters:
        if not param_file:
            param_file = find_strategy_param_file(strategy_name)
            if not param_file:
                error_msg = f"No parameter file found for strategy: {strategy_name}"
                logger.error(error_msg)
                
                # Log workflow failure
                print_workflow_log(
                    workflow_name="Monte Carlo Workflow",
                    strategy_name=strategy_name,
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date,
                    status="FAILED",
                    additional_info={"error": error_msg}
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
                    "message": error_msg,
                    "output_dir": output_dir
                }
        
        try:
            with open(param_file, 'r') as f:
                parameters = json.load(f)
        except Exception as e:
            error_msg = f"Error loading parameter file: {str(e)}"
            logger.error(error_msg)
            
            # Log workflow failure
            print_workflow_log(
                workflow_name="Monte Carlo Workflow",
                strategy_name=strategy_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                status="FAILED",
                additional_info={"error": error_msg}
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
                "message": error_msg,
                "output_dir": output_dir
            }
    
    # Print parameters
    print_section("Strategy Parameters")
    print_parameters(parameters)
    
    # Check for parameter grids and convert to single values if needed
    has_grid = any(isinstance(v, list) for v in parameters.values())
    if has_grid:
        logger.info("Parameter grid detected in Monte Carlo workflow. Converting to single values.")
        original_params = parameters.copy()
        parameters = convert_grid_to_single_values(parameters)
        logger.info("Converted parameters from grid to single values.")
        print_parameters(parameters)
    
    try:
        # Check if stock_data.csv exists but don't regenerate it
        stock_data_path = os.path.join(data_dir, "stock_data.csv")
        if os.path.exists(stock_data_path):
            stock_csv = stock_data_path
            logger.info(f"Using existing stock data from: {stock_csv}")
        else:
            error_msg = f"Stock data file not found at: {stock_data_path}. Please run data_setup.py first."
            logger.error(error_msg)
            
            # Log workflow failure
            print_workflow_log(
                workflow_name="Monte Carlo Workflow",
                strategy_name=strategy_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                status="FAILED",
                additional_info={"error": error_msg}
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
                "message": error_msg,
                "output_dir": output_dir
            }
        
        # Log workflow header
        print_header(f"Running Initial Backtest for Monte Carlo Analysis")
        
        # Adapt parameters for strategy if needed
        adapted_parameters = adapt_strategy_parameters(strategy_name, parameters)
        
        # Add validation for AuctionMarket strategy parameters to prevent division by zero
        if strategy_name == "AuctionMarket":
            # Check if adaptations needed at top level
            if not adapted_parameters.get("auction_zones") and parameters.get("auction_zones"):
                adapted_parameters["auction_zones"] = parameters["auction_zones"].copy()
            
            # Ensure balance_threshold is valid if it exists
            if adapted_parameters.get("auction_zones") and "balance_threshold" in adapted_parameters["auction_zones"]:
                if adapted_parameters["auction_zones"]["balance_threshold"] <= 0:
                    logger.warning("balance_threshold must be greater than 0. Setting to default value 0.5")
                    adapted_parameters["auction_zones"]["balance_threshold"] = 0.5
        
        logger.info(f"Adapted parameters for {strategy_name} strategy")
        
        # Run backtest
        backtest_result = run_backtest(
            strategy_name=strategy_name,
            parameters=adapted_parameters,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            commission=commission,
            data_dir=data_dir,
            output_dir=output_dir,
            verbose=verbose,  # Add verbose parameter back
            plot=False       # Don't generate plots here, we'll do it in the Monte Carlo analysis
        )
        
        if backtest_result is None or (isinstance(backtest_result, dict) and backtest_result.get("status") != "success"):
            error_msg = f"Backtest failed: {backtest_result.get('message', 'Unknown error') if isinstance(backtest_result, dict) else 'No results'}"
            logger.error(error_msg)
            
            # Create error summary file
            try:
                summary_file = os.path.join(output_dir, "monte_carlo_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write("=" * 80 + "\n")
                    f.write("MONTE CARLO SIMULATION RESULTS\n")
                    f.write("=" * 80 + "\n\n")
                    
                    # Basic test information
                    f.write(f"Strategy: {strategy_name}\n")
                    f.write(f"Period: {start_date} to {end_date}\n")
                    f.write(f"Tickers: {', '.join(tickers)}\n")
                    f.write(f"Initial Capital: ${initial_capital:.2f}\n")
                    f.write(f"Commission Rate: {commission:.4f}\n\n")
                    
                    # Error information
                    f.write("-" * 80 + "\n")
                    f.write("Error Information:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Error: {error_msg}\n\n")
                    
                    # End of summary
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("WORKFLOW STATUS: ERROR\n")
                    f.write("=" * 80 + "\n")
                
                logger.info(f"Monte Carlo error summary saved to: {summary_file}")
            except Exception as summary_err:
                logger.error(f"Failed to create error summary file: {str(summary_err)}")
            
            # Log workflow failure
            print_workflow_log(
                workflow_name="Monte Carlo Workflow",
                strategy_name=strategy_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                status="FAILED",
                additional_info={"error": error_msg}
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
                "message": error_msg,
                "output_dir": output_dir
            }
        
        # Print metrics
        print_section("Backtest Results")
        if isinstance(backtest_result, dict) and 'metrics' in backtest_result:
            print_metrics(backtest_result.get('metrics', {}))
        
        # Extract equity curve data
        try:
            # If backtest_result provided or from previous backtest, extract equity curve
            if backtest_result:
                # Try to extract equity curve from results
                if "equity_curve" in backtest_result:
                    backtest_data = backtest_result["equity_curve"]
                else:
                    # Try to load equity curve from file
                    equity_curve_file = os.path.join(output_dir, "backtest", "equity_curve.csv")
                    if os.path.exists(equity_curve_file):
                        backtest_data = pd.read_csv(equity_curve_file, index_col=0, parse_dates=True)
                    else:
                        error_msg = "No equity curve data found in backtest result or file"
                        logger.error(error_msg)
                        
                        # Log workflow failure and error handling
                        write_error_summary(output_dir, strategy_name, tickers, start_date, end_date, initial_capital, commission, error_msg)
                        
                        # Remove output dir handlers
                        remove_output_dir_logging()
                        
                        return {
                            "status": "error", 
                            "message": error_msg,
                            "output_dir": output_dir
                        }
            else:
                error_msg = "No backtest result available for Monte Carlo analysis"
                logger.error(error_msg)
                
                # Log workflow failure and error handling
                write_error_summary(output_dir, strategy_name, tickers, start_date, end_date, initial_capital, commission, error_msg)
                
                # Remove output dir handlers
                remove_output_dir_logging()
                
                return {
                    "status": "error", 
                    "message": error_msg,
                    "output_dir": output_dir
                }
            
            # Verify backtest data is valid for Monte Carlo analysis
            if backtest_data.empty:
                error_msg = "Equity curve is empty. Cannot perform Monte Carlo analysis."
                logger.error(error_msg)
                
                # Log workflow failure and error handling
                write_error_summary(output_dir, strategy_name, tickers, start_date, end_date, initial_capital, commission, error_msg)
                
                # Remove output dir handlers
                remove_output_dir_logging()
                
                return {
                    "status": "error", 
                    "message": error_msg,
                    "output_dir": output_dir
                }
                
            if len(backtest_data) < 5:  # Require at least 5 data points
                error_msg = f"Not enough data points for Monte Carlo analysis. Found {len(backtest_data)}, need at least 5."
                logger.error(error_msg)
                
                # Log workflow failure and error handling
                write_error_summary(output_dir, strategy_name, tickers, start_date, end_date, initial_capital, commission, error_msg)
                
                # Remove output dir handlers
                remove_output_dir_logging()
                
                return {
                    "status": "error", 
                    "message": error_msg,
                    "output_dir": output_dir
                }
                
            # Ensure initial equity is not zero (which would cause division by zero)
            initial_equity_value = backtest_data.iloc[0]
            if isinstance(initial_equity_value, pd.Series):
                initial_equity_value = initial_equity_value.iloc[0]
                
            if initial_equity_value == 0:
                logger.warning("Initial equity is zero. Setting to 0.01 to avoid division by zero.")
                if isinstance(backtest_data.iloc[0], pd.Series):
                    backtest_data.iloc[0] = 0.01
                else:
                    backtest_data.iloc[0] = 0.01
            
            # Run Monte Carlo analysis
            print_section("Running Monte Carlo Analysis")
            logger.info(f"Number of simulations: {n_simulations}")
            
            # Default values for missing parameters
            confidence_level = confidence_level or 0.95
            bootstrap_pct = bootstrap_pct or 0.5
            random_seed = random_seed or 42
            
            logger.info(f"Confidence level: {confidence_level}")
            logger.info(f"Bootstrap percentage: {bootstrap_pct}")
            
            try:
                mc_analyzer = MonteCarloAnalysis(
                    equity_curve=backtest_data,
                    num_simulations=n_simulations,
                    confidence_level=confidence_level,
                    random_seed=random_seed,
                    bootstrap_pct=bootstrap_pct
                )
            except Exception as e:
                error_msg = f"Error initializing Monte Carlo analysis: {str(e)}"
                logger.error(error_msg)
                if verbose:
                    logger.exception("Full error traceback:")
                
                # Log workflow failure and error handling
                write_error_summary(output_dir, strategy_name, tickers, start_date, end_date, initial_capital, commission, error_msg)
                
                # Remove output dir handlers
                remove_output_dir_logging()
                
                return {
                    "status": "error", 
                    "message": error_msg,
                    "output_dir": output_dir
                }
            
            # Check for progress file in workflow_config.json
            progress_file = None
            workflow_config_path = os.path.join(os.path.dirname(output_dir), 'workflow_config.json')
            if os.path.exists(workflow_config_path):
                try:
                    with open(workflow_config_path, 'r') as f:
                        workflow_config = json.load(f)
                    if 'frontend' in workflow_config and 'progress_file' in workflow_config['frontend']:
                        progress_file = workflow_config['frontend']['progress_file']
                        logger.info(f"Found progress file for frontend updates: {progress_file}")
                except Exception as e:
                    logger.warning(f"Error reading workflow config for progress updates: {e}")
            
            # Run the analysis with error handling
            try:
                mc_results = mc_analyzer.run(progress_file)
            except Exception as e:
                error_msg = f"Error running Monte Carlo simulations: {str(e)}"
                logger.error(error_msg)
                if verbose:
                    logger.exception("Full error traceback:")
                
                # Log workflow failure and error handling
                write_error_summary(output_dir, strategy_name, tickers, start_date, end_date, initial_capital, commission, error_msg)
                
                # Remove output dir handlers
                remove_output_dir_logging()
                
                return {
                    "status": "error", 
                    "message": error_msg,
                    "output_dir": output_dir
                }
            
            if not mc_results:
                error_msg = "Monte Carlo analysis failed to produce results"
                logger.error(error_msg)
                
                # Log workflow failure and error handling
                write_error_summary(output_dir, strategy_name, tickers, start_date, end_date, initial_capital, commission, error_msg)
                
                # Remove output dir handlers
                remove_output_dir_logging()
                
                return {
                    "status": "error", 
                    "message": error_msg,
                    "output_dir": output_dir
                }
            
            # Validate key Monte Carlo results to catch potential NaN or division by zero issues
            key_metrics = ['initial_equity', 'final_equity_original', 'return_original', 
                          'mean_final_equity', 'mean_return', 'probability_of_profit']
            
            for metric in key_metrics:
                if metric not in mc_results or pd.isna(mc_results[metric]):
                    error_msg = f"Monte Carlo analysis produced invalid results for metric: {metric}"
                    logger.error(error_msg)
                    
                    # Log workflow failure and error handling
                    write_error_summary(output_dir, strategy_name, tickers, start_date, end_date, initial_capital, commission, error_msg)
                    
                    # Remove output dir handlers
                    remove_output_dir_logging()
                    
                    return {
                        "status": "error", 
                        "message": error_msg,
                        "output_dir": output_dir
                    }
            
            # Generate plots only if plot flag is enabled
            equity_values = mc_analyzer.equity_values
            plot_files = {}
            
            logger.debug(f"Plot flag is set to: {plot}, enhanced_plots flag is set to: {enhanced_plots}")
            
            # Only proceed with plotting if there's price movement
            if (len(equity_values) > 1) and (equity_values.iloc[0] != equity_values.iloc[-1]):
                # Basic plot using the MonteCarloAnalysis class
                if plot:
                    plot_file = os.path.join(output_dir, f"{strategy_name}_monte_carlo.png")
                    logger.debug(f"Generating basic Monte Carlo plot with {len(equity_values)} data points")
                    mc_analyzer.plot(save_path=plot_file)
                    plot_files['basic'] = plot_file
                    logger.info(f"Basic Monte Carlo plot saved to: {plot_file}")
                
                # Enhanced plots using the MonteCarloVisualizer class
                if enhanced_plots:
                    logger.debug(f"Generating enhanced Monte Carlo visualizations")
                    visualizer = MonteCarloVisualizer(
                        equity_values=equity_values,
                        simulated_paths=mc_analyzer.simulated_paths,
                        simulation_results=mc_results,
                        confidence_level=confidence_level,
                        output_dir=output_dir,
                        strategy_name=strategy_name,
                        workflow_type=workflow_type  # Pass the workflow_type parameter
                    )
                    
                    # Create all available plot types
                    enhanced_plot_files = visualizer.create_all_plots(save=True)
                    
                    # Merge with plot_files
                    plot_files.update(enhanced_plot_files)
                    
                    logger.info(f"Enhanced visualizations saved to: {output_dir}")
                    for plot_type, file_path in enhanced_plot_files.items():
                        if file_path:
                            logger.debug(f"- {plot_type}: {os.path.basename(file_path)}")
            else:
                logger.debug(f"Skipping plot generation: plot={plot}, data_points={len(equity_values) if equity_values is not None else 0}, price_movement={equity_values.iloc[0] != equity_values.iloc[-1] if equity_values is not None and len(equity_values) > 1 else False}")
            
            # Extract and display results
            print_section("Monte Carlo Results")
            
            # Display key metrics
            logger.info(f"Initial equity: ${mc_results['initial_equity']:.2f}")
            logger.info(f"Final equity (original): ${mc_results['final_equity_original']:.2f}")
            logger.info(f"Return (original): {mc_results['return_original']:.2%}")
            
            logger.info(f"\nMean final equity: ${mc_results['mean_final_equity']:.2f}")
            logger.info(f"Median final equity: ${mc_results['median_final_equity']:.2f}")
            logger.info(f"Mean return: {mc_results['mean_return']:.2%}")
            
            logger.info(f"\nConfidence interval ({confidence_level*100:.0f}%):")
            logger.info(f"  Lower bound (final equity): ${mc_results['ci_lower_final_equity']:.2f}")
            logger.info(f"  Upper bound (final equity): ${mc_results['ci_upper_final_equity']:.2f}")
            logger.info(f"  Lower bound (return): {mc_results['ci_lower_return']:.2%}")
            logger.info(f"  Upper bound (return): {mc_results['ci_upper_return']:.2%}")
            
            logger.info(f"\nVaR ({confidence_level*100:.0f}%): {mc_results['var_pct']:.2%}")
            logger.info(f"CVaR ({confidence_level*100:.0f}%): {mc_results['cvar_pct']:.2%}")
            
            logger.info(f"\nWorst return: {mc_results['worst_return']:.2%}")
            logger.info(f"Best return: {mc_results['best_return']:.2%}")
            
            logger.info(f"\nProbability of profit: {mc_results['probability_of_profit']:.2%}")
            
            # Save results to file
            results_file = os.path.join(output_dir, f"{strategy_name}_monte_carlo_results.json")
            with open(results_file, 'w') as f:
                json.dump(mc_results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float64) else x)
            
            logger.info(f"\nDetailed results saved to: {results_file}")
            
            # Generate summary file in the same format as complete_workflow_summary.txt
            summary_file = os.path.join(output_dir, "monte_carlo_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("MONTE CARLO SIMULATION RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                # Basic test information
                f.write(f"Strategy: {strategy_name}\n")
                f.write(f"Period: {start_date} to {end_date}\n")
                f.write(f"Tickers: {', '.join(tickers)}\n")
                f.write(f"Initial Capital: ${initial_capital:.2f}\n")
                f.write(f"Commission Rate: {commission:.4f}\n\n")
                
                # Monte Carlo Results
                f.write("=" * 80 + "\n")
                f.write("SIMULATION PARAMETERS\n")
                f.write("=" * 80 + "\n\n")
                
                # Monte Carlo parameters
                f.write(f"Number of Simulations: {n_simulations}\n")
                f.write(f"Confidence Level: {int(confidence_level * 100)}%\n\n")
                
                # Portfolio Statistics section
                f.write("-" * 80 + "\n")
                f.write("Portfolio Statistics:\n")
                f.write("-" * 80 + "\n")
                f.write(f"initial_value: {initial_capital:.4f}\n")
                f.write(f"final_value_original: {mc_results['final_equity_original']:.4f}\n")
                f.write(f"original_return: {mc_results['return_original']:.4f}\n")
                f.write(f"original_return_pct: {mc_results['return_original'] * 100:.4f}\n")
                
                # Simulation Results section
                f.write("\n" + "-" * 80 + "\n")
                f.write("Simulation Results:\n")
                f.write("-" * 80 + "\n")
                f.write(f"mean_final_value: {mc_results['mean_final_equity']:.4f}\n")
                f.write(f"median_final_value: {mc_results['median_final_equity']:.4f}\n")
                f.write(f"mean_return: {mc_results['mean_return']:.4f}\n")
                f.write(f"mean_return_pct: {mc_results['mean_return'] * 100:.4f}\n")
                
                # Confidence Intervals section
                f.write("\n" + "-" * 80 + "\n")
                f.write(f"Confidence Intervals ({int(confidence_level * 100)}%):\n")
                f.write("-" * 80 + "\n")
                f.write(f"final_equity_min: {mc_results['ci_lower_final_equity']:.4f}\n")
                f.write(f"final_equity_max: {mc_results['ci_upper_final_equity']:.4f}\n")
                f.write(f"return_min: {mc_results['ci_lower_return']:.4f}\n")
                f.write(f"return_min_pct: {mc_results['ci_lower_return'] * 100:.4f}\n")
                f.write(f"return_max: {mc_results['ci_upper_return']:.4f}\n")
                f.write(f"return_max_pct: {mc_results['ci_upper_return'] * 100:.4f}\n")
                
                # Risk Metrics section
                f.write("\n" + "-" * 80 + "\n")
                f.write("Risk Metrics:\n")
                f.write("-" * 80 + "\n")
                f.write(f"var_95: {mc_results['var_pct']:.4f}\n")
                f.write(f"var_95_pct: {mc_results['var_pct'] * 100:.4f}\n")
                f.write(f"cvar_95: {mc_results['cvar_pct']:.4f}\n")
                f.write(f"cvar_95_pct: {mc_results['cvar_pct'] * 100:.4f}\n")
                f.write(f"worst_return: {mc_results['worst_return']:.4f}\n")
                f.write(f"worst_return_pct: {mc_results['worst_return'] * 100:.4f}\n")
                f.write(f"best_return: {mc_results['best_return']:.4f}\n")
                f.write(f"best_return_pct: {mc_results['best_return'] * 100:.4f}\n")
                
                # Probability Metrics section
                f.write("\n" + "-" * 80 + "\n")
                f.write("Probability Metrics:\n")
                f.write("-" * 80 + "\n")
                f.write(f"profit_probability: {mc_results['probability_of_profit']:.4f}\n")
                f.write(f"profit_probability_pct: {mc_results['probability_of_profit'] * 100:.4f}\n")
                
                # End of summary
                f.write("\n" + "=" * 80 + "\n")
                f.write("WORKFLOW STATUS: SUCCESS\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"Monte Carlo summary saved to: {summary_file}")
            
            # Create a combined result
            workflow_result = {
                "status": "success",
                "strategy_name": strategy_name,
                "dates": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "parameters": parameters,
                "backtest_metrics": backtest_result.get("metrics", {}),
                "monte_carlo_results": mc_results,
                "equity_curve": backtest_data,  # Add this to make complete_workflow.py happy
                "output_dir": output_dir,
                "plot_files": plot_files
            }
            
        except Exception as e:
            error_msg = f"Monte Carlo workflow failed: {str(e)}"
            logger.error(error_msg)
            if verbose:
                logger.exception("Full error traceback:")
            
            # Log workflow failure and error handling
            write_error_summary(output_dir, strategy_name, tickers, start_date, end_date, initial_capital, commission, error_msg)
            
            # Remove output dir handlers
            remove_output_dir_logging()
            
            return {
                "status": "error",
                "message": error_msg,
                "output_dir": output_dir
            }
        
    except Exception as e:
        error_msg = f"Monte Carlo workflow failed: {str(e)}"
        logger.error(error_msg)
        if verbose:
            logger.exception("Full error traceback:")
        
        # Log workflow failure and error handling
        write_error_summary(output_dir, strategy_name, tickers, start_date, end_date, initial_capital, commission, error_msg)
        
        # Remove output dir handlers
        remove_output_dir_logging()
        
        return {
            "status": "error",
            "message": error_msg,
            "output_dir": output_dir
        }
    
    # Reset logging level if it was changed
    if verbose:
        logging_system.set_level('INFO', 'workflows')
    
    # Log workflow completion
    completion_info = {
        "n_simulations": n_simulations,
        "mean_return": f"{mc_results['mean_return']:.2%}",
        "probability_of_profit": f"{mc_results['probability_of_profit']:.2%}",
        "output_dir": output_dir
    }
    print_workflow_log(
        workflow_name="Monte Carlo Workflow",
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
        workflow_result["log_errors"] = {
            "count": sum(len(errors) for errors in error_logs.values()),
            "files": len(error_logs)
        }
        
        # Generate error report and save to file
        error_report_path = os.path.join(output_dir, "error_report.txt")
        print_error_report(error_logs, error_report_path)
        logger.warning(f"Found errors in logs. Error report saved to: {error_report_path}")
    else:
        logger.info("No errors found in logs.")
        workflow_result["log_errors"] = {"count": 0, "files": 0}
    
    # Remove output dir handlers
    remove_output_dir_logging()
    
    return workflow_result

def write_error_summary(output_dir, strategy_name, tickers, start_date, end_date, initial_capital, commission, error_msg):
    """Helper function to write error summary and check logs for errors."""
    # Try to use the enhanced error reporting module if available
    try:
        from utils.error_reporting import generate_error_summary
        
        # Generate comprehensive error summary
        summary_file = generate_error_summary(
            output_dir=output_dir,
            workflow_type="monte_carlo",
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            tickers=tickers,
            error_msg=error_msg
        )
        
        logger.info(f"Error summary saved to: {summary_file}")
        
    except ImportError:
        # Fall back to basic error summary if the module isn't available
        logger.warning("Enhanced error reporting module not available. Using basic error summary.")
        
        # Create error summary file
        try:
            summary_file = os.path.join(output_dir, "monte_carlo_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("MONTE CARLO SIMULATION RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                # Basic test information
                f.write(f"Strategy: {strategy_name}\n")
                if isinstance(tickers, list):
                    f.write(f"Tickers: {', '.join(tickers)}\n")
                else:
                    f.write(f"Tickers: {tickers}\n")
                f.write(f"Period: {start_date} to {end_date}\n")
                f.write(f"Initial Capital: ${initial_capital:.2f}\n")
                f.write(f"Commission Rate: {commission:.4f}\n\n")
                
                # Error information
                f.write("-" * 80 + "\n")
                f.write("Error Information:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Error: {error_msg}\n\n")
                
                # End of summary
                f.write("\n" + "=" * 80 + "\n")
                f.write("WORKFLOW STATUS: ERROR\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"Monte Carlo error summary saved to: {summary_file}")
        except Exception as summary_err:
            logger.error(f"Failed to create error summary file: {str(summary_err)}")
    
    # Log workflow failure
    print_workflow_log(
        workflow_name="Monte Carlo Workflow",
        strategy_name=strategy_name,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        status="FAILED",
        additional_info={"error": error_msg}
    )
    
    # Check logs for errors
    logger.info("Checking logs for errors...")
    error_logs = check_logs_for_errors(output_dir)
    
    if error_logs:
        # Generate error report and save to file
        error_report_path = os.path.join(output_dir, "error_report.txt")
        print_error_report(error_logs, error_report_path)
        logger.warning(f"Found errors in logs. Error report saved to: {error_report_path}")
    
    return 