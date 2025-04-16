#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete workflow module that combines multiple workflow types.
"""
import os
import sys
import json
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
    logger, logging_system
)

# Import individual workflow modules
from workflows.simple_workflow import run_simple_workflow, ensure_data_available
from workflows.optimization_workflow import run_optimization_workflow
from workflows.monte_carlo_workflow import run_monte_carlo_workflow

@time_execution("complete workflow")
def run_complete_workflow(
    strategy_name: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_dir: str,
    parameters: Optional[Dict[str, Any]] = None,
    param_file: Optional[str] = None,
    plot: bool = True,
    n_trials: int = 50,
    n_simulations: int = 100,
    optimization_metric: str = "sharpe_ratio",
    keep_permuted_data: bool = False,
    verbose: bool = False,
    initial_capital: float = 100000.0,
    commission: float = 0.001,
    data_dir: str = "input"
) -> Dict[str, Any]:
    """
    Run a complete workflow including backtest, optimization, and Monte Carlo simulation.
    
    Args:
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
    
    Returns:
        Dict containing the results from all workflow steps
    """
    print_header(f"Complete Workflow: {strategy_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up subdirectories for each part of the workflow
    simple_dir = os.path.join(output_dir, "01_simple_backtest")
    optimization_dir = os.path.join(output_dir, "02_optimization")
    montecarlo_dir = os.path.join(output_dir, "03_monte_carlo")
    
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
        # Ensure stock_data.csv is available and contains required tickers
        stock_csv = ensure_data_available(tickers, start_date, end_date, data_dir)
        logger.info(f"Using stock data from: {stock_csv}")
        
        # Step 1: Run optimization to find the best parameters
        print_section("Step 1: Parameter Optimization")
        
        optimization_output_dir = os.path.join(output_dir, "1_optimization")
        if not os.path.exists(optimization_output_dir):
            os.makedirs(optimization_output_dir)
            
        # Prepare optimization kwargs
        optimization_kwargs = {
            "strategy_name": strategy_name,
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "output_dir": optimization_output_dir,
            "parameters": parameters,
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
    
    # Prepare backtest kwargs
    backtest_kwargs = {
        "strategy_name": strategy_name,
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "output_dir": backtest_output_dir,
        "parameters": best_params,
        "plot": plot,
        "verbose": verbose,
        "initial_capital": initial_capital,
        "commission": commission,
        "data_dir": data_dir
    }
    
    backtest_result = run_simple_workflow(**backtest_kwargs)
    
    # Step 3: Run Monte Carlo simulation
    print_section("Step 3: Monte Carlo Simulation")
    
    monte_carlo_output_dir = os.path.join(output_dir, "3_monte_carlo")
    if not os.path.exists(monte_carlo_output_dir):
        os.makedirs(monte_carlo_output_dir)
    
    # Prepare monte carlo kwargs
    monte_carlo_kwargs = {
        "strategy_name": strategy_name,
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "output_dir": monte_carlo_output_dir,
        "parameters": best_params,
        "n_simulations": n_simulations,
        "keep_permuted_data": keep_permuted_data,
        "verbose": verbose,
        "initial_capital": initial_capital,
        "commission": commission,
        "data_dir": data_dir,
        "plot": plot
    }
    
    monte_carlo_result = run_monte_carlo_workflow(**monte_carlo_kwargs)
    
    if "equity_curve" in monte_carlo_result:
        combined_results["monte_carlo"] = monte_carlo_result.get("results", {})
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
        
        if "simple_backtest" in combined_results and isinstance(combined_results["simple_backtest"], dict):
            metrics = combined_results["simple_backtest"].get("metrics", {})
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
        else:
            f.write("Simple backtest failed or was skipped\n")
        
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
        if "monte_carlo" in combined_results and combined_results["monte_carlo"].get("status") != "skipped":
            f.write("\n" + "=" * 80 + "\n")
            f.write("MONTE CARLO SIMULATION RESULTS\n")
            f.write("=" * 80 + "\n")
            
            monte_carlo_results = combined_results.get("monte_carlo", {}).get("results", {})
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
        
        # Overall workflow status
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"WORKFLOW STATUS: {combined_results['status'].upper()}\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Complete workflow finished. Results saved to: {output_dir}")
    
    # Reset logging level if it was changed
    if verbose:
        logging_system.set_level('INFO', 'workflows')
    
    # Make sure to return in the format the cli.py expects
    return {
        "status": "success",
        "results": combined_results,
        "output_dir": output_dir
    } 