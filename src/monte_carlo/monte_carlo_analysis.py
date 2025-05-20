#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo Analysis module for equity curve simulation and analysis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
import random
from datetime import datetime, timedelta, date

class MonteCarloAnalysis:
    """
    Monte Carlo Analysis for backtesting results.
    
    This class performs Monte Carlo simulations by bootstrapping returns
    from an equity curve to analyze the distribution of potential outcomes.
    """
    
    def __init__(
        self,
        equity_curve: pd.DataFrame,
        num_simulations: int = 1000,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None,
        bootstrap_pct: float = 0.5,
        bootstrap_method: str = 'standard',
        block_size: int = 21
    ):
        """
        Initialize the Monte Carlo analysis.
        
        Args:
            equity_curve: DataFrame with equity curve data (datetime index and equity values)
            num_simulations: Number of Monte Carlo simulations to run
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95% confidence)
            random_seed: Random seed for reproducibility
            bootstrap_pct: Percentage of original data to use in each bootstrap sample
            bootstrap_method: Method for bootstrapping returns ('standard', 'block', 'stationary')
            block_size: Size of blocks for block bootstrapping (in trading days)
        """
        self.equity_curve = equity_curve
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        self.bootstrap_pct = bootstrap_pct
        self.bootstrap_method = bootstrap_method
        self.block_size = block_size
        
        # Process equity curve data
        processed_data = self._preprocess_equity_curve(equity_curve)
        
        # Extract single equity values series if needed
        if len(processed_data.columns) > 1:
            self.equity_values = processed_data.iloc[:, 0]
        else:
            self.equity_values = processed_data.iloc[:, 0]
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Store results
        self.simulated_paths = None
        self.simulation_results = None
    
    def _preprocess_equity_curve(self, equity_curve: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Preprocess equity curve data for Monte Carlo analysis.
        
        Args:
            equity_curve: DataFrame or Series with equity curve data
            
        Returns:
            Properly formatted DataFrame with numeric equity values
        """
        # Convert Series to DataFrame if needed
        if isinstance(equity_curve, pd.Series):
            equity_curve = equity_curve.to_frame()
        
        # Identify date column and equity column
        date_col = None
        equity_col = None
        
        # Find date column
        date_cols = [col for col in equity_curve.columns if col.lower() == 'date']
        if date_cols:
            date_col = date_cols[0]
        
        # If data has a date column, set it as index
        if date_col is not None and date_col in equity_curve.columns:
            equity_curve = equity_curve.set_index(date_col)
        
        # If no numeric columns, raise error
        numeric_cols = equity_curve.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in equity curve data")
        
        # Find equity column (prefer 'equity' column if it exists)
        equity_col_candidates = [col for col in numeric_cols if col.lower() in ['equity', 'balance', 'portfolio_value', 'value']]
        if equity_col_candidates:
            equity_col = equity_col_candidates[0]
        else:
            # Otherwise, use the first numeric column
            equity_col = numeric_cols[0]
        
        # Ensure equity column exists
        if equity_col not in equity_curve.columns:
            # If not found, try to identify based on content
            try:
                # Select the column with equity values (highest starting value usually)
                for col in equity_curve.columns:
                    if pd.api.types.is_numeric_dtype(equity_curve[col]):
                        equity_col = col
                        break
            except:
                raise ValueError("Could not identify equity column in data")
        
        # Return processed data
        try:
            # Try to extract equity column
            result = equity_curve[[equity_col]].copy()
            return result
        except Exception as e:
            # If specific equity column extraction fails, use all numeric data
            try:
                for col in equity_curve.columns:
                    if pd.api.types.is_numeric_dtype(equity_curve[col]):
                        self.equity_values = equity_curve[col]
                        break
                
                return equity_curve
            except Exception as e:
                raise ValueError(f"Could not preprocess equity curve: {e}")
    
    def run(self, progress_file=None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations and analyze results.
        
        Args:
            progress_file: Optional path to a progress file for frontend updates
            
        Returns:
            Dict containing simulation results
        """
        # Run simulations
        self.simulated_paths = self._run_simulations(progress_file)
        
        # Update progress that we're analyzing results if progress file is provided
        if progress_file and os.path.exists(progress_file):
            try:
                import json
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                progress_data.update({
                    'current_step': "Analyzing Monte Carlo results",
                    'progress': 90,
                    'current_step_progress': 0,
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=4)
            except Exception as e:
                print(f"Error updating progress file: {e}")
        
        # Calculate key metrics
        results = self._calculate_metrics()
        
        # Update progress file to indicate completion
        if progress_file and os.path.exists(progress_file):
            try:
                import json
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                progress_data.update({
                    'current_step': "Monte Carlo simulation completed",
                    'progress': 100,
                    'current_step_progress': 100,
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=4)
            except Exception as e:
                print(f"Error updating progress file: {e}")
        
        # Store results and return them
        self.simulation_results = results
        
        return results
    
    def _run_simulations(self, progress_file=None) -> pd.DataFrame:
        """
        Run Monte Carlo simulations using bootstrap of returns.
        
        Args:
            progress_file: Path to a file for tracking progress
            
        Returns:
            DataFrame with simulated equity paths
        """
        # Initialize containers
        initial_equity = self.equity_values.iloc[0]
        
        # Ensure initial equity is not zero to avoid division by zero
        if initial_equity == 0:
            initial_equity = 0.01  # Set a minimal positive value instead of zero
            print("Warning: Initial equity was zero. Setting to 0.01 to avoid division by zero.")
        
        # Get log returns for better numerical stability (log(1+r))
        # Log returns are more suitable for Monte Carlo simulations as they can be
        # added rather than multiplied, providing better numerical stability
        log_returns_array = self.log_returns.values
        num_returns = len(log_returns_array)
        
        # Calculate bootstrap sample size - used for status updates
        sample_size = int(num_returns * self.bootstrap_pct)
        
        # Pre-allocate a list to store all paths - avoid DataFrame fragmentation
        all_paths = []
        
        print(f"Running {self.num_simulations} Monte Carlo simulations on CPU with log returns for numerical stability...")
            
        start_time = datetime.now()
        
        # Vectorize CPU implementation for better performance
        paths_array = np.zeros((num_returns + 1, self.num_simulations), dtype=np.float64)
        paths_array[0, :] = initial_equity
        
        # Create batches for progress reporting
        batch_size = min(1000, self.num_simulations)
        num_batches = (self.num_simulations + batch_size - 1) // batch_size
        
        # We're already using log returns for numerical stability
        
        # Process in batches
        for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, self.num_simulations)
                batch_count = end_idx - start_idx
                
                # Update progress if progress file provided
                if progress_file and os.path.exists(progress_file):
                    progress_pct = min(80, int(20 + (batch_idx / num_batches) * 60))
                    try:
                        import json
                        with open(progress_file, 'r') as f:
                            progress_data = json.load(f)
                        
                        progress_data.update({
                            'current_step': "Monte Carlo CPU Simulation",
                            'progress': progress_pct,
                            'current_step_progress': int((batch_idx / num_batches) * 100),
                            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        
                        with open(progress_file, 'w') as f:
                            json.dump(progress_data, f, indent=4)
                    except Exception as e:
                        print(f"Error updating progress file: {e}")
                
                # For each simulation in the batch
                for sim_idx in range(start_idx, end_idx):
                    # Generate bootstrap sample indices to cover the full simulation length
                    # We need enough returns for the full paths_array length, not just sample_size
                    required_returns = num_returns  # Use full path length
                    
                    if self.bootstrap_method == 'block' and num_returns > self.block_size:
                        # Block bootstrap - sample in blocks rather than individual returns
                        # This preserves some of the time series properties
                        max_start_idx = num_returns - self.block_size
                        num_blocks_needed = (required_returns + self.block_size - 1) // self.block_size
                        block_starts = np.random.randint(0, max_start_idx, size=num_blocks_needed)
                        indices = []
                        for start in block_starts:
                            block_indices = np.arange(start, min(start + self.block_size, num_returns))
                            indices.extend(block_indices)
                        
                        # Make sure we have enough indices (repeat if necessary)
                        while len(indices) < required_returns:
                            # Add more blocks if needed
                            start = np.random.randint(0, max_start_idx)
                            block_indices = np.arange(start, min(start + self.block_size, num_returns))
                            indices.extend(block_indices)
                        
                        # Trim to the exact size we need
                        indices = indices[:required_returns]
                    else:
                        # Standard bootstrap - randomly sample with replacement
                        indices = np.random.choice(num_returns, size=required_returns, replace=True)
                    
                    # Get the bootstrap sample of log returns
                    bootstrap_log_returns = log_returns_array[indices]
                    
                    # Generate path using cumulative log returns for better numerical stability
                    cum_log_return = 0.0
                    paths_array[0, sim_idx] = initial_equity
                    
                    # Generate all time steps for this simulation path
                    # Ensure we don't exceed the paths_array dimensions
                    for t in range(min(len(bootstrap_log_returns), num_returns)):
                        # Add the log return to the cumulative sum
                        cum_log_return += bootstrap_log_returns[t]
                        
                        # Calculate the equity using exp of log returns (initial_equity * e^(cum_log_return))
                        equity = initial_equity * np.exp(cum_log_return)
                        
                        # Prevent equity from becoming too small - establish a minimum floor
                        if equity < 0.01:
                            equity = 0.01
                            # Reset the cumulative log return based on the minimum equity
                            cum_log_return = np.log(equity / initial_equity)
                        
                        # Prevent equity from becoming NaN or infinity
                        if np.isnan(equity) or np.isinf(equity):
                            equity = paths_array[t, sim_idx]  # Use the previous value
                            cum_log_return = np.log(equity / initial_equity)
                        
                        # Store the result
                        paths_array[t+1, sim_idx] = equity
                            
                    # Ensure all remaining rows have valid values (in case bootstrap returns are shorter)
                    for t in range(len(bootstrap_log_returns), num_returns):
                        # If we run out of bootstrapped returns, use the last valid equity value
                        paths_array[t+1, sim_idx] = paths_array[t, sim_idx]
                
                # Update progress file if provided
                if progress_file and os.path.exists(progress_file) and batch_idx % max(1, num_batches//10) == 0:
                    try:
                        import json
                        with open(progress_file, 'r') as f:
                            progress_data = json.load(f)
                        
                        progress_data.update({
                            'current_step': "Running Monte Carlo simulations",
                            'progress': min(80, int(30 + (batch_idx / num_batches) * 50)),
                            'current_step_progress': int((batch_idx / num_batches) * 100),
                            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        
                        with open(progress_file, 'w') as f:
                            json.dump(progress_data, f, indent=4)
                    except Exception as e:
                        print(f"Error updating progress file: {e}")
                
                # End timer
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                print(f"CPU Monte Carlo completed in {duration:.2f} seconds ({duration/self.num_simulations:.6f} seconds per simulation)")
                
                # Convert results to pandas Series
                for sim_idx in range(self.num_simulations):
                    all_paths.append(pd.Series(paths_array[:, sim_idx], name=f'sim_{sim_idx}'))
        
        # Convert all paths to DataFrame
        if len(all_paths) == 0:
            raise ValueError("No simulation paths generated")
        
        simulated_df = pd.concat(all_paths, axis=1)
        
        # Return the DataFrame with simulated paths
        return simulated_df

    @property
    def returns(self):
        """Get returns from equity values."""
        return self.equity_values.pct_change().dropna()
        
    @property
    def log_returns(self):
        """Get log returns from equity values for better numerical stability.
        
        Log returns (log(1+r)) provide better numerical stability, especially for
        Monte Carlo simulations with many iterations or when computing cumulative returns.
        """
        return np.log1p(self.returns)
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics from the Monte Carlo simulation results.
        
        Returns:
            Dict containing calculated metrics
        """
        if self.simulated_paths is None:
            raise ValueError("No simulation results available. Run the simulation first.")
        
        # Get initial and final values for the original equity curve
        initial_equity = self.equity_values.iloc[0]
        final_equity = self.equity_values.iloc[-1]
        return_original = final_equity / initial_equity - 1
        
        # Calculate key metrics across all simulations
        final_equities = self.simulated_paths.iloc[-1, :]
        
        # Mean and median final equity
        mean_final_equity = final_equities.mean()
        median_final_equity = final_equities.median()
        
        # Mean return
        mean_return = mean_final_equity / initial_equity - 1
        
        # Confidence interval for final equity
        ci_lower_pct = (1 - self.confidence_level) / 2
        ci_upper_pct = 1 - ci_lower_pct
        
        ci_lower_final_equity = final_equities.quantile(ci_lower_pct)
        ci_upper_final_equity = final_equities.quantile(ci_upper_pct)
        
        # Confidence interval for returns
        returns = final_equities / initial_equity - 1
        ci_lower_return = returns.quantile(ci_lower_pct)
        ci_upper_return = returns.quantile(ci_upper_pct)
        
        # Value at Risk (VaR) and Conditional VaR (CVaR)
        var_pct = -returns.quantile(ci_lower_pct)  # Negative of the lower CI bound
        cvar_pct = -returns[returns <= -var_pct].mean()  # Average of losses beyond VaR
        
        # Worst and best case returns
        worst_return = returns.min()
        best_return = returns.max()
        
        # Probability of profit
        profit_prob = (returns > 0).mean()
        
        # Collect all metrics into a dictionary
        results = {
            'initial_equity': initial_equity,
            'final_equity_original': final_equity,
            'return_original': return_original,
            'mean_final_equity': mean_final_equity,
            'median_final_equity': median_final_equity,
            'mean_return': mean_return,
            'ci_lower_final_equity': ci_lower_final_equity,
            'ci_upper_final_equity': ci_upper_final_equity,
            'ci_lower_return': ci_lower_return,
            'ci_upper_return': ci_upper_return,
            'var_pct': var_pct,
            'cvar_pct': cvar_pct,
            'worst_return': worst_return,
            'best_return': best_return,
            'probability_of_profit': profit_prob,
            'confidence_level': self.confidence_level
        }
        
        return results
    
    def plot(self, save_path: Optional[str] = None) -> None:
        """
        Create a visualization of the Monte Carlo simulations.
        
        Args:
            save_path: Path to save the plot (if None, plot is displayed)
        """
        if self.simulated_paths is None:
            raise ValueError("No simulation results available. Run the simulation first.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot simulated paths (sample for clarity)
        sample_size = min(100, self.num_simulations)
        sample_cols = np.random.choice(self.simulated_paths.columns, sample_size, replace=False)
        
        for col in sample_cols:
            ax.plot(self.simulated_paths[col], color='skyblue', alpha=0.1)
        
        # Plot optimized equity curve
        original_values = np.concatenate([[self.equity_values.iloc[0]], self.equity_values.values])
        ax.plot(original_values, color='red', linewidth=2, label='Optimized Equity Curve')
        
        # Plot confidence interval
        lower_bound = self.simulated_paths.quantile((1 - self.confidence_level) / 2, axis=1)
        upper_bound = self.simulated_paths.quantile(1 - (1 - self.confidence_level) / 2, axis=1)
        median = self.simulated_paths.median(axis=1)
        
        ax.plot(median, color='blue', linewidth=2, label='Median Simulation')
        ax.fill_between(range(len(lower_bound)), lower_bound, upper_bound, color='blue', alpha=0.2, 
                        label=f'{self.confidence_level*100:.0f}% Confidence Interval')
        
        # Add labels and title
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Equity')
        ax.set_title(f'Monte Carlo Simulation ({self.num_simulations} runs)')
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Format y-axis as currency
        plt.ticklabel_format(style='plain', axis='y')
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()