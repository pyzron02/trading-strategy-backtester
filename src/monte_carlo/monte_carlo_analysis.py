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
        block_size: int = 21,
        keep_permuted_data: bool = False,
        output_dir: Optional[str] = None
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
            keep_permuted_data: Whether to save permuted data files for analysis
            output_dir: Directory to save permuted data files (required if keep_permuted_data=True)
        """
        self.equity_curve = equity_curve
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        self.bootstrap_pct = bootstrap_pct
        self.bootstrap_method = bootstrap_method
        self.block_size = block_size
        self.keep_permuted_data = keep_permuted_data
        self.output_dir = output_dir
        
        # Create permuted data directory if needed
        self.permuted_data_dir = None
        if self.keep_permuted_data and self.output_dir:
            self.permuted_data_dir = os.path.join(self.output_dir, "permuted_data")
            os.makedirs(self.permuted_data_dir, exist_ok=True)
        
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
        # Validate data sufficiency before running simulations
        validation_warnings = self._validate_data_sufficiency()
        if validation_warnings:
            print("\n" + "="*60)
            print("MONTE CARLO VALIDATION WARNINGS:")
            for warning in validation_warnings:
                print(f"  - {warning}")
            print("="*60 + "\n")
        
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
        
        # Add result validation
        validation_report = self._validate_monte_carlo_results(results)
        if validation_report['warnings']:
            print("\n" + "="*60)
            print("MONTE CARLO RESULT VALIDATION:")
            for warning in validation_report['warnings']:
                print(f"  - {warning}")
            print("="*60 + "\n")
        
        # Store validation in results
        results['validation'] = validation_report
        
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
    
    def _validate_data_sufficiency(self) -> List[str]:
        """
        Validate that we have sufficient data for meaningful Monte Carlo analysis.
        
        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []
        
        # Check minimum data points
        MIN_DATA_POINTS = 30
        if len(self.equity_values) < MIN_DATA_POINTS:
            warnings.append(f"Only {len(self.equity_values)} data points available. "
                          f"Recommend at least {MIN_DATA_POINTS} for meaningful results.")
        
        # Check for variance in returns
        if hasattr(self, 'log_returns') and len(self.log_returns) > 0:
            unique_returns = np.unique(self.log_returns)
            if len(unique_returns) == 1:
                warnings.append("All returns are identical. Monte Carlo results will show no variation.")
            elif len(unique_returns) < 5:
                warnings.append(f"Only {len(unique_returns)} unique return values. Results may show limited variation.")
        
        # Check for extremely low volatility
        if hasattr(self, 'log_returns') and len(self.log_returns) > 0:
            volatility = np.std(self.log_returns)
            if volatility < 0.0001:  # Less than 0.01% daily volatility
                warnings.append(f"Extremely low volatility detected ({volatility:.6f}). "
                              "Monte Carlo results may not be meaningful.")
        
        # Estimate number of trades from return patterns
        if hasattr(self, 'log_returns'):
            # Count non-zero returns as a proxy for trading activity
            non_zero_returns = np.sum(self.log_returns != 0)
            if non_zero_returns < 10:
                warnings.append(f"Very low trading activity detected ({non_zero_returns} non-zero returns). "
                              "Consider checking strategy parameters.")
        
        return warnings
    
    def _validate_monte_carlo_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Monte Carlo simulation results for common issues.
        
        Args:
            results: Dictionary containing Monte Carlo results
            
        Returns:
            Dictionary with validation status and warnings
        """
        validation_report = {
            'status': 'valid',
            'warnings': [],
            'recommendations': []
        }
        
        # Check if all simulations produced identical results
        if 'all_final_equity' in results:
            unique_values = len(set(results['all_final_equity']))
            total_sims = len(results['all_final_equity'])
            
            if unique_values == 1:
                validation_report['warnings'].append(
                    "All Monte Carlo simulations produced identical results. "
                    "This indicates insufficient variance in the underlying data."
                )
                validation_report['recommendations'].append(
                    "Check if strategy has sufficient trading activity and return variation."
                )
                validation_report['status'] = 'suspicious'
            elif unique_values < total_sims * 0.1:  # Less than 10% unique values
                validation_report['warnings'].append(
                    f"Very low variation in results: only {unique_values} unique values "
                    f"from {total_sims} simulations."
                )
                
        # Check coefficient of variation for key metrics
        for metric in ['final_equity', 'max_drawdown', 'sharpe_ratio']:
            all_values_key = f'all_{metric}'
            if all_values_key in results:
                values = results[all_values_key]
                if len(values) > 0 and np.std(values) > 0:
                    cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                    if cv < 0.01:  # Less than 1% coefficient of variation
                        validation_report['warnings'].append(
                            f"Very low variation in {metric} (CV={cv:.4f}). "
                            "Results may not be meaningful."
                        )
                        
        # Check for reasonable probability of profit
        if 'probability_of_profit' in results:
            prob_profit = results['probability_of_profit']
            if prob_profit == 1.0:
                validation_report['warnings'].append(
                    "100% probability of profit detected. This is unrealistic and indicates "
                    "insufficient data variation."
                )
            elif prob_profit == 0.0:
                validation_report['warnings'].append(
                    "0% probability of profit detected. Strategy may be fundamentally flawed."
                )
                
        # Check simulation count
        if 'num_simulations' in results:
            if results['num_simulations'] < 1000:
                validation_report['warnings'].append(
                    f"Low simulation count ({results['num_simulations']}). "
                    "Consider running more simulations for better statistical significance."
                )
                
        return validation_report
    
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
        if initial_equity == 0 or np.isnan(initial_equity) or np.isinf(initial_equity):
            initial_equity = 0.01  # Set a minimal positive value instead of zero
            print(f"Warning: Initial equity was invalid ({initial_equity}). Setting to 0.01 to avoid numerical issues.")
        
        # Validate that we have sufficient data
        if len(self.equity_values) < 2:
            raise ValueError(f"Insufficient data for Monte Carlo simulation. Need at least 2 data points, got {len(self.equity_values)}")
        
        # Get log returns for better numerical stability (log(1+r))
        # Log returns are more suitable for Monte Carlo simulations as they can be
        # added rather than multiplied, providing better numerical stability
        log_returns_array = self.log_returns.values
        num_returns = len(log_returns_array)
        
        # Calculate bootstrap sample size - used for status updates
        sample_size = int(num_returns * self.bootstrap_pct)
        
        # Pre-allocate a list to store all paths - avoid DataFrame fragmentation
        all_paths = []
        
        # Initialize permuted data storage if needed
        permuted_data_list = []
        if self.keep_permuted_data:
            print(f"Running {self.num_simulations} Monte Carlo simulations on CPU with log returns for numerical stability...")
            print(f"Permuted data will be saved to: {self.permuted_data_dir}")
        else:
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
                
                # Update progress if progress file provided (only update every 10% to reduce I/O)
                if progress_file and os.path.exists(progress_file) and batch_idx % max(1, num_batches // 10) == 0:
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
                    
                    # Save permuted data if requested
                    if self.keep_permuted_data:
                        # Store the permuted log returns for this simulation
                        permuted_returns_df = pd.DataFrame({
                            'index': range(len(bootstrap_log_returns)),
                            'log_returns': bootstrap_log_returns,
                            'original_indices': indices
                        })
                        permuted_data_list.append({
                            'simulation_id': sim_idx,
                            'bootstrap_indices': indices,
                            'bootstrap_log_returns': bootstrap_log_returns,
                            'returns_df': permuted_returns_df
                        })
                    
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
                        min_equity = 0.01
                        max_equity = initial_equity * 1000  # Prevent unrealistic growth
                        
                        if equity < min_equity:
                            equity = min_equity
                            # Reset the cumulative log return based on the minimum equity
                            cum_log_return = np.log(equity / initial_equity)
                        elif equity > max_equity:
                            equity = max_equity
                            # Reset the cumulative log return based on the maximum equity
                            cum_log_return = np.log(equity / initial_equity)
                        
                        # Prevent equity from becoming NaN or infinity
                        if np.isnan(equity) or np.isinf(equity):
                            equity = paths_array[t, sim_idx]  # Use the previous value
                            # Safely calculate cum_log_return
                            if equity > 0 and initial_equity > 0:
                                cum_log_return = np.log(equity / initial_equity)
                            else:
                                cum_log_return = 0.0
                        
                        # Store the result
                        paths_array[t+1, sim_idx] = equity
                            
                    # Ensure all remaining rows have valid values (in case bootstrap returns are shorter)
                    for t in range(len(bootstrap_log_returns), num_returns):
                        # If we run out of bootstrapped returns, use the last valid equity value
                        paths_array[t+1, sim_idx] = paths_array[t, sim_idx]
                
                # Progress update moved to after loop to avoid duplicate updates
                
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
        
        # Save permuted data files if requested
        if self.keep_permuted_data and self.permuted_data_dir:
            self._save_permuted_data(permuted_data_list, simulated_df)
        
        # Return the DataFrame with simulated paths
        return simulated_df
    
    def _save_permuted_data(self, permuted_data_list: List[Dict], simulated_paths: pd.DataFrame):
        """
        Save permuted data files to disk for analysis.
        
        Args:
            permuted_data_list: List of dictionaries containing permuted data for each simulation
            simulated_paths: DataFrame containing all simulated equity paths
        """
        import json
        from datetime import datetime
        
        try:
            print(f"Saving permuted data for {len(permuted_data_list)} simulations...")
            
            # 1. Save original equity curve data
            original_file = os.path.join(self.permuted_data_dir, "original_equity_curve.csv")
            self.equity_values.to_csv(original_file, header=['Original_Equity'])
            print(f"Saved original equity curve to: {original_file}")
            
            # 2. Save all simulated paths
            all_paths_file = os.path.join(self.permuted_data_dir, "all_simulation_paths.csv")
            simulated_paths.to_csv(all_paths_file)
            print(f"Saved all simulation paths to: {all_paths_file}")
            
            # Individual permuted return files are not needed since all_simulation_paths.csv contains all data
            print("Individual permuted return files skipped - all data available in all_simulation_paths.csv")
            
            # 4. Save metadata about the permutations
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'total_simulations': self.num_simulations,
                'bootstrap_method': self.bootstrap_method,
                'bootstrap_pct': self.bootstrap_pct,
                'block_size': self.block_size,
                'confidence_level': self.confidence_level,
                'random_seed': getattr(self, 'random_seed', None),
                'original_equity_length': len(self.equity_values),
                'file_structure': {
                    'original_equity_curve.csv': 'Original equity curve data',
                    'all_simulation_paths.csv': 'All Monte Carlo simulation equity paths',
                    'metadata.json': 'This metadata file',
                    'README.md': 'Documentation about the permuted data'
                }
            }
            
            metadata_file = os.path.join(self.permuted_data_dir, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to: {metadata_file}")
            
            # 5. Save README documentation
            readme_content = f"""# Monte Carlo Permuted Data

This directory contains permuted data generated from Monte Carlo analysis.

## Overview
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Simulations**: {self.num_simulations}
- **Bootstrap Method**: {self.bootstrap_method}
- **Bootstrap Percentage**: {self.bootstrap_pct}
- **Confidence Level**: {self.confidence_level}

## File Structure

### original_equity_curve.csv
Contains the original equity curve data that was used as input for the Monte Carlo analysis.

### all_simulation_paths.csv
Contains all {self.num_simulations} simulated equity paths. Each column represents one simulation.
This file contains the complete Monte Carlo simulation results.

### metadata.json
Contains detailed metadata about the simulation parameters and file structure.

## Usage
This data can be used for:
- Analyzing the distribution of potential outcomes from Monte Carlo simulations
- Statistical analysis of equity curve variations
- Understanding portfolio performance under different scenarios
- Reproducing or extending the Monte Carlo analysis
- Risk assessment and stress testing
"""
            
            readme_file = os.path.join(self.permuted_data_dir, "README.md")
            with open(readme_file, 'w') as f:
                f.write(readme_content)
            print(f"Saved documentation to: {readme_file}")
            
            print(f"✅ Successfully saved all permuted data to: {self.permuted_data_dir}")
            
        except Exception as e:
            print(f"❌ Error saving permuted data: {str(e)}")
            import traceback
            traceback.print_exc()

    @property
    def returns(self):
        """Get returns from equity values with safe division."""
        returns = self.equity_values.pct_change()
        # Replace infinite values with NaN
        returns = returns.replace([np.inf, -np.inf], np.nan)
        # Drop NaN values
        returns = returns.dropna()
        # Validate that we have returns
        if len(returns) == 0:
            raise ValueError("No valid returns calculated from equity curve")
        return returns
        
    @property
    def log_returns(self):
        """Get log returns from equity values for better numerical stability.
        
        Log returns (log(1+r)) provide better numerical stability, especially for
        Monte Carlo simulations with many iterations or when computing cumulative returns.
        """
        returns = self.returns
        # Clip extreme returns to avoid numerical issues
        # Limit returns to -99% to +1000% to prevent log of negative values
        clipped_returns = np.clip(returns, -0.99, 10.0)
        if (returns != clipped_returns).any():
            print(f"Warning: {(returns != clipped_returns).sum()} extreme returns clipped for numerical stability")
        
        log_returns = np.log1p(clipped_returns)
        
        # Final validation
        if np.isnan(log_returns).any() or np.isinf(log_returns).any():
            # Remove any remaining invalid values
            log_returns = pd.Series(log_returns).replace([np.inf, -np.inf], np.nan).dropna()
            print(f"Warning: Removed {np.isnan(log_returns).sum() + np.isinf(log_returns).sum()} invalid log returns")
        
        return log_returns
    
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
        
        # Calculate CVaR with safety checks
        losses_beyond_var = returns[returns <= -var_pct]
        if len(losses_beyond_var) > 0:
            cvar_pct = -losses_beyond_var.mean()  # Average of losses beyond VaR
        else:
            cvar_pct = var_pct  # If no losses beyond VaR, use VaR itself
        
        # Worst and best case returns
        worst_return = returns.min()
        best_return = returns.max()
        
        # Probability of profit
        profit_prob = (returns > 0).mean()
        
        # Calculate maximum drawdown in dollars for each simulation
        all_max_drawdown_dollars = []
        all_max_drawdown_pct = []
        
        for col in self.simulated_paths.columns:
            sim_path = self.simulated_paths[col]
            peak = sim_path.cummax()
            
            # Dollar drawdown
            drawdown_dollars = peak - sim_path
            max_dd_dollars = drawdown_dollars.max()
            all_max_drawdown_dollars.append(max_dd_dollars)
            
            # Percentage drawdown
            drawdown_pct = (sim_path - peak) / peak
            max_dd_pct = abs(drawdown_pct.min())
            all_max_drawdown_pct.append(max_dd_pct)
        
        # Calculate original drawdown
        original_peak = self.equity_values.cummax()
        original_drawdown_dollars = (original_peak - self.equity_values).max()
        original_drawdown_pct = abs(((self.equity_values - original_peak) / original_peak).min())
        
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
            'confidence_level': self.confidence_level,
            'max_drawdown_original': original_drawdown_dollars,
            'max_drawdown_original_pct': original_drawdown_pct,
            'all_max_drawdown': all_max_drawdown_dollars,
            'all_max_drawdown_pct': all_max_drawdown_pct,
            'mean_max_drawdown': np.mean(all_max_drawdown_dollars),
            'median_max_drawdown': np.median(all_max_drawdown_dollars),
            'ci_lower_max_drawdown': np.percentile(all_max_drawdown_dollars, ci_lower_pct * 100),
            'ci_upper_max_drawdown': np.percentile(all_max_drawdown_dollars, ci_upper_pct * 100)
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