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
        bootstrap_pct: float = 0.5
    ):
        """
        Initialize the Monte Carlo analysis.
        
        Args:
            equity_curve: DataFrame with equity curve data (datetime index and equity values)
            num_simulations: Number of Monte Carlo simulations to run
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95% confidence)
            random_seed: Random seed for reproducibility
            bootstrap_pct: Percentage of original data to use in each bootstrap sample
        """
        self.equity_curve = equity_curve
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        self.bootstrap_pct = bootstrap_pct
        
        # Preprocess the equity curve to ensure it's properly formatted
        self.equity_curve = self._preprocess_equity_curve(equity_curve)
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Store results
        self.simulated_paths = None
        self.simulation_results = None
    
    def _preprocess_equity_curve(self, equity_curve: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Preprocess the equity curve to ensure it's properly formatted for Monte Carlo analysis.
        
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
        
        # Find equity column - look for names containing 'equity', 'value', or 'portfolio'
        equity_col_keywords = ['equity', 'value', 'portfolio', 'capital']
        for keyword in equity_col_keywords:
            equity_cols = [col for col in equity_curve.columns if keyword.lower() in col.lower()]
            if equity_cols:
                equity_col = equity_cols[0]
                break
        
        # If no equity column found, use the first numeric column
        if equity_col is None:
            for col in equity_curve.columns:
                if pd.api.types.is_numeric_dtype(equity_curve[col]):
                    equity_col = col
                    break
        
        # If we found date and equity columns, create a clean DataFrame
        if date_col and equity_col:
            try:
                # Ensure date is in datetime format
                clean_df = pd.DataFrame()
                clean_df['date'] = pd.to_datetime(equity_curve[date_col])
                clean_df['equity'] = equity_curve[equity_col]
                clean_df = clean_df.set_index('date')
                
                # Extract equity values as Series
                self.equity_values = clean_df['equity']
                
                # Calculate returns
                self.returns = self.equity_values.pct_change().dropna()
                
                return clean_df
            except Exception as e:
                print(f"Error preprocessing equity curve: {e}")
        
        # Fallback: try to find usable data in the DataFrame
        try:
            # If date is already the index, use that
            if pd.api.types.is_datetime64_any_dtype(equity_curve.index):
                if equity_col:
                    self.equity_values = equity_curve[equity_col]
                else:
                    # Use the first column
                    self.equity_values = equity_curve.iloc[:, 0]
            else:
                # No date index - use the first column that looks like numeric values
                for col in equity_curve.columns:
                    if pd.api.types.is_numeric_dtype(equity_curve[col]):
                        self.equity_values = equity_curve[col]
                        break
            
            # Calculate returns
            self.returns = self.equity_values.pct_change().dropna()
            
            return equity_curve
        except Exception as e:
            raise ValueError(f"Could not preprocess equity curve: {e}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations and analyze results.
        
        Returns:
            Dict containing simulation results
        """
        # Run simulations
        self.simulated_paths = self._run_simulations()
        
        # Analyze results
        self.simulation_results = self._analyze_results()
        
        return self.simulation_results
    
    def _run_simulations(self) -> pd.DataFrame:
        """
        Run Monte Carlo simulations by bootstrapping returns.
        
        Returns:
            DataFrame with simulated paths
        """
        # Get the initial value
        initial_equity = self.equity_values.iloc[0]
        
        # Create an empty DataFrame to store simulated paths
        simulated_paths = pd.DataFrame(index=range(len(self.returns) + 1))
        
        # Set the initial value for all simulations
        simulated_paths.loc[0] = initial_equity
        
        # Calculate bootstrap sample size
        sample_size = int(len(self.returns) * self.bootstrap_pct)
        
        # Run simulations
        for i in range(self.num_simulations):
            # Bootstrap returns
            bootstrap_returns = self.returns.sample(n=len(self.returns), replace=True).values
            
            # Generate path
            path = [initial_equity]
            for ret in bootstrap_returns:
                # Apply the return to the previous value
                path.append(path[-1] * (1 + ret))
            
            # Store the path
            simulated_paths[f'sim_{i}'] = path
        
        return simulated_paths
    
    def _analyze_results(self) -> Dict[str, Any]:
        """
        Analyze simulation results to extract key metrics.
        
        Returns:
            Dict containing analysis results
        """
        # Get the initial and final equity from the original curve
        initial_equity = self.equity_values.iloc[0]
        final_equity_original = self.equity_values.iloc[-1]
        return_original = (final_equity_original / initial_equity) - 1
        
        # Extract final equity values from all simulations and convert to numpy array
        final_equity_values = np.array(self.simulated_paths.iloc[-1].values)
        
        # Calculate returns
        returns = (final_equity_values / initial_equity) - 1
        
        # Calculate mean and median
        mean_final_equity = np.mean(final_equity_values)
        median_final_equity = np.median(final_equity_values)
        mean_return = np.mean(returns)
        
        # Calculate confidence intervals
        ci_lower_final_equity, ci_upper_final_equity = np.percentile(
            final_equity_values, 
            [(1 - self.confidence_level) * 100 / 2, 100 - (1 - self.confidence_level) * 100 / 2]
        )
        
        ci_lower_return, ci_upper_return = np.percentile(
            returns, 
            [(1 - self.confidence_level) * 100 / 2, 100 - (1 - self.confidence_level) * 100 / 2]
        )
        
        # Calculate VaR and CVaR
        var_idx = int(np.ceil((1 - self.confidence_level) * len(returns)))
        sorted_returns = np.sort(returns)
        var_pct = sorted_returns[var_idx]
        cvar_pct = np.mean(sorted_returns[:var_idx])
        
        # Calculate probability of profit (fix multi-dimensional indexing)
        returns_array = np.array(returns)  # Convert to numpy array before indexing
        probability_of_profit = np.sum(returns_array > 0) / len(returns_array)
        
        # Get best and worst returns
        worst_return = np.min(returns)
        best_return = np.max(returns)
        
        # Store results
        results = {
            'initial_equity': initial_equity,
            'final_equity_original': final_equity_original,
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
            'probability_of_profit': probability_of_profit,
            'worst_return': worst_return,
            'best_return': best_return,
            'num_simulations': self.num_simulations,
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
        
        # Plot original equity curve
        original_values = np.concatenate([[self.equity_values.iloc[0]], self.equity_values.values])
        ax.plot(original_values, color='red', linewidth=2, label='Original Equity Curve')
        
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