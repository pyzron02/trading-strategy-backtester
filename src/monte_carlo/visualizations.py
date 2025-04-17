#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo visualization module.

This module provides enhanced visualization capabilities for Monte Carlo simulations.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from typing import Dict, List, Any, Optional, Union, Tuple

# Try to import seaborn, but continue if it's not available
try:
    import seaborn as sns
except ImportError:
    sns = None

def format_currency(x, pos):
    """Format y-axis ticks as currency."""
    return f"${x:,.0f}"

def format_percent(x, pos):
    """Format y-axis ticks as percentage."""
    return f"{x:.1%}"

class MonteCarloVisualizer:
    """
    Enhanced visualization tools for Monte Carlo simulations.
    
    This class provides multiple visualization types for Monte Carlo simulation results.
    """
    
    def __init__(
        self, 
        equity_values: pd.Series,
        simulated_paths: pd.DataFrame,
        simulation_results: Dict[str, Any],
        confidence_level: float = 0.95,
        output_dir: str = None,
        strategy_name: str = "Strategy"
    ):
        """
        Initialize the Monte Carlo visualizer.
        
        Args:
            equity_values: Original equity curve values
            simulated_paths: DataFrame with simulated equity paths
            simulation_results: Dictionary with simulation results
            confidence_level: Confidence level for intervals
            output_dir: Directory to save plots
            strategy_name: Name of the strategy for plot titles
        """
        self.equity_values = equity_values
        self.simulated_paths = simulated_paths
        self.simulation_results = simulation_results
        self.confidence_level = confidence_level
        self.output_dir = output_dir
        self.strategy_name = strategy_name
        
        # Set style
        try:
            # Try to set a style, but catch exceptions if styles aren't available
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except:
                try:
                    plt.style.use('seaborn-whitegrid')
                except:
                    # If all else fails, use default style
                    pass
        except:
            # If plt.style is not available, continue without setting a style
            pass
        
    def create_all_plots(self, save: bool = True) -> Dict[str, str]:
        """
        Create all available plot types.
        
        Args:
            save: Whether to save the plots to disk
            
        Returns:
            Dictionary with plot paths
        """
        plot_paths = {}
        
        # Create individual plots
        plot_paths['simulation_paths'] = self.plot_simulation_paths(save=save)
        plot_paths['return_distribution'] = self.plot_return_distribution(save=save)
        plot_paths['drawdown_analysis'] = self.plot_drawdown_analysis(save=save)
        plot_paths['dashboard'] = self.create_dashboard(save=save)
        
        return plot_paths
    
    def plot_simulation_paths(self, save: bool = True) -> Optional[str]:
        """
        Create a visualization of the Monte Carlo simulation paths.
        
        Args:
            save: Whether to save the plot to disk
            
        Returns:
            Path to the saved plot if save=True, None otherwise
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot simulated paths (sample for clarity)
        sample_size = min(100, len(self.simulated_paths.columns))
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
        ax.set_title(f'{self.strategy_name}: Monte Carlo Simulation Paths')
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(FuncFormatter(format_currency))
        
        plt.tight_layout()
        
        # Save or show the plot
        if save and self.output_dir:
            plot_path = os.path.join(self.output_dir, f"{self.strategy_name}_monte_carlo_paths.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return plot_path
        else:
            plt.show()
            return None
    
    def plot_return_distribution(self, save: bool = True) -> Optional[str]:
        """
        Create a visualization of the return distribution.
        
        Args:
            save: Whether to save the plot to disk
            
        Returns:
            Path to the saved plot if save=True, None otherwise
        """
        # Calculate returns for all simulations
        initial_equity = self.simulation_results['initial_equity']
        final_equity_values = np.array(self.simulated_paths.iloc[-1].values)  # Convert to numpy array
        returns = (final_equity_values / initial_equity) - 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram of returns - avoid using seaborn due to multi-dimensional indexing issues
        # Convert to numpy array first
        returns_array = np.array(returns) if not isinstance(returns, np.ndarray) else returns
        # Use matplotlib directly instead of seaborn to avoid multi-dimensional indexing issues
        n, bins, patches = ax.hist(returns_array, bins=50, alpha=0.6, density=True)
        
        # Add a simple kde-like curve if we have numpy
        try:
            from scipy import stats
            kde_x = np.linspace(min(returns_array), max(returns_array), 1000)
            kde = stats.gaussian_kde(returns_array)
            ax.plot(kde_x, kde(kde_x), 'r-')
        except:
            # If scipy is not available, skip the kde part
            pass
        
        # Add vertical lines for key metrics
        original_return = self.simulation_results['return_original']
        mean_return = self.simulation_results['mean_return']
        var_pct = self.simulation_results['var_pct']
        
        ax.axvline(original_return, color='red', linestyle='--', linewidth=2, 
                   label=f'Original Return: {original_return:.2%}')
        ax.axvline(mean_return, color='green', linestyle='--', linewidth=2, 
                   label=f'Mean Return: {mean_return:.2%}')
        ax.axvline(var_pct, color='orange', linestyle='--', linewidth=2, 
                   label=f'VaR ({self.confidence_level:.0%}): {var_pct:.2%}')
        
        # Highlight probability of profit
        prob_profit = self.simulation_results['probability_of_profit']
        ax.axvline(0, color='black', linestyle='-', linewidth=2)
        ax.text(0.05, 0.95, f'Probability of Profit: {prob_profit:.2%}', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add labels and title
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{self.strategy_name}: Return Distribution')
        
        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(FuncFormatter(format_percent))
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        
        # Save or show the plot
        if save and self.output_dir:
            plot_path = os.path.join(self.output_dir, f"{self.strategy_name}_return_distribution.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return plot_path
        else:
            plt.show()
            return None
    
    def _calculate_drawdowns(self, equity_series: pd.Series) -> pd.Series:
        """
        Calculate drawdowns for an equity series.
        
        Args:
            equity_series: Series with equity values
            
        Returns:
            Series with drawdown percentages
        """
        # Calculate running maximum
        running_max = equity_series.cummax()
        
        # Calculate drawdown
        drawdown = (equity_series / running_max) - 1
        
        return drawdown
    
    def plot_drawdown_analysis(self, save: bool = True) -> Optional[str]:
        """
        Create a visualization of drawdown analysis.
        
        Args:
            save: Whether to save the plot to disk
            
        Returns:
            Path to the saved plot if save=True, None otherwise
        """
        # Calculate drawdowns for all simulations
        all_max_drawdowns = []
        
        for col in self.simulated_paths.columns:
            drawdowns = self._calculate_drawdowns(self.simulated_paths[col])
            all_max_drawdowns.append(drawdowns.min())
        
        # Calculate drawdown for original equity curve
        original_drawdowns = self._calculate_drawdowns(pd.Series(self.equity_values))
        original_max_drawdown = original_drawdowns.min()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram of max drawdowns - avoid using seaborn due to multi-dimensional indexing issues
        # Convert to numpy array first
        max_drawdowns_array = np.array(all_max_drawdowns) if not isinstance(all_max_drawdowns, np.ndarray) else all_max_drawdowns
        # Use matplotlib directly instead of seaborn to avoid multi-dimensional indexing issues
        n, bins, patches = ax.hist(max_drawdowns_array, bins=50, alpha=0.6, density=True)
        
        # Add a simple kde-like curve if we have numpy
        try:
            from scipy import stats
            kde_x = np.linspace(min(max_drawdowns_array), max(max_drawdowns_array), 1000)
            kde = stats.gaussian_kde(max_drawdowns_array)
            ax.plot(kde_x, kde(kde_x), 'r-')
        except:
            # If scipy is not available, skip the kde part
            pass
        
        # Add vertical line for original max drawdown
        ax.axvline(original_max_drawdown, color='red', linestyle='--', linewidth=2, 
                   label=f'Original Max Drawdown: {original_max_drawdown:.2%}')
        
        # Calculate statistics
        mean_max_drawdown = np.mean(all_max_drawdowns)
        median_max_drawdown = np.median(all_max_drawdowns)
        worst_drawdown = np.min(all_max_drawdowns)
        
        # Add vertical lines for key metrics
        ax.axvline(mean_max_drawdown, color='green', linestyle='--', linewidth=2, 
                   label=f'Mean Max Drawdown: {mean_max_drawdown:.2%}')
        ax.axvline(worst_drawdown, color='orange', linestyle='--', linewidth=2, 
                   label=f'Worst Drawdown: {worst_drawdown:.2%}')
        
        # Add labels and title
        ax.set_xlabel('Maximum Drawdown')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{self.strategy_name}: Maximum Drawdown Distribution')
        
        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(FuncFormatter(format_percent))
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        
        # Save or show the plot
        if save and self.output_dir:
            plot_path = os.path.join(self.output_dir, f"{self.strategy_name}_drawdown_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return plot_path
        else:
            plt.show()
            return None
    
    def create_dashboard(self, save: bool = True) -> Optional[str]:
        """
        Create a comprehensive dashboard of Monte Carlo visualization.
        
        Args:
            save: Whether to save the plot to disk
            
        Returns:
            Path to the saved plot if save=True, None otherwise
        """
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
        
        # 1. Simulation Paths (top row, spans both columns)
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot simulated paths (sample for clarity)
        sample_size = min(100, len(self.simulated_paths.columns))
        sample_cols = np.random.choice(self.simulated_paths.columns, sample_size, replace=False)
        
        for col in sample_cols:
            ax1.plot(self.simulated_paths[col], color='skyblue', alpha=0.1)
        
        # Plot original equity curve
        original_values = np.concatenate([[self.equity_values.iloc[0]], self.equity_values.values])
        ax1.plot(original_values, color='red', linewidth=2, label='Original Equity Curve')
        
        # Plot confidence interval
        lower_bound = self.simulated_paths.quantile((1 - self.confidence_level) / 2, axis=1)
        upper_bound = self.simulated_paths.quantile(1 - (1 - self.confidence_level) / 2, axis=1)
        median = self.simulated_paths.median(axis=1)
        
        ax1.plot(median, color='blue', linewidth=2, label='Median Simulation')
        ax1.fill_between(range(len(lower_bound)), lower_bound, upper_bound, color='blue', alpha=0.2, 
                         label=f'{self.confidence_level*100:.0f}% Confidence Interval')
        
        ax1.set_title(f'{self.strategy_name}: Monte Carlo Equity Curves ({self.simulation_results["num_simulations"]} simulations)')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Equity')
        ax1.yaxis.set_major_formatter(FuncFormatter(format_currency))
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 2. Return Distribution (middle row, left)
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Calculate returns for all simulations
        initial_equity = self.simulation_results['initial_equity']
        final_equity_values = np.array(self.simulated_paths.iloc[-1].values)  # Convert to numpy array
        returns = (final_equity_values / initial_equity) - 1
        
        # Plot histogram of returns - avoid using seaborn due to multi-dimensional indexing issues
        # Convert to numpy array first
        returns_array = np.array(returns) if not isinstance(returns, np.ndarray) else returns
        # Use matplotlib directly instead of seaborn to avoid multi-dimensional indexing issues
        n, bins, patches = ax2.hist(returns_array, bins=30, alpha=0.6, density=True)
        
        # Add a simple kde-like curve if we have numpy
        try:
            from scipy import stats
            kde_x = np.linspace(min(returns_array), max(returns_array), 1000)
            kde = stats.gaussian_kde(returns_array)
            ax2.plot(kde_x, kde(kde_x), 'r-')
        except:
            # If scipy is not available, skip the kde part
            pass
        
        # Add vertical lines for key metrics
        original_return = self.simulation_results['return_original']
        mean_return = self.simulation_results['mean_return']
        var_pct = self.simulation_results['var_pct']
        
        ax2.axvline(original_return, color='red', linestyle='--', linewidth=2, 
                    label=f'Original: {original_return:.2%}')
        ax2.axvline(mean_return, color='green', linestyle='--', linewidth=2, 
                    label=f'Mean: {mean_return:.2%}')
        ax2.axvline(var_pct, color='orange', linestyle='--', linewidth=2, 
                    label=f'VaR: {var_pct:.2%}')
        ax2.axvline(0, color='black', linestyle='-', linewidth=1)
        
        ax2.set_title('Return Distribution')
        ax2.set_xlabel('Return')
        ax2.set_ylabel('Frequency')
        ax2.xaxis.set_major_formatter(FuncFormatter(format_percent))
        ax2.legend(loc='upper left')
        
        # 3. Drawdown Analysis (middle row, right)
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Calculate drawdowns for all simulations
        all_max_drawdowns = []
        
        for col in self.simulated_paths.columns:
            drawdowns = self._calculate_drawdowns(self.simulated_paths[col])
            all_max_drawdowns.append(drawdowns.min())
        
        # Calculate drawdown for original equity curve
        original_drawdowns = self._calculate_drawdowns(pd.Series(self.equity_values))
        original_max_drawdown = original_drawdowns.min()
        
        # Plot histogram of max drawdowns - avoid using seaborn due to multi-dimensional indexing issues
        # Convert to numpy array first
        max_drawdowns_array = np.array(all_max_drawdowns) if not isinstance(all_max_drawdowns, np.ndarray) else all_max_drawdowns
        # Use matplotlib directly instead of seaborn to avoid multi-dimensional indexing issues
        n, bins, patches = ax3.hist(max_drawdowns_array, bins=30, alpha=0.6, density=True)
        
        # Add a simple kde-like curve if we have numpy
        try:
            from scipy import stats
            kde_x = np.linspace(min(max_drawdowns_array), max(max_drawdowns_array), 1000)
            kde = stats.gaussian_kde(max_drawdowns_array)
            ax3.plot(kde_x, kde(kde_x), 'r-')
        except:
            # If scipy is not available, skip the kde part
            pass
        
        # Add vertical line for original max drawdown
        ax3.axvline(original_max_drawdown, color='red', linestyle='--', linewidth=2, 
                    label=f'Original: {original_max_drawdown:.2%}')
        
        # Calculate statistics
        mean_max_drawdown = np.mean(all_max_drawdowns)
        worst_drawdown = np.min(all_max_drawdowns)
        
        ax3.axvline(mean_max_drawdown, color='green', linestyle='--', linewidth=2, 
                    label=f'Mean: {mean_max_drawdown:.2%}')
        ax3.axvline(worst_drawdown, color='orange', linestyle='--', linewidth=2, 
                    label=f'Worst: {worst_drawdown:.2%}')
        
        ax3.set_title('Maximum Drawdown Distribution')
        ax3.set_xlabel('Maximum Drawdown')
        ax3.set_ylabel('Frequency')
        ax3.xaxis.set_major_formatter(FuncFormatter(format_percent))
        ax3.legend(loc='upper right')
        
        # 4. Key Statistics Summary (bottom row, spans both columns)
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')  # Hide axes
        
        # Prepare statistics text
        stats = [
            f"Initial Equity: ${self.simulation_results['initial_equity']:,.2f}",
            f"Final Equity (Original): ${self.simulation_results['final_equity_original']:,.2f}",
            f"Return (Original): {self.simulation_results['return_original']:.2%}",
            f"Mean Final Equity: ${self.simulation_results['mean_final_equity']:,.2f}",
            f"Mean Return: {self.simulation_results['mean_return']:.2%}",
            f"Probability of Profit: {self.simulation_results['probability_of_profit']:.2%}",
            f"Value at Risk ({self.confidence_level:.0%}): {self.simulation_results['var_pct']:.2%}",
            f"Conditional VaR ({self.confidence_level:.0%}): {self.simulation_results['cvar_pct']:.2%}",
            f"Worst Return: {self.simulation_results['worst_return']:.2%}",
            f"Best Return: {self.simulation_results['best_return']:.2%}",
        ]
        
        # Create a table
        try:
            col_widths = [0.5, 0.5]  # Equal width columns
            table_data = []
            
            # Split stats into two columns
            mid_point = len(stats) // 2
            for i in range(mid_point):
                if i < len(stats) - mid_point:
                    table_data.append([stats[i], stats[i + mid_point]])
                else:
                    table_data.append([stats[i], ""])
            
            table = ax4.table(cellText=table_data, cellLoc='left', loc='center', 
                            colWidths=col_widths, bbox=[0.0, 0.0, 1.0, 1.0])
        except Exception as e:
            # If table creation fails, add text instead
            ax4.text(0.5, 0.5, "\n".join(stats), ha='center', va='center', fontsize=10)
        
        # Style the table (only if table creation succeeded)
        try:
            if 'table' in locals():
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                for i in range(len(table_data)):
                    for j in range(len(col_widths)):
                        cell = table.get_celld()[i, j]
                        cell.set_height(0.1)
        except Exception:
            # Skip table styling if it fails
            pass
                
        # Add title for the statistics section
        ax4.text(0.5, 1.02, 'Monte Carlo Simulation Statistics', 
                 horizontalalignment='center', verticalalignment='bottom',
                 fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save or show the plot
        if save and self.output_dir:
            plot_path = os.path.join(self.output_dir, f"{self.strategy_name}_monte_carlo_dashboard.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return plot_path
        else:
            plt.show()
            return None