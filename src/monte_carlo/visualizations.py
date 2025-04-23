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

# Import plotly for interactive visualizations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Check if kaleido is available for saving static images
try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False
    import warnings
    warnings.warn("kaleido package not found. Static image export will be disabled. Install with 'pip install kaleido'")

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
        strategy_name: str = "Strategy",
        workflow_type: str = None
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
            workflow_type: Type of workflow (simple, optimization, monte_carlo, complete)
        """
        self.equity_values = equity_values
        self.simulated_paths = simulated_paths
        self.simulation_results = simulation_results
        self.confidence_level = confidence_level
        self.output_dir = output_dir
        self.strategy_name = strategy_name
        self.workflow_type = workflow_type
        
        # Set Plotly template
        pio.templates.default = "plotly_white"
        
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
        Create an interactive visualization of the Monte Carlo simulation paths.
        
        Args:
            save: Whether to save the plot to disk
            
        Returns:
            Path to the saved plot if save=True, None otherwise
        """
        # Check if we have valid data
        if self.simulated_paths is None or len(self.simulated_paths.columns) == 0:
            import logging
            logging.getLogger(__name__).error("No simulation paths available for plotting")
            return None
            
        # Create figure
        fig = go.Figure()
        
        # Plot simulated paths (sample for clarity)
        sample_size = min(100, len(self.simulated_paths.columns))
        sample_cols = np.random.choice(self.simulated_paths.columns, sample_size, replace=False)
        
        # Add simulated paths with low opacity
        for col in sample_cols:
            fig.add_trace(
                go.Scatter(
                    y=self.simulated_paths[col],
                    mode='lines',
                    line=dict(color='rgba(0, 120, 220, 0.1)'),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
        
        # Plot original equity curve
        original_values = np.concatenate([[self.equity_values.iloc[0]], self.equity_values.values])
        
        # Determine the curve label based on workflow type
        curve_label = "Original Equity Curve"
        if hasattr(self, 'workflow_type') and self.workflow_type == 'complete':
            curve_label = "Optimized Equity Curve"
        
        fig.add_trace(
            go.Scatter(
                y=original_values,
                mode='lines',
                name=curve_label,
                line=dict(color='red', width=2)
            )
        )
        
        # Plot confidence interval
        lower_bound = self.simulated_paths.quantile((1 - self.confidence_level) / 2, axis=1)
        upper_bound = self.simulated_paths.quantile(1 - (1 - self.confidence_level) / 2, axis=1)
        median = self.simulated_paths.median(axis=1)
        
        # Add median path
        fig.add_trace(
            go.Scatter(
                y=median,
                mode='lines',
                name='Median Simulation',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add confidence interval as filled area
        fig.add_trace(
            go.Scatter(
                y=upper_bound,
                mode='lines',
                name=f'{self.confidence_level*100:.0f}% Confidence Interval',
                line=dict(width=0),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                y=lower_bound,
                mode='lines',
                name=f'{self.confidence_level*100:.0f}% Confidence Interval',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 100, 220, 0.2)'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'{self.strategy_name}: Monte Carlo Simulation Paths',
            xaxis_title='Trading Days',
            yaxis_title='Equity',
            yaxis_tickprefix='$',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Save or show the plot
        if save and self.output_dir:
            # Save HTML version
            html_path = os.path.join(self.output_dir, f"{self.strategy_name}_monte_carlo_paths.html")
            try:
                fig.write_html(html_path)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Error saving plot: {e}")
            
            return html_path
        else:
            fig.show()
            return None
    
    def plot_return_distribution(self, save: bool = True) -> Optional[str]:
        """
        Create an interactive visualization of the return distribution.
        
        Args:
            save: Whether to save the plot to disk
            
        Returns:
            Path to the saved plot if save=True, None otherwise
        """
        # Check if we have valid data
        if self.simulated_paths is None or len(self.simulated_paths.columns) == 0:
            import logging
            logging.getLogger(__name__).error("No simulation paths available for plotting return distribution")
            return None
        
        # Calculate returns for all simulations
        initial_equity = self.simulation_results['initial_equity']
        try:
            final_equity_values = np.array(self.simulated_paths.iloc[-1].values)
            returns = (final_equity_values / initial_equity) - 1
        except (IndexError, KeyError) as e:
            import logging
            logging.getLogger(__name__).error(f"Error calculating returns: {e}")
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Create histogram
        histogram = go.Histogram(
            x=returns,
            nbinsx=30,
            opacity=0.7,
            name='Return Distribution'
        )
        fig.add_trace(histogram)
        
        # Add key metrics as vertical lines
        original_return = self.simulation_results['return_original']
        mean_return = self.simulation_results['mean_return']
        var_pct = self.simulation_results['var_pct']
        
        # Calculate histogram's max y value by pre-computing the histogram data
        bin_values, bin_edges = np.histogram(returns, bins=30)
        max_frequency = max(bin_values) if len(bin_values) > 0 else 1
        y_range_max = max_frequency * 1.2  # Add 20% headroom
        
        # Add original return line
        fig.add_trace(
            go.Scatter(
                x=[original_return, original_return],
                y=[0, y_range_max],
                mode='lines',
                name=f'Original Return: {original_return:.2%}',
                line=dict(color='red', width=2, dash='dash')
            )
        )
        
        # Add mean return line
        fig.add_trace(
            go.Scatter(
                x=[mean_return, mean_return],
                y=[0, y_range_max],
                mode='lines',
                name=f'Mean Return: {mean_return:.2%}',
                line=dict(color='green', width=2, dash='dash')
            )
        )
        
        # Add VaR line
        fig.add_trace(
            go.Scatter(
                x=[var_pct, var_pct],
                y=[0, y_range_max],
                mode='lines',
                name=f'VaR ({self.confidence_level:.0%}): {var_pct:.2%}',
                line=dict(color='orange', width=2, dash='dash')
            )
        )
        
        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[0, y_range_max],
                mode='lines',
                name='Breakeven',
                line=dict(color='black', width=1)
            )
        )
        
        # Highlight probability of profit
        prob_profit = self.simulation_results['probability_of_profit']
        
        # Add annotation for probability of profit
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f'Probability of Profit: {prob_profit:.2%}',
            showarrow=False,
            font=dict(size=14),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
        
        # Update layout
        fig.update_layout(
            title=f'{self.strategy_name}: Return Distribution',
            xaxis_title='Return',
            yaxis_title='Frequency',
            xaxis_tickformat='.1%',
            hovermode='x unified',
            bargap=0.01,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            yaxis_range=[0, y_range_max]  # Set fixed y-axis range
        )
        
        # Save or show the plot
        if save and self.output_dir:
            # Save HTML version
            html_path = os.path.join(self.output_dir, f"{self.strategy_name}_return_distribution.html")
            try:
                fig.write_html(html_path)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Error saving plot: {e}")
            
            return html_path
        else:
            fig.show()
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
        Create an interactive visualization of drawdown analysis.
        
        Args:
            save: Whether to save the plot to disk
            
        Returns:
            Path to the saved plot if save=True, None otherwise
        """
        # Check if we have valid data
        if self.simulated_paths is None or len(self.simulated_paths.columns) == 0:
            import logging
            logging.getLogger(__name__).error("No simulation paths available for plotting drawdown analysis")
            return None
        
        # Calculate drawdowns for all simulations
        all_max_drawdowns = []
        
        try:
            for col in self.simulated_paths.columns:
                drawdowns = self._calculate_drawdowns(self.simulated_paths[col])
                all_max_drawdowns.append(drawdowns.min())
            
            # Calculate drawdown for original equity curve
            original_drawdowns = self._calculate_drawdowns(pd.Series(self.equity_values))
            original_max_drawdown = original_drawdowns.min()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error calculating drawdowns: {e}")
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Create histogram
        histogram = go.Histogram(
            x=all_max_drawdowns,
            nbinsx=30,
            opacity=0.7,
            name='Drawdown Distribution'
        )
        fig.add_trace(histogram)
        
        # Calculate statistics
        mean_max_drawdown = np.mean(all_max_drawdowns)
        median_max_drawdown = np.median(all_max_drawdowns)
        worst_drawdown = np.min(all_max_drawdowns)
        
        # Calculate histogram's max y value by pre-computing the histogram data
        bin_values, bin_edges = np.histogram(all_max_drawdowns, bins=30)
        max_frequency = max(bin_values) if len(bin_values) > 0 else 1
        y_range_max = max_frequency * 1.2  # Add 20% headroom
        
        # Add original max drawdown line
        fig.add_trace(
            go.Scatter(
                x=[original_max_drawdown, original_max_drawdown],
                y=[0, y_range_max],
                mode='lines',
                name=f'Original Max Drawdown: {original_max_drawdown:.2%}',
                line=dict(color='red', width=2, dash='dash')
            )
        )
        
        # Add mean max drawdown line
        fig.add_trace(
            go.Scatter(
                x=[mean_max_drawdown, mean_max_drawdown],
                y=[0, y_range_max],
                mode='lines',
                name=f'Mean Max Drawdown: {mean_max_drawdown:.2%}',
                line=dict(color='green', width=2, dash='dash')
            )
        )
        
        # Add worst drawdown line
        fig.add_trace(
            go.Scatter(
                x=[worst_drawdown, worst_drawdown],
                y=[0, y_range_max],
                mode='lines',
                name=f'Worst Drawdown: {worst_drawdown:.2%}',
                line=dict(color='orange', width=2, dash='dash')
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'{self.strategy_name}: Maximum Drawdown Distribution',
            xaxis_title='Maximum Drawdown',
            yaxis_title='Frequency',
            xaxis_tickformat='.1%',
            hovermode='x unified',
            bargap=0.01,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            yaxis_range=[0, y_range_max]  # Set fixed y-axis range
        )
        
        # Save or show the plot
        if save and self.output_dir:
            # Save HTML version
            html_path = os.path.join(self.output_dir, f"{self.strategy_name}_drawdown_analysis.html")
            try:
                fig.write_html(html_path)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Error saving plot: {e}")
            
            return html_path
        else:
            fig.show()
            return None
    
    def create_dashboard(self, save: bool = True) -> Optional[str]:
        """
        Create a comprehensive interactive dashboard of Monte Carlo visualization.
        
        Args:
            save: Whether to save the plot to disk
            
        Returns:
            Path to the saved plot if save=True, None otherwise
        """
        # Check if we have valid data
        if self.simulated_paths is None or len(self.simulated_paths.columns) == 0:
            import logging
            logging.getLogger(__name__).error("No simulation paths available for creating dashboard")
            return None
        
        # Create a figure with subplots - adjust layout for better screen fit
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"colspan": 2}, None],
                [{"type": "xy"}, {"type": "xy"}],
                [{"colspan": 2, "type": "table"}, None]
            ],
            row_heights=[0.45, 0.35, 0.20],  # Adjusted to give more space to the equity curve
            subplot_titles=(
                f'{self.strategy_name}: Monte Carlo Equity Curves ({self.simulation_results.get("num_simulations", 0)} simulations)',
                'Return Distribution', 'Maximum Drawdown Distribution',
                'Monte Carlo Simulation Statistics'
            )
        )
        
        try:
            # 1. Simulation Paths (top row, spans both columns)
            # Plot simulated paths (sample for clarity)
            sample_size = min(100, len(self.simulated_paths.columns))
            sample_cols = np.random.choice(self.simulated_paths.columns, sample_size, replace=False)
            
            for col in sample_cols:
                fig.add_trace(
                    go.Scatter(
                        y=self.simulated_paths[col],
                        mode='lines',
                        line=dict(color='rgba(0, 120, 220, 0.1)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
            
            # Plot original equity curve
            original_values = np.concatenate([[self.equity_values.iloc[0]], self.equity_values.values])
            
            # Determine the curve label based on workflow type
            curve_label = "Original Equity Curve"
            if hasattr(self, 'workflow_type') and self.workflow_type == 'complete':
                curve_label = "Optimized Equity Curve"
            
            fig.add_trace(
                go.Scatter(
                    y=original_values,
                    mode='lines',
                    name=curve_label,
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
            
            # Plot confidence interval
            lower_bound = self.simulated_paths.quantile((1 - self.confidence_level) / 2, axis=1)
            upper_bound = self.simulated_paths.quantile(1 - (1 - self.confidence_level) / 2, axis=1)
            median = self.simulated_paths.median(axis=1)
            
            # Add median path
            fig.add_trace(
                go.Scatter(
                    y=median,
                    mode='lines',
                    name='Median Simulation',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Add confidence interval as filled area
            fig.add_trace(
                go.Scatter(
                    y=upper_bound,
                    mode='lines',
                    name=f'{self.confidence_level*100:.0f}% Confidence Interval',
                    line=dict(width=0),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    y=lower_bound,
                    mode='lines',
                    name=f'{self.confidence_level*100:.0f}% Confidence Interval',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 100, 220, 0.2)'
                ),
                row=1, col=1
            )
            
            # 2. Return Distribution (middle row, left)
            # Calculate returns for all simulations
            initial_equity = self.simulation_results.get('initial_equity', 100000)
            final_equity_values = np.array(self.simulated_paths.iloc[-1].values)
            returns = (final_equity_values / initial_equity) - 1
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=30,
                    opacity=0.7,
                    name='Return Distribution'
                ),
                row=2, col=1
            )
            
            # Add key metrics as vertical lines
            original_return = self.simulation_results.get('return_original', 0)
            mean_return = self.simulation_results.get('mean_return', 0)
            var_pct = self.simulation_results.get('var_pct', 0)
            
            # Add original return line
            fig.add_trace(
                go.Scatter(
                    x=[original_return, original_return],
                    y=[0, 100],  # Will be scaled later
                    mode='lines',
                    name=f'Original: {original_return:.2%}',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=2, col=1
            )
            
            # Add mean return line
            fig.add_trace(
                go.Scatter(
                    x=[mean_return, mean_return],
                    y=[0, 100],  # Will be scaled later
                    mode='lines',
                    name=f'Mean: {mean_return:.2%}',
                    line=dict(color='green', width=2, dash='dash')
                ),
                row=2, col=1
            )
            
            # Add VaR line
            fig.add_trace(
                go.Scatter(
                    x=[var_pct, var_pct],
                    y=[0, 100],  # Will be scaled later
                    mode='lines',
                    name=f'VaR: {var_pct:.2%}',
                    line=dict(color='orange', width=2, dash='dash')
                ),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_trace(
                go.Scatter(
                    x=[0, 0],
                    y=[0, 100],  # Will be scaled later
                    mode='lines',
                    showlegend=False,
                    line=dict(color='black', width=1)
                ),
                row=2, col=1
            )
            
            # 3. Drawdown Analysis (middle row, right)
            # Calculate drawdowns for all simulations
            all_max_drawdowns = []
            
            for col in self.simulated_paths.columns:
                drawdowns = self._calculate_drawdowns(self.simulated_paths[col])
                all_max_drawdowns.append(drawdowns.min())
            
            # Calculate drawdown for original equity curve
            original_drawdowns = self._calculate_drawdowns(pd.Series(self.equity_values))
            original_max_drawdown = original_drawdowns.min()
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=all_max_drawdowns,
                    nbinsx=30,
                    opacity=0.7,
                    name='Drawdown Distribution'
                ),
                row=2, col=2
            )
            
            # Calculate statistics
            mean_max_drawdown = np.mean(all_max_drawdowns)
            worst_drawdown = np.min(all_max_drawdowns)
            
            # Add original max drawdown line
            fig.add_trace(
                go.Scatter(
                    x=[original_max_drawdown, original_max_drawdown],
                    y=[0, 100],  # Will be scaled later
                    mode='lines',
                    name=f'Original: {original_max_drawdown:.2%}',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=2, col=2
            )
            
            # Add mean max drawdown line
            fig.add_trace(
                go.Scatter(
                    x=[mean_max_drawdown, mean_max_drawdown],
                    y=[0, 100],  # Will be scaled later
                    mode='lines',
                    name=f'Mean: {mean_max_drawdown:.2%}',
                    line=dict(color='green', width=2, dash='dash')
                ),
                row=2, col=2
            )
            
            # Add worst drawdown line
            fig.add_trace(
                go.Scatter(
                    x=[worst_drawdown, worst_drawdown],
                    y=[0, 100],  # Will be scaled later
                    mode='lines',
                    name=f'Worst: {worst_drawdown:.2%}',
                    line=dict(color='orange', width=2, dash='dash')
                ),
                row=2, col=2
            )
            
            # 4. Key Statistics Summary (bottom row, spans both columns)
            # Prepare statistics text
            prob_profit = self.simulation_results.get('probability_of_profit', 0)
            
            # Create a table for statistics
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['Performance Metrics', 'Value', 'Risk Metrics', 'Value'],
                        align='left',
                        font=dict(size=12, color='white'),
                        fill_color='royalblue'
                    ),
                    cells=dict(
                        values=[
                            # Performance metrics column names
                            ['Initial Equity', 'Final Equity (Original)', 'Return (Original)', 
                             'Mean Final Equity', 'Mean Return', 'Probability of Profit'],
                            # Performance metrics values
                            [f"${self.simulation_results.get('initial_equity', 0):,.2f}",
                             f"${self.simulation_results.get('final_equity_original', 0):,.2f}",
                             f"{self.simulation_results.get('return_original', 0):.2%}",
                             f"${self.simulation_results.get('mean_final_equity', 0):,.2f}",
                             f"{self.simulation_results.get('mean_return', 0):.2%}",
                             f"{prob_profit:.2%}"],
                            # Risk metrics column names
                            [f"Value at Risk ({self.confidence_level:.0%})", 
                             f"Conditional VaR ({self.confidence_level:.0%})",
                             'Worst Return', 'Best Return', 
                             'Mean Max Drawdown', 'Worst Max Drawdown'],
                            # Risk metrics values
                            [f"{self.simulation_results.get('var_pct', 0):.2%}",
                             f"{self.simulation_results.get('cvar_pct', 0):.2%}",
                             f"{self.simulation_results.get('worst_return', 0):.2%}",
                             f"{self.simulation_results.get('best_return', 0):.2%}",
                             f"{mean_max_drawdown:.2%}",
                             f"{worst_drawdown:.2%}"]
                        ],
                        align='left',
                        font=dict(size=11),
                        fill_color=[['whitesmoke', 'white'] * 5]
                    )
                ),
                row=3, col=1
            )
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error creating dashboard: {e}")
            logger.exception("Full traceback:")
            return None
        
        # Update layout to fit better on screen without scrolling
        fig.update_layout(
            height=850,  # Increased height to accommodate legend
            width=1200,  # Standard width for most displays
            margin=dict(l=40, r=40, t=100, b=40),  # Increased top margin for legend
            showlegend=True,
            legend=dict(
                orientation="h",  # Horizontal orientation
                yanchor="bottom",
                y=1.02,  # Position above the chart
                xanchor="center",
                x=0.5,  # Center horizontally
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.3)',
                borderwidth=1,
                font=dict(size=10),  # Slightly smaller font
                tracegroupgap=10  # Reduce spacing between legend groups
            ),
            hovermode='closest'
        )
        
        # Update x-axis labels
        fig.update_xaxes(title_text='Trading Days', row=1, col=1)
        fig.update_xaxes(title_text='Return', tickformat='.1%', row=2, col=1)
        fig.update_xaxes(title_text='Maximum Drawdown', tickformat='.1%', row=2, col=2)
        
        # Update y-axis labels
        fig.update_yaxes(title_text='Equity', tickprefix='$', row=1, col=1)
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        fig.update_yaxes(title_text='Frequency', row=2, col=2)
        
        # Update subplot titles to be more concise
        fig.update_annotations(font_size=12)
        
        # Make legend items more concise for better display
        for trace in fig.data:
            if hasattr(trace, 'name'):
                if trace.name and "Original Equity Curve" in trace.name:
                    trace.name = "Original"
                elif trace.name and "Optimized Equity Curve" in trace.name:
                    trace.name = "Optimized"
                elif trace.name and "Median Simulation" in trace.name:
                    trace.name = "Median"
                elif trace.name and "Confidence Interval" in trace.name:
                    trace.name = f"{self.confidence_level*100:.0f}% Confidence"
                elif trace.name and "Original Max Drawdown" in trace.name:
                    trace.name = "Original DD"
                elif trace.name and "Mean Max Drawdown" in trace.name:
                    trace.name = "Mean DD"
                elif trace.name and "Worst Drawdown" in trace.name:
                    trace.name = "Worst DD"
                elif trace.name and "Original: " in trace.name:
                    # Keep the percentage value but with shorter label
                    pct_value = trace.name.split(": ")[1] if ": " in trace.name else ""
                    trace.name = f"Original: {pct_value}"
                elif trace.name and "Mean: " in trace.name:
                    # Keep the percentage value but with shorter label
                    pct_value = trace.name.split(": ")[1] if ": " in trace.name else ""
                    trace.name = f"Mean: {pct_value}"
                elif trace.name and "VaR: " in trace.name:
                    # Keep the percentage value but with shorter label
                    pct_value = trace.name.split(": ")[1] if ": " in trace.name else ""
                    trace.name = f"VaR: {pct_value}"
                elif trace.name and "Worst: " in trace.name:
                    # Keep the percentage value but with shorter label
                    pct_value = trace.name.split(": ")[1] if ": " in trace.name else ""
                    trace.name = f"Worst: {pct_value}"
                elif trace.name == "Return Distribution" or trace.name == "Drawdown Distribution":
                    # These can remain as is - they don't appear in the main legend
                    pass
        
        # Save or show the plot
        if save and self.output_dir:
            # Save HTML version
            html_path = os.path.join(self.output_dir, f"{self.strategy_name}_monte_carlo_dashboard.html")
            try:
                fig.write_html(html_path, config={'responsive': True})
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Error saving plot: {e}")
            
            return html_path
        else:
            fig.show()
            return None