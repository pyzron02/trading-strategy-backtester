"""
Reporting utilities for the trading strategy backtester.
Provides standardized reporting functions for all workflows.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Union, Optional

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

def print_header(title: str) -> None:
    """Print a formatted header for console output.
    
    Args:
        title: The title to display in the header
    """
    print("\n" + "=" * 80)
    print(f"{title.center(80)}")
    print("=" * 80 + "\n")

def print_section(title: str) -> None:
    """Print a formatted section header for console output.
    
    Args:
        title: The title to display in the section header
    """
    print("\n" + "-" * 80)
    print(f"{title}")
    print("-" * 80)

def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary into a readable string.
    
    Args:
        metrics: Dictionary of metric names and values
        
    Returns:
        Formatted string of metrics
    """
    result = []
    for key, value in metrics.items():
        # Format numeric values
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                # Format percentage values
                if key.endswith("_pct") or key.endswith("_rate") or "percent" in key or "ratio" in key:
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
        else:
            formatted_value = str(value)
            
        # Convert key from snake_case to Title Case for display
        display_key = " ".join(word.capitalize() for word in key.split("_"))
        result.append(f"{display_key}: {formatted_value}")
    
    return "\n".join(result)

def save_metrics(metrics: Dict[str, Any], file_path: str) -> None:
    """Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        file_path: Path to save the metrics JSON file
    """
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def save_summary_report(
    summary: Dict[str, Any], 
    file_path: str,
    title: str = "Backtest Results Summary",
    include_timestamp: bool = True
) -> None:
    """Save a formatted summary report to a text file.
    
    Args:
        summary: Dictionary containing summary sections and their content
        file_path: Path to save the summary report
        title: Title for the summary report
        include_timestamp: Whether to include a timestamp in the report
    """
    with open(file_path, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write(f"{title.center(80)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Add timestamp if requested
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Generated: {timestamp}\n\n")
        
        # Write each section
        for section_title, section_content in summary.items():
            f.write("-" * 80 + "\n")
            f.write(f"{section_title}\n")
            f.write("-" * 80 + "\n\n")
            
            if isinstance(section_content, dict):
                # Format and write dictionary content
                f.write(format_metrics(section_content) + "\n\n")
            elif isinstance(section_content, (list, tuple)):
                # Format and write list content
                for item in section_content:
                    if isinstance(item, dict):
                        f.write(format_metrics(item) + "\n\n")
                    else:
                        f.write(str(item) + "\n")
                f.write("\n")
            else:
                # Write string content
                f.write(str(section_content) + "\n\n")

def plot_equity_curve(
    equity_curve: pd.DataFrame,
    output_path: str,
    title: str = "Equity Curve",
    include_drawdown: bool = True
) -> None:
    """Plot and save equity curve with optional drawdown subplot.
    
    Args:
        equity_curve: DataFrame with equity curve data
        output_path: Path to save the plot
        title: Title for the plot
        include_drawdown: Whether to include drawdown subplot
    """
    fig = plt.figure(figsize=(12, 8 if include_drawdown else 6))
    
    if include_drawdown:
        # Create two subplots
        ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2, sharex=ax1)
        
        # Plot equity curve
        equity_curve['equity'].plot(ax=ax1, linewidth=2)
        ax1.set_title(title)
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True)
        
        # Plot drawdown
        (-equity_curve['drawdown']).plot(ax=ax2, linewidth=1.5, color='red', alpha=0.7)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True)
        ax2.set_xlabel('Date')
        
        # Format drawdown as percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
    else:
        # Plot just equity curve
        equity_curve['equity'].plot(linewidth=2)
        plt.title(title)
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.xlabel('Date')
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_monthly_returns_heatmap(
    monthly_returns: pd.DataFrame,
    output_path: str,
    title: str = "Monthly Returns Heatmap"
) -> None:
    """Plot and save monthly returns heatmap.
    
    Args:
        monthly_returns: DataFrame with monthly returns
        output_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(14, 8))
    
    # Create the heatmap
    ax = sns.heatmap(
        monthly_returns,
        annot=True,
        fmt=".2%",
        cmap=sns.diverging_palette(10, 220, as_cmap=True),
        center=0,
        vmin=-max(abs(monthly_returns.min().min()), abs(monthly_returns.max().max())),
        vmax=max(abs(monthly_returns.min().min()), abs(monthly_returns.max().max())),
        linewidths=0.5
    )
    
    plt.title(title)
    plt.ylabel('Year')
    plt.xlabel('Month')
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_tearsheet(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    output_dir: str,
    prefix: str = ""
) -> None:
    """Create a comprehensive performance tearsheet with multiple plots.
    
    Args:
        equity_curve: DataFrame with equity curve data
        trade_log: DataFrame with trade log data
        output_dir: Directory to save the tearsheet plots
        prefix: Prefix for file names
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Equity curve with drawdown
    plot_equity_curve(
        equity_curve,
        os.path.join(output_dir, f"{prefix}equity_curve.png"),
        "Equity Curve with Drawdown",
        include_drawdown=True
    )
    
    # 2. Monthly returns heatmap
    if 'monthly_returns' in equity_curve:
        # If monthly returns are already calculated
        monthly_returns = equity_curve['monthly_returns'].unstack()
    else:
        # Calculate monthly returns from equity curve
        returns = equity_curve['equity'].pct_change().fillna(0)
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns = monthly_returns.to_frame('return')
        monthly_returns['year'] = monthly_returns.index.year
        monthly_returns['month'] = monthly_returns.index.month
        monthly_returns = monthly_returns.pivot(index='year', columns='month', values='return')
    
    plot_monthly_returns_heatmap(
        monthly_returns,
        os.path.join(output_dir, f"{prefix}monthly_returns.png"),
        "Monthly Returns Heatmap"
    )
    
    # 3. Trade analysis plots (if trade_log is not empty)
    if not trade_log.empty:
        # Trade P&L distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(trade_log['pnl_pct'], kde=True, bins=20)
        plt.title('Trade P&L Distribution')
        plt.xlabel('P&L %')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.savefig(os.path.join(output_dir, f"{prefix}pnl_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Trade P&L over time
        plt.figure(figsize=(12, 6))
        plt.bar(
            trade_log.index, 
            trade_log['pnl_pct'],
            color=trade_log['pnl_pct'].apply(lambda x: 'green' if x > 0 else 'red')
        )
        plt.title('Trade P&L Over Time')
        plt.xlabel('Trade Number')
        plt.ylabel('P&L %')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{prefix}pnl_over_time.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Trade duration analysis
        if 'duration' in trade_log.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(trade_log['duration'], kde=True, bins=20)
            plt.title('Trade Duration Distribution')
            plt.xlabel('Duration (days)')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, f"{prefix}duration_distribution.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Rolling performance metrics (if equity_curve has enough data)
    if len(equity_curve) > 60:  # At least 60 data points for rolling metrics
        returns = equity_curve['equity'].pct_change().fillna(0)
        
        # Rolling Sharpe ratio (annualized, 60-day window)
        rolling_sharpe = returns.rolling(60).apply(
            lambda x: x.mean() / x.std() * (252**0.5) if x.std() != 0 else 0
        )
        
        plt.figure(figsize=(12, 6))
        rolling_sharpe.plot(linewidth=1.5)
        plt.title('Rolling 60-Day Sharpe Ratio')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{prefix}rolling_sharpe.png"), dpi=300, bbox_inches='tight')
        plt.close() 