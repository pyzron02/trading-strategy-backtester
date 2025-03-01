import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pathlib import Path

def extract_strategy_name(directory_path):
    """Extract strategy name from directory path."""
    dir_name = os.path.basename(directory_path)
    # Extract strategy name after timestamp (format: YYYY-MM-DD_HH-MM-SS_StrategyName)
    parts = dir_name.split('_', 2)
    if len(parts) >= 3:
        return parts[2]
    return dir_name  # Fallback to directory name if pattern doesn't match

def load_equity_curves(output_dirs):
    """Load equity curves from multiple strategy output directories."""
    equity_curves = {}
    
    for output_dir in output_dirs:
        equity_curve_path = os.path.join(output_dir, 'equity_curve.csv')
        if os.path.exists(equity_curve_path):
            strategy_name = extract_strategy_name(output_dir)
            equity_curve = pd.read_csv(equity_curve_path)
            equity_curve['Date'] = pd.to_datetime(equity_curve['Date'])
            equity_curve.set_index('Date', inplace=True)
            equity_curves[strategy_name] = equity_curve
    
    return equity_curves

def load_performance_metrics(output_dirs):
    """Extract performance metrics from backtest results files."""
    metrics = []
    
    for output_dir in output_dirs:
        strategy_name = extract_strategy_name(output_dir)
        print(f"\nProcessing strategy: {strategy_name}")
        
        # Try to find metrics in a results.txt file if it exists
        results_file = os.path.join(output_dir, 'results.txt')
        if os.path.exists(results_file):
            print(f"Found results file: {results_file}")
            with open(results_file, 'r') as f:
                content = f.read()
                print(f"Results file content length: {len(content)} characters")
                
                # Extract metrics using simple parsing
                total_return = extract_metric(content, "Total Return:", "%")
                annual_return = extract_metric(content, "Annualized Return:", "%")
                volatility = extract_metric(content, "Annualized Volatility:", "%")
                sharpe = extract_metric(content, "Sharpe Ratio:", None)
                max_dd = extract_metric(content, "Maximum Drawdown:", "%")
                benchmark_return = extract_metric(content, "Benchmark Total Return:", "%")
                
                metrics.append({
                    'Strategy': strategy_name,
                    'Total Return (%)': total_return,
                    'Annualized Return (%)': annual_return,
                    'Annualized Volatility (%)': volatility,
                    'Sharpe Ratio': sharpe,
                    'Maximum Drawdown (%)': max_dd,
                    'Benchmark Return (%)': benchmark_return
                })
        else:
            print(f"No results.txt file found in {output_dir}")
            # If no results file, try to recreate metrics from equity curve
            equity_curve_path = os.path.join(output_dir, 'equity_curve.csv')
            if os.path.exists(equity_curve_path):
                print(f"Using equity curve to calculate metrics: {equity_curve_path}")
                equity_curve = pd.read_csv(equity_curve_path)
                equity_curve['Date'] = pd.to_datetime(equity_curve['Date'])
                
                # Calculate basic metrics
                initial_value = equity_curve['Value'].iloc[0]
                final_value = equity_curve['Value'].iloc[-1]
                total_return = ((final_value / initial_value) - 1) * 100
                
                start_date = equity_curve['Date'].min()
                end_date = equity_curve['Date'].max()
                years = (end_date - start_date).days / 365.25
                
                annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else np.nan
                
                metrics.append({
                    'Strategy': strategy_name,
                    'Total Return (%)': total_return,
                    'Annualized Return (%)': annual_return,
                    'Annualized Volatility (%)': np.nan,
                    'Sharpe Ratio': np.nan,
                    'Maximum Drawdown (%)': np.nan,
                    'Benchmark Return (%)': np.nan
                })
            else:
                print(f"No equity curve found in {output_dir}")
    
    return pd.DataFrame(metrics)

def extract_metric(content, label, suffix=None):
    """Extract a numeric metric from text content."""
    try:
        start_idx = content.find(label)
        if start_idx == -1:
            print(f"Warning: Could not find '{label}' in results file")
            return np.nan
            
        start_idx += len(label)
        end_idx = content.find('\n', start_idx)
        if end_idx == -1:
            value_str = content[start_idx:].strip()
        else:
            value_str = content[start_idx:end_idx].strip()
        
        # Debug the extracted string
        print(f"Extracted '{label}': '{value_str}'")
            
        # Remove percentage sign if present
        if suffix and value_str.endswith(suffix):
            value_str = value_str[:-len(suffix)]
        
        # Handle percentage values (e.g., "63.30%")
        if "%" in value_str:
            value_str = value_str.replace("%", "")
            
        return float(value_str)
    except Exception as e:
        print(f"Error extracting '{label}': {e}")
        return np.nan

def plot_combined_equity_curves(equity_curves, output_path):
    """Plot combined equity curves for multiple strategies."""
    # Set Seaborn style for better aesthetics
    sns.set_style("whitegrid")
    
    # Create a DataFrame for plotting
    combined_data = []
    
    for strategy_name, equity_curve in equity_curves.items():
        # Normalize to starting at 100
        normalized = equity_curve['Value'] / equity_curve['Value'].iloc[0] * 100
        df = pd.DataFrame({
            'Date': equity_curve.index,
            'Value': normalized,
            'Strategy': strategy_name
        })
        combined_data.append(df)
    
    if combined_data:
        plot_df = pd.concat(combined_data)
        
        # Resample to weekly for cleaner plots
        strategies = plot_df['Strategy'].unique()
        resampled_data = []
        
        for strategy in strategies:
            strategy_data = plot_df[plot_df['Strategy'] == strategy].copy()
            strategy_data.set_index('Date', inplace=True)
            weekly = strategy_data.resample('W').last()
            weekly['Strategy'] = strategy
            weekly_reset = weekly.reset_index()
            resampled_data.append(weekly_reset)
        
        resampled_df = pd.concat(resampled_data)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each strategy separately using numpy arrays
        for strategy in strategies:
            strategy_data = resampled_df[resampled_df['Strategy'] == strategy]
            dates = strategy_data['Date'].to_numpy()
            values = strategy_data['Value'].to_numpy()
            ax.plot(dates, values, label=strategy)
        
        # Customize the plot
        ax.set_title('Equity Curves Comparison (Normalized to 100)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value (Starting at 100)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def plot_metrics_comparison(metrics_df, output_dir):
    """Create bar charts comparing key metrics across strategies."""
    if metrics_df.empty:
        print("No metrics data available for comparison.")
        return
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Plot Total Return
    plt.figure(figsize=(10, 6))
    strategies = metrics_df['Strategy'].tolist()
    values = metrics_df['Total Return (%)'].tolist()
    plt.bar(strategies, values, color='skyblue')
    plt.title('Total Return Comparison')
    plt.xlabel('Strategy')
    plt.ylabel('Total Return (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_return_comparison.png'))
    plt.close()
    
    # Plot Annualized Return
    plt.figure(figsize=(10, 6))
    values = metrics_df['Annualized Return (%)'].tolist()
    plt.bar(strategies, values, color='lightgreen')
    plt.title('Annualized Return Comparison')
    plt.xlabel('Strategy')
    plt.ylabel('Annualized Return (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'annual_return_comparison.png'))
    plt.close()
    
    # Plot Sharpe Ratio
    if not metrics_df['Sharpe Ratio'].isna().all():
        plt.figure(figsize=(10, 6))
        values = metrics_df['Sharpe Ratio'].tolist()
        plt.bar(strategies, values, color='coral')
        plt.title('Sharpe Ratio Comparison')
        plt.xlabel('Strategy')
        plt.ylabel('Sharpe Ratio')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sharpe_ratio_comparison.png'))
        plt.close()
    
    # Plot Maximum Drawdown
    if not metrics_df['Maximum Drawdown (%)'].isna().all():
        plt.figure(figsize=(10, 6))
        values = metrics_df['Maximum Drawdown (%)'].tolist()
        plt.bar(strategies, values, color='salmon')
        plt.title('Maximum Drawdown Comparison')
        plt.xlabel('Strategy')
        plt.ylabel('Maximum Drawdown (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'max_drawdown_comparison.png'))
        plt.close()

def compare_strategies(output_dirs, comparison_output_dir):
    """Compare multiple strategy backtest results."""
    # Create output directory if it doesn't exist
    os.makedirs(comparison_output_dir, exist_ok=True)
    
    # Load equity curves
    print("Loading equity curves...")
    equity_curves = load_equity_curves(output_dirs)
    
    # Load performance metrics
    print("Loading performance metrics...")
    metrics_df = load_performance_metrics(output_dirs)
    
    # Save metrics to CSV
    metrics_csv_path = os.path.join(comparison_output_dir, 'strategy_metrics_comparison.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to {metrics_csv_path}")
    
    # Plot combined equity curves
    print("Plotting combined equity curves...")
    equity_curves_path = os.path.join(comparison_output_dir, 'combined_equity_curves.png')
    plot_combined_equity_curves(equity_curves, equity_curves_path)
    
    # Plot metrics comparison
    print("Plotting metrics comparison...")
    plot_metrics_comparison(metrics_df, comparison_output_dir)
    
    print(f"Strategy comparison completed. Results saved to {comparison_output_dir}")
    
    # Print metrics table
    print("\nStrategy Performance Comparison:")
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_strategies.py <comparison_output_dir> <strategy_output_dir1> <strategy_output_dir2> ...")
        sys.exit(1)
    
    comparison_output_dir = sys.argv[1]
    strategy_dirs = sys.argv[2:]
    
    compare_strategies(strategy_dirs, comparison_output_dir) 