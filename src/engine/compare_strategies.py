import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

def extract_strategy_name(directory_path):
    """Extract strategy name from directory path."""
    dir_name = os.path.basename(directory_path.rstrip('/'))
    print(f"Extracting strategy name from: {dir_name}")
    
    # For paths like "2025-03-01_15-58-13_MultiPosition_portfolio"
    # Try to extract the strategy name part
    if '_' in dir_name:
        # Split by underscore and look for the pattern
        parts = dir_name.split('_')
        if len(parts) >= 3:
            # Check if the first part looks like a date (YYYY-MM-DD)
            if len(parts[0]) == 10 and parts[0].count('-') == 2:
                # Check if the second part looks like a time (HH-MM-SS)
                if len(parts[1]) == 8 and parts[1].count('-') == 2:
                    # Everything after the timestamp is the strategy name
                    strategy_name = '_'.join(parts[2:])
                    print(f"Extracted strategy name: {strategy_name}")
                    return strategy_name
    
    # If we couldn't extract a name using the expected pattern,
    # use the directory name as a fallback
    print(f"Using directory name as strategy name: {dir_name}")
    return dir_name

def load_equity_curves(output_dirs):
    """Load equity curves from multiple strategy output directories."""
    equity_curves = {}
    
    for output_dir in output_dirs:
        equity_curve_path = os.path.join(output_dir, 'equity_curve.csv')
        if os.path.exists(equity_curve_path):
            strategy_name = extract_strategy_name(output_dir)
            print(f"Loading equity curve for strategy: {strategy_name}")
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
    
    if not equity_curves:
        print("No equity curves available for plotting.")
        return
        
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
        print(f"Strategies for plotting: {strategies}")
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
            ax.plot(dates, values, label=strategy, linewidth=2)
        
        # Customize the plot
        ax.set_title('Equity Curves Comparison (Normalized to 100)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value (Starting at 100)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
        plt.xticks(rotation=45)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Combined equity curves plot saved to {output_path}")
    else:
        print("No data available for plotting equity curves.")

def plot_metrics_comparison(metrics_df, output_dir):
    """Create bar charts comparing key metrics across strategies."""
    if metrics_df.empty:
        print("No metrics data available for comparison.")
        return
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Check if we have strategy names
    if 'Strategy' not in metrics_df.columns or metrics_df['Strategy'].isna().all():
        print("Warning: Strategy names are missing in metrics data. Using index as labels.")
        strategies = [f"Strategy {i+1}" for i in range(len(metrics_df))]
    else:
        strategies = metrics_df['Strategy'].tolist()
        
    print(f"Strategies for metrics comparison: {strategies}")
    
    # Plot Total Return
    plt.figure(figsize=(10, 6))
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

def plot_return_distributions(equity_curves, output_dir):
    """Create histograms and boxplots of returns for each strategy."""
    if not equity_curves:
        print("No equity curves available for return distribution analysis.")
        return
    
    # Set Seaborn style
    sns.set_style("whitegrid")
    
    # Calculate daily returns for each strategy
    returns_data = {}
    for strategy_name, equity_curve in equity_curves.items():
        # Calculate daily returns
        if 'Value' in equity_curve.columns:
            returns = equity_curve['Value'].pct_change().dropna() * 100  # Convert to percentage
            returns_data[strategy_name] = returns
    
    if not returns_data:
        print("No return data available for plotting distributions.")
        return
    
    # Create a DataFrame for combined plotting
    combined_returns = pd.DataFrame(returns_data)
    
    # 1. Histograms of returns - use separate plots for each strategy
    plt.figure(figsize=(12, 8))
    for strategy_name in combined_returns.columns:
        # Convert to numpy array to avoid pandas indexing issues
        returns_array = combined_returns[strategy_name].values
        sns.kdeplot(returns_array, label=strategy_name, fill=True, alpha=0.3)
    
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Daily Return (%)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'returns_distribution.png'))
    plt.close()
    print(f"Returns distribution plot saved to {os.path.join(output_dir, 'returns_distribution.png')}")
    
    # 2. Boxplot of returns
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=combined_returns)
    plt.title('Boxplot of Daily Returns')
    plt.xlabel('Strategy')
    plt.ylabel('Daily Return (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'returns_boxplot.png'))
    plt.close()
    print(f"Returns boxplot saved to {os.path.join(output_dir, 'returns_boxplot.png')}")
    
    # 3. Combined monthly returns plot
    try:
        # Process monthly returns for all strategies
        monthly_returns_data = {}
        
        for strategy_name, returns in returns_data.items():
            if isinstance(returns.index, pd.DatetimeIndex):
                # Resample to monthly returns
                monthly_returns = returns.resample('ME').apply(lambda x: (1 + x/100).prod() - 1) * 100
                monthly_returns_data[strategy_name] = monthly_returns
        
        if monthly_returns_data:
            # Create a combined DataFrame
            monthly_df = pd.DataFrame(monthly_returns_data)
            
            # Plot combined monthly returns
            plt.figure(figsize=(14, 8))
            
            # Set width for bars
            width = 0.35
            num_strategies = len(monthly_returns_data)
            if num_strategies > 1:
                width = 0.8 / num_strategies
            
            # Get x positions
            x = np.arange(len(monthly_df.index))
            
            # Plot bars for each strategy
            for i, (strategy_name, monthly_returns) in enumerate(monthly_returns_data.items()):
                offset = (i - (num_strategies-1)/2) * width
                plt.bar(x + offset, monthly_returns.values, width, 
                       label=strategy_name, 
                       alpha=0.7)
                
                # Removed data labels for cleaner visualization
            
            # Customize plot
            plt.title('Monthly Returns Comparison')
            plt.xlabel('Month')
            plt.ylabel('Return (%)')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.grid(axis='y', alpha=0.3)
            
            # Format x-axis with month names
            plt.xticks(x, [date.strftime('%Y-%m') for date in monthly_df.index], rotation=45)
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'combined_monthly_returns.png'))
            plt.close()
            print(f"Combined monthly returns plot saved")
    except Exception as e:
        print(f"Error creating combined monthly returns plot: {e}")
    
    # 4. Combined histogram plot
    plt.figure(figsize=(14, 8))
    
    # Create subplots for each strategy
    num_strategies = len(combined_returns.columns)
    colors = plt.cm.tab10(np.linspace(0, 1, num_strategies))
    
    for i, strategy_name in enumerate(combined_returns.columns):
        returns_array = combined_returns[strategy_name].values
        
        # Plot histogram with transparency
        plt.hist(returns_array, bins=30, alpha=0.5, 
                 label=strategy_name, color=colors[i], 
                 edgecolor='black', linewidth=0.5)
    
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Distribution of Daily Returns - All Strategies')
    plt.xlabel('Daily Return (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_returns_histogram.png'))
    plt.close()
    print(f"Combined histogram saved")
    
    # 5. Individual histograms for each strategy (keep these for detailed analysis)
    for strategy_name in combined_returns.columns:
        plt.figure(figsize=(10, 6))
        returns_array = combined_returns[strategy_name].values
        plt.hist(returns_array, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title(f'Distribution of Daily Returns - {strategy_name}')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'returns_histogram_{strategy_name}.png'))
        plt.close()
        print(f"Histogram for {strategy_name} saved")
        
    # 6. Add a drawdown plot for each strategy
    for strategy_name, equity_curve in equity_curves.items():
        try:
            if 'Value' in equity_curve.columns:
                # Calculate drawdown
                peak = equity_curve['Value'].cummax()
                drawdown = (equity_curve['Value'] - peak) / peak * 100
                
                # Plot drawdown
                plt.figure(figsize=(12, 6))
                drawdown.plot(color='red')
                plt.title(f'Drawdown (%) - {strategy_name}')
                plt.xlabel('Date')
                plt.ylabel('Drawdown (%)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'drawdown_{strategy_name}.png'))
                plt.close()
                print(f"Drawdown plot for {strategy_name} saved")
        except Exception as e:
            print(f"Error creating drawdown plot for {strategy_name}: {e}")
            
    # 7. Combined drawdown plot
    plt.figure(figsize=(14, 8))
    
    for strategy_name, equity_curve in equity_curves.items():
        try:
            if 'Value' in equity_curve.columns:
                # Calculate drawdown
                peak = equity_curve['Value'].cummax()
                drawdown = (equity_curve['Value'] - peak) / peak * 100
                
                # Plot drawdown
                drawdown.plot(label=strategy_name)
        except Exception as e:
            print(f"Error adding {strategy_name} to combined drawdown plot: {e}")
    
    plt.title('Drawdown Comparison (%)')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_drawdown.png'))
    plt.close()
    print(f"Combined drawdown plot saved")

def create_tearsheet(equity_curves, metrics_df, output_dir):
    """Create a comprehensive tearsheet with all plots in a single PDF."""
    pdf_path = os.path.join(output_dir, 'strategy_comparison_tearsheet.pdf')
    
    with PdfPages(pdf_path) as pdf:
        # Set style for all plots
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 9,
            'axes.titlesize': 11,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 14
        })
        
        # 1. First page: Title and Performance Metrics Table
        fig = plt.figure(figsize=(11, 8.5))  # US Letter size
        fig.suptitle('Trading Strategy Comparison Tearsheet', fontsize=16, y=0.98)
        
        # Add timestamp
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.95, 0.02, f'Generated: {timestamp}', fontsize=8, ha='right')
        
        # Create a table with metrics
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Convert metrics DataFrame to a format suitable for table
        table_data = []
        headers = ['Metric'] + metrics_df['Strategy'].tolist()
        table_data.append(headers)
        
        metrics_to_display = [
            'Total Return (%)', 
            'Annualized Return (%)', 
            'Annualized Volatility (%)', 
            'Sharpe Ratio', 
            'Maximum Drawdown (%)',
            'Benchmark Return (%)'
        ]
        
        for metric in metrics_to_display:
            if metric in metrics_df.columns:
                row = [metric]
                for _, strategy_row in metrics_df.iterrows():
                    if pd.notna(strategy_row.get(metric, np.nan)):
                        row.append(f"{strategy_row[metric]:.2f}")
                    else:
                        row.append("N/A")
                table_data.append(row)
        
        # Create the table
        table = ax.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc='center',
            cellLoc='center',
            colColours=['#f2f2f2'] + ['#e6f3ff'] * len(metrics_df)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Add description
        fig.text(0.5, 0.15, 'This tearsheet provides a comprehensive comparison of trading strategies, including performance metrics,\nequity curves, return distributions, and drawdowns.', 
                 ha='center', fontsize=10)
        
        pdf.savefig(fig)
        plt.close(fig)
        
        # 2. Second page: Equity Curves and Monthly Returns
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Equity Curves and Monthly Returns', fontsize=14, y=0.98)
        
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4)
        
        # Equity Curves
        ax1 = fig.add_subplot(gs[0])
        
        # Create a DataFrame for plotting equity curves
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
            
            # Plot each strategy separately
            for strategy in strategies:
                strategy_data = resampled_df[resampled_df['Strategy'] == strategy]
                dates = strategy_data['Date'].to_numpy()
                values = strategy_data['Value'].to_numpy()
                ax1.plot(dates, values, label=strategy, linewidth=1.5)
            
            # Customize the plot
            ax1.set_title('Equity Curves Comparison (Normalized to 100)')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Value (Starting at 100)')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax1.tick_params(axis='x', rotation=45, labelsize=8)
            ax1.legend(loc='best', fontsize=8)
            ax1.grid(True, alpha=0.3)
        
        # Monthly Returns
        ax2 = fig.add_subplot(gs[1])
        
        # Calculate daily returns for each strategy
        returns_data = {}
        for strategy_name, equity_curve in equity_curves.items():
            if 'Value' in equity_curve.columns:
                returns = equity_curve['Value'].pct_change().dropna() * 100
                returns_data[strategy_name] = returns
        
        # Process monthly returns for all strategies
        monthly_returns_data = {}
        
        for strategy_name, returns in returns_data.items():
            if isinstance(returns.index, pd.DatetimeIndex):
                monthly_returns = returns.resample('ME').apply(lambda x: (1 + x/100).prod() - 1) * 100
                monthly_returns_data[strategy_name] = monthly_returns
        
        if monthly_returns_data:
            # Create a combined DataFrame
            monthly_df = pd.DataFrame(monthly_returns_data)
            
            # Set width for bars
            width = 0.35
            num_strategies = len(monthly_returns_data)
            if num_strategies > 1:
                width = 0.8 / num_strategies
            
            # Get x positions
            x = np.arange(len(monthly_df.index))
            
            # Plot bars for each strategy
            for i, (strategy_name, monthly_returns) in enumerate(monthly_returns_data.items()):
                offset = (i - (num_strategies-1)/2) * width
                ax2.bar(x + offset, monthly_returns.values, width, 
                       label=strategy_name, 
                       alpha=0.7)
            
            # Customize plot
            ax2.set_title('Monthly Returns Comparison')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Return (%)')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(axis='y', alpha=0.3)
            
            # Format x-axis with month names - use fewer labels if there are many months
            if len(x) > 12:
                # Show every other month if there are many
                show_indices = np.arange(0, len(x), 2)
                ax2.set_xticks(show_indices)
                ax2.set_xticklabels([date.strftime('%Y-%m') for date in monthly_df.index[show_indices]], rotation=45, fontsize=7)
            else:
                ax2.set_xticks(x)
                ax2.set_xticklabels([date.strftime('%Y-%m') for date in monthly_df.index], rotation=45, fontsize=8)
            
            ax2.legend(loc='best', fontsize=8)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
        
        # 3. Third page: Return Distributions
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Return Distributions', fontsize=14, y=0.98)
        
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.4, wspace=0.3)
        
        # KDE Plot
        ax1 = fig.add_subplot(gs[0, 0])
        
        combined_returns = pd.DataFrame(returns_data)
        
        for strategy_name in combined_returns.columns:
            returns_array = combined_returns[strategy_name].values
            sns.kdeplot(returns_array, label=strategy_name, fill=True, alpha=0.3, ax=ax1)
        
        ax1.set_title('Distribution of Daily Returns')
        ax1.set_xlabel('Daily Return (%)')
        ax1.set_ylabel('Density')
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=8)
        
        # Boxplot
        ax2 = fig.add_subplot(gs[0, 1])
        
        sns.boxplot(data=combined_returns, ax=ax2)
        ax2.set_title('Boxplot of Daily Returns')
        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Daily Return (%)')
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Combined Histogram
        ax3 = fig.add_subplot(gs[1, :])
        
        num_strategies = len(combined_returns.columns)
        colors = plt.cm.tab10(np.linspace(0, 1, num_strategies))
        
        for i, strategy_name in enumerate(combined_returns.columns):
            returns_array = combined_returns[strategy_name].values
            ax3.hist(returns_array, bins=30, alpha=0.5, 
                    label=strategy_name, color=colors[i], 
                    edgecolor='black', linewidth=0.5)
        
        ax3.axvline(x=0, color='red', linestyle='--')
        ax3.set_title('Distribution of Daily Returns - All Strategies')
        ax3.set_xlabel('Daily Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        ax3.tick_params(labelsize=8)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
        
        # 4. Fourth page: Drawdowns
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Drawdown Analysis', fontsize=14, y=0.98)
        
        gs = gridspec.GridSpec(len(equity_curves) + 1, 1, height_ratios=[1] * (len(equity_curves) + 1), hspace=0.4)
        
        # Combined Drawdown plot
        ax1 = fig.add_subplot(gs[0])
        
        for strategy_name, equity_curve in equity_curves.items():
            if 'Value' in equity_curve.columns:
                peak = equity_curve['Value'].cummax()
                drawdown = (equity_curve['Value'] - peak) / peak * 100
                drawdown.plot(label=strategy_name, ax=ax1, linewidth=1.5)
        
        ax1.set_title('Drawdown Comparison (%)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Drawdown (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        ax1.tick_params(labelsize=8)
        
        # Individual Drawdown plots
        for i, (strategy_name, equity_curve) in enumerate(equity_curves.items(), 1):
            if i < len(equity_curves) + 1:  # Ensure we don't exceed grid size
                ax = fig.add_subplot(gs[i])
                
                if 'Value' in equity_curve.columns:
                    peak = equity_curve['Value'].cummax()
                    drawdown = (equity_curve['Value'] - peak) / peak * 100
                    drawdown.plot(color='red', ax=ax, linewidth=1.5)
                
                ax.set_title(f'Drawdown (%) - {strategy_name}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Drawdown (%)')
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"Comprehensive tearsheet saved to {pdf_path}")
    return pdf_path

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
    
    # Plot return distributions
    print("Plotting return distributions...")
    plot_return_distributions(equity_curves, comparison_output_dir)
    
    # Create comprehensive tearsheet
    print("Creating comprehensive tearsheet...")
    create_tearsheet(equity_curves, metrics_df, comparison_output_dir)
    
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