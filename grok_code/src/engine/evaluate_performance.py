import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Set Seaborn theme for consistent styling
sns.set_theme(style="whitegrid")

def evaluate_performance(results_path):
    """
    Evaluate backtest performance, compare with S&P 500 benchmark, and generate plots using Seaborn.
    
    Args:
        results_path (str): Path to the backtest_results.pkl file.
    """
    results_dir = os.path.dirname(os.path.abspath(results_path))
    if not os.path.exists(results_path):
        print(f"Backtest results file not found at {results_path}")
        return

    equity_curve_path = os.path.join(results_dir, 'equity_curve.csv')
    if not os.path.exists(equity_curve_path):
        print("Equity curve file not found.")
        return
    equity_curve = pd.read_csv(equity_curve_path)
    equity_curve['Date'] = pd.to_datetime(equity_curve['Date'])
    equity_curve.set_index('Date', inplace=True)
    
    daily_returns = equity_curve['Value'].pct_change().dropna()
    
    initial_value = equity_curve['Value'].iloc[0]
    final_value = equity_curve['Value'].iloc[-1]
    total_return = (final_value - initial_value) / initial_value
    
    start_date = equity_curve.index.min()
    end_date = equity_curve.index.max()
    num_years = (end_date - start_date).days / 365.25
    
    annualized_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else np.nan
    annualized_volatility = daily_returns.std() * (252 ** 0.5) if not daily_returns.empty else np.nan
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan
    
    cum_max = equity_curve['Value'].cummax()
    drawdown = (cum_max - equity_curve['Value']) / cum_max
    max_drawdown = drawdown.max() if not drawdown.empty else np.nan
    
    stock_data_path = 'input/stock_data.csv'
    if not os.path.exists(stock_data_path):
        print("Stock data file not found. Skipping benchmark metrics.")
        benchmark_total_return = np.nan
        sp500_equity = pd.Series()
        sp500_drawdown = pd.Series()
        sp500_returns = pd.Series()
    else:
        stock_data = pd.read_csv(stock_data_path)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)
        if 'SP500_Close' in stock_data.columns:
            sp500_data = stock_data['SP500_Close'].loc[start_date:end_date]
            if not sp500_data.empty:
                benchmark_total_return = (sp500_data.iloc[-1] / sp500_data.iloc[0]) - 1
                sp500_equity = (sp500_data / sp500_data.iloc[0]) * initial_value
                sp500_cum_max = sp500_equity.cummax()
                sp500_drawdown = (sp500_cum_max - sp500_equity) / sp500_cum_max
                sp500_returns = sp500_data.pct_change().dropna()
            else:
                print("Warning: S&P 500 data does not cover the equity curve date range.")
                benchmark_total_return = np.nan
                sp500_equity = pd.Series()
                sp500_drawdown = pd.Series()
                sp500_returns = pd.Series()
        else:
            print("Warning: 'SP500_Close' column not found in stock_data.csv.")
            benchmark_total_return = np.nan
            sp500_equity = pd.Series()
            sp500_drawdown = pd.Series()
            sp500_returns = pd.Series()
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Volatility: {annualized_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print("\nBenchmark Metrics (S&P 500):")
    print(f"Benchmark Total Return: {benchmark_total_return:.2%}")
    
    # Save metrics to a text file for easier comparison later
    results_txt_path = os.path.join(results_dir, 'results.txt')
    with open(results_txt_path, 'w') as f:
        f.write("Performance Metrics:\n")
        f.write(f"Total Return: {total_return*100:.2f}%\n")
        f.write(f"Annualized Return: {annualized_return*100:.2f}%\n")
        f.write(f"Annualized Volatility: {annualized_volatility*100:.2f}%\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
        f.write(f"Maximum Drawdown: {max_drawdown*100:.2f}%\n")
        f.write("\nBenchmark Metrics (S&P 500):\n")
        f.write(f"Benchmark Total Return: {benchmark_total_return*100:.2f}%\n")
        f.write(f"\nBacktest Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
    
    plot_equity_curve(equity_curve, sp500_equity, results_dir)
    plot_drawdown(drawdown, sp500_drawdown, results_dir)
    plot_returns_histogram(daily_returns, sp500_returns, results_dir)

def plot_equity_curve(equity_curve, sp500_equity, results_dir):
    """Plot equity curves using Seaborn styling with Matplotlib plotting."""
    # Set Seaborn style for better aesthetics
    sns.set_style("whitegrid")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Resample strategy equity curve to weekly frequency
    strategy_weekly = equity_curve.resample('W').last()
    strategy_data = strategy_weekly.reset_index()
    dates_strategy = strategy_data['Date'].to_numpy()
    values_strategy = strategy_data['Value'].to_numpy()
    ax.plot(dates_strategy, values_strategy, label='Strategy Portfolio', linestyle='-')
    
    # Plot S&P 500 equity curve if available
    if not sp500_equity.empty:
        # Resample S&P 500 equity curve to weekly frequency
        sp500_weekly = sp500_equity.resample('W').last()
        sp500_data = sp500_weekly.reset_index()
        sp500_data.columns = ['Date', 'Value']
        dates_sp500 = sp500_data['Date'].to_numpy()
        values_sp500 = sp500_data['Value'].to_numpy()
        ax.plot(dates_sp500, values_sp500, label='S&P 500', linestyle='--')
    
    # Customize the plot
    ax.set_title('Equity Curve (Weekly)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value')
    
    # Set weekly date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # Show every other week
    
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'equity_curve.png'))
    plt.close()

def plot_drawdown(drawdown, sp500_drawdown, results_dir):
    """Plot drawdown using Seaborn styling with Matplotlib plotting."""
    # Set Seaborn style for better aesthetics
    sns.set_style("whitegrid")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Resample strategy drawdown to weekly frequency
    drawdown_weekly = drawdown.resample('W').max()  # Use max for drawdown to show worst case
    strategy_data = drawdown_weekly.reset_index()
    strategy_data.columns = ['Date', 'Drawdown']
    dates_strategy = strategy_data['Date'].to_numpy()
    values_strategy = strategy_data['Drawdown'].to_numpy() * 100  # Convert to percentage
    ax.plot(dates_strategy, values_strategy, label='Strategy Drawdown', color='red', linestyle='-')
    
    # Plot S&P 500 drawdown if available
    if not sp500_drawdown.empty:
        # Resample S&P 500 drawdown to weekly frequency
        sp500_drawdown_weekly = sp500_drawdown.resample('W').max()  # Use max for drawdown
        sp500_data = sp500_drawdown_weekly.reset_index()
        sp500_data.columns = ['Date', 'Drawdown']
        dates_sp500 = sp500_data['Date'].to_numpy()
        values_sp500 = sp500_data['Drawdown'].to_numpy() * 100  # Convert to percentage
        ax.plot(dates_sp500, values_sp500, label='S&P 500 Drawdown', color='orange', linestyle='--')
    
    # Customize the plot
    ax.set_title('Drawdown (Weekly)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    
    # Set weekly date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # Show every other week
    
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'drawdown.png'))
    plt.close()

def plot_returns_histogram(daily_returns, sp500_returns, results_dir):
    """Plot overlaid histograms of daily returns using Seaborn."""
    # Create a DataFrame for plotting
    plot_data = []
    
    # Add strategy returns
    strategy_data = pd.DataFrame({'Returns': daily_returns.to_numpy(), 'Source': 'Strategy'})
    plot_data.append(strategy_data)
    
    # Add S&P 500 returns if available
    if not sp500_returns.empty:
        sp500_data = pd.DataFrame({'Returns': sp500_returns.to_numpy(), 'Source': 'S&P 500'})
        plot_data.append(sp500_data)
    
    # Combine data for plotting
    plot_df = pd.concat(plot_data, ignore_index=True)
    
    # Create the plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    palette = {'Strategy': 'blue', 'S&P 500': 'orange'}
    sns.histplot(data=plot_df, x='Returns', hue='Source', stat='density',
                 alpha=0.5, bins=50, palette=palette)

    # Customize the plot
    plt.title('Daily Returns Distribution')
    plt.xlabel('Daily Return')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'returns_histogram.png'))
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_performance.py <path_to_backtest_results.pkl>")
        sys.exit(1)
    results_path = sys.argv[1]
    evaluate_performance(results_path)