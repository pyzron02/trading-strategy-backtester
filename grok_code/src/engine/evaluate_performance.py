import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import calendar
from matplotlib.colors import LinearSegmentedColormap

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
    
    # Calculate Sortino Ratio (using only negative returns for downside risk)
    downside_returns = daily_returns[daily_returns < 0]
    downside_deviation = downside_returns.std() * (252 ** 0.5) if not downside_returns.empty else np.nan
    sortino_ratio = annualized_return / downside_deviation if downside_deviation != 0 else np.nan
    
    cum_max = equity_curve['Value'].cummax()
    drawdown = (cum_max - equity_curve['Value']) / cum_max
    max_drawdown = drawdown.max() if not drawdown.empty else np.nan
    
    # Calculate Calmar Ratio (annualized return / maximum drawdown)
    calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else np.nan
    
    # Calculate Maximum Drawdown Period
    if not drawdown.empty:
        # Find the peak before the deepest trough
        i = drawdown.idxmax()
        # Find the last time the equity curve was at a peak before the drawdown
        j = drawdown[:i][::-1].idxmin()
        # Find the recovery point (when drawdown returns to 0 after the trough)
        try:
            k = drawdown[i:][drawdown[i:] <= 0.0001].index[0]
        except IndexError:
            # If no recovery point is found, use the last date
            k = drawdown.index[-1]
        
        max_dd_start = j
        max_dd_end = i
        max_dd_recovery = k
        max_dd_length = (max_dd_end - max_dd_start).days
        max_dd_recovery_length = (max_dd_recovery - max_dd_end).days if max_dd_recovery != max_dd_end else np.nan
        max_dd_total_length = (max_dd_recovery - max_dd_start).days if max_dd_recovery != max_dd_start else np.nan
    else:
        max_dd_start = np.nan
        max_dd_end = np.nan
        max_dd_recovery = np.nan
        max_dd_length = np.nan
        max_dd_recovery_length = np.nan
        max_dd_total_length = np.nan
    
    # Load trade log for trade metrics
    trade_log_path = os.path.join(results_dir, 'trade_log.csv')
    trade_metrics = {}
    if os.path.exists(trade_log_path):
        trade_log = pd.read_csv(trade_log_path)
        
        # Calculate trade metrics
        closed_trades = trade_log[trade_log['type'] == 'close']
        if not closed_trades.empty:
            total_trades = len(closed_trades)
            winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
            losing_trades = len(closed_trades[closed_trades['pnl'] <= 0])
            
            # Win Rate
            win_rate = winning_trades / total_trades if total_trades > 0 else np.nan
            
            # Profit Factor
            gross_profit = closed_trades[closed_trades['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(closed_trades[closed_trades['pnl'] <= 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
            
            # Average Win/Loss Ratio
            avg_win = gross_profit / winning_trades if winning_trades > 0 else np.nan
            avg_loss = gross_loss / losing_trades if losing_trades > 0 else np.nan
            avg_win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else np.inf
            
            # Calculate Maximum Consecutive Wins/Losses
            # First, sort trades by date
            closed_trades_sorted = closed_trades.sort_values('date')
            # Create a series of 1 (win) and 0 (loss)
            win_loss_series = (closed_trades_sorted['pnl'] > 0).astype(int)
            
            # Calculate consecutive wins
            win_streaks = []
            current_streak = 0
            for win in win_loss_series:
                if win == 1:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        win_streaks.append(current_streak)
                    current_streak = 0
            if current_streak > 0:
                win_streaks.append(current_streak)
            
            # Calculate consecutive losses
            loss_streaks = []
            current_streak = 0
            for win in win_loss_series:
                if win == 0:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        loss_streaks.append(current_streak)
                    current_streak = 0
            if current_streak > 0:
                loss_streaks.append(current_streak)
            
            max_consecutive_wins = max(win_streaks) if win_streaks else 0
            max_consecutive_losses = max(loss_streaks) if loss_streaks else 0
            
            # Calculate average holding period
            # First, we need to match open and close trades
            open_trades = trade_log[trade_log['type'] == 'open']
            
            # This is a simplified approach - in a real system, you'd need more sophisticated matching
            # based on trade IDs or other identifiers
            trade_metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_win_loss_ratio': avg_win_loss_ratio,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses
            }
    
    # Calculate Monthly Returns Table
    monthly_returns = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    monthly_returns_table = pd.DataFrame(monthly_returns)
    monthly_returns_table.index = pd.MultiIndex.from_arrays([
        monthly_returns_table.index.year,
        monthly_returns_table.index.month
    ], names=['Year', 'Month'])
    monthly_returns_table = monthly_returns_table.unstack('Month')
    monthly_returns_table.columns = monthly_returns_table.columns.droplevel()
    monthly_returns_table.columns = [calendar.month_abbr[m] for m in monthly_returns_table.columns]
    
    # Add annual returns
    annual_returns = daily_returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
    annual_returns.index = annual_returns.index.year
    monthly_returns_table['Annual'] = annual_returns
    
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
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    
    # Print max drawdown period info if available
    if max_dd_start is not None and not pd.isna(max_dd_start) and max_dd_end is not None and not pd.isna(max_dd_end):
        print(f"Maximum Drawdown Period: {max_dd_length} days (from {max_dd_start.strftime('%Y-%m-%d')} to {max_dd_end.strftime('%Y-%m-%d')})")
        if max_dd_recovery is not None and not pd.isna(max_dd_recovery) and max_dd_recovery != max_dd_end:
            print(f"Recovery Period: {max_dd_recovery_length} days (until {max_dd_recovery.strftime('%Y-%m-%d')})")
    
    if trade_metrics:
        print("\nTrade Metrics:")
        print(f"Total Trades: {trade_metrics['total_trades']}")
        print(f"Win Rate: {trade_metrics['win_rate']:.2%}")
        print(f"Profit Factor: {trade_metrics['profit_factor']:.2f}")
        print(f"Average Win: ${trade_metrics['avg_win']:.2f}")
        print(f"Average Loss: ${trade_metrics['avg_loss']:.2f}")
        print(f"Average Win/Loss Ratio: {trade_metrics['avg_win_loss_ratio']:.2f}")
        print(f"Maximum Consecutive Wins: {trade_metrics['max_consecutive_wins']}")
        print(f"Maximum Consecutive Losses: {trade_metrics['max_consecutive_losses']}")
    
    print("\nBenchmark Metrics (S&P 500):")
    print(f"Benchmark Total Return: {benchmark_total_return:.2%}")
    
    # Print Monthly Returns Table
    print("\nMonthly Returns:")
    print(monthly_returns_table.to_string(float_format=lambda x: f"{x:.2%}" if not np.isnan(x) else "N/A"))
    
    # Save metrics to a text file for easier comparison later
    results_txt_path = os.path.join(results_dir, 'results.txt')
    with open(results_txt_path, 'w') as f:
        f.write("Performance Metrics:\n")
        f.write(f"Total Return: {total_return*100:.2f}%\n")
        f.write(f"Annualized Return: {annualized_return*100:.2f}%\n")
        f.write(f"Annualized Volatility: {annualized_volatility*100:.2f}%\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
        f.write(f"Sortino Ratio: {sortino_ratio:.2f}\n")
        f.write(f"Calmar Ratio: {calmar_ratio:.2f}\n")
        f.write(f"Maximum Drawdown: {max_drawdown*100:.2f}%\n")
        
        # Write max drawdown period info if available
        if max_dd_start is not None and not pd.isna(max_dd_start) and max_dd_end is not None and not pd.isna(max_dd_end):
            f.write(f"Maximum Drawdown Period: {max_dd_length} days (from {max_dd_start.strftime('%Y-%m-%d')} to {max_dd_end.strftime('%Y-%m-%d')})\n")
            if max_dd_recovery is not None and not pd.isna(max_dd_recovery) and max_dd_recovery != max_dd_end:
                f.write(f"Recovery Period: {max_dd_recovery_length} days (until {max_dd_recovery.strftime('%Y-%m-%d')})\n")
        
        if trade_metrics:
            f.write("\nTrade Metrics:\n")
            f.write(f"Total Trades: {trade_metrics['total_trades']}\n")
            f.write(f"Winning Trades: {trade_metrics['winning_trades']} ({trade_metrics['win_rate']*100:.2f}%)\n")
            f.write(f"Losing Trades: {trade_metrics['losing_trades']} ({(1-trade_metrics['win_rate'])*100:.2f}%)\n")
            f.write(f"Profit Factor: {trade_metrics['profit_factor']:.2f}\n")
            f.write(f"Average Win: ${trade_metrics['avg_win']:.2f}\n")
            f.write(f"Average Loss: ${trade_metrics['avg_loss']:.2f}\n")
            f.write(f"Average Win/Loss Ratio: {trade_metrics['avg_win_loss_ratio']:.2f}\n")
            f.write(f"Gross Profit: ${trade_metrics['gross_profit']:.2f}\n")
            f.write(f"Gross Loss: ${trade_metrics['gross_loss']:.2f}\n")
            f.write(f"Maximum Consecutive Wins: {trade_metrics['max_consecutive_wins']}\n")
            f.write(f"Maximum Consecutive Losses: {trade_metrics['max_consecutive_losses']}\n")
        
        f.write("\nBenchmark Metrics (S&P 500):\n")
        f.write(f"Benchmark Total Return: {benchmark_total_return*100:.2f}%\n")
        f.write(f"\nBacktest Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
        
        f.write("\nMonthly Returns:\n")
        f.write(monthly_returns_table.to_string(float_format=lambda x: f"{x*100:.2f}%" if not np.isnan(x) else "N/A"))
    
    plot_equity_curve(equity_curve, sp500_equity, results_dir)
    plot_drawdown(drawdown, sp500_drawdown, results_dir)
    plot_returns_histogram(daily_returns, sp500_returns, results_dir)
    plot_monthly_returns_heatmap(monthly_returns_table, results_dir)
    
    if trade_metrics:
        plot_win_loss_distribution(trade_metrics, results_dir)
        plot_underwater_chart(drawdown, results_dir, max_dd_start, max_dd_end, max_dd_recovery)

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

def plot_win_loss_distribution(trade_metrics, results_dir):
    """Plot win/loss distribution as a pie chart."""
    # Create a pie chart of winning vs losing trades
    plt.figure(figsize=(8, 8))
    labels = [f"Winning Trades\n{trade_metrics['winning_trades']} ({trade_metrics['win_rate']:.1%})", 
              f"Losing Trades\n{trade_metrics['losing_trades']} ({1-trade_metrics['win_rate']:.1%})"]
    sizes = [trade_metrics['winning_trades'], trade_metrics['losing_trades']]
    colors = ['#4CAF50', '#F44336']  # Green for wins, red for losses
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'})
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Win/Loss Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'win_loss_distribution.png'))
    plt.close()

def plot_monthly_returns_heatmap(monthly_returns_table, results_dir):
    """Plot monthly returns as a heatmap."""
    # Drop the Annual column for the heatmap
    monthly_data = monthly_returns_table.drop('Annual', axis=1).copy()
    
    # Create a custom colormap (green for positive, red for negative)
    cmap = LinearSegmentedColormap.from_list('rg', ["#F44336", "#FFFFFF", "#4CAF50"], N=256)
    
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(monthly_data, annot=True, cmap=cmap, center=0,
                     fmt=".1%", linewidths=.5, cbar_kws={"shrink": .8})
    
    # Customize the plot
    plt.title('Monthly Returns Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'monthly_returns_heatmap.png'))
    plt.close()

def plot_underwater_chart(drawdown, results_dir, max_dd_start, max_dd_end, max_dd_recovery):
    """Plot underwater chart (enhanced drawdown visualization)."""
    plt.figure(figsize=(12, 6))
    
    # Convert drawdown to percentage for better visualization
    underwater = drawdown * -100
    
    # Convert pandas DatetimeIndex to numpy arrays before plotting
    dates = underwater.index.to_numpy()
    values = underwater.to_numpy()
    
    # Plot the underwater chart
    plt.fill_between(dates, values, 0, color='red', alpha=0.3)
    plt.plot(dates, values, color='red', linewidth=1)
    
    # Highlight the maximum drawdown period if available
    if (max_dd_start is not None and not pd.isna(max_dd_start) and 
        max_dd_end is not None and not pd.isna(max_dd_end)):
        plt.axvspan(max_dd_start, max_dd_end, color='red', alpha=0.2)
        
        # Add recovery period if available
        if max_dd_recovery is not None and not pd.isna(max_dd_recovery) and max_dd_recovery != max_dd_end:
            plt.axvspan(max_dd_end, max_dd_recovery, color='orange', alpha=0.2)
            
        # Add annotations
        max_dd_value = drawdown.loc[max_dd_end] * -100
        plt.annotate(f'Max DD: {max_dd_value:.1f}%', 
                     xy=(max_dd_end, max_dd_value),
                     xytext=(10, -30),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
    
    # Customize the plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Underwater Chart (Drawdown)')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'underwater_chart.png'))
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_performance.py <path_to_backtest_results.pkl>")
        sys.exit(1)
    results_path = sys.argv[1]
    evaluate_performance(results_path)