#!/usr/bin/env python3
# auction_market_visualization.py - Visualize Auction Market Theory analysis

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
from datetime import datetime, timedelta

class AuctionMarketVisualizer:
    """Visualization tools for Auction Market Theory analysis."""
    
    def __init__(self, results_dir=None):
        """
        Initialize the visualizer with results directory.
        
        Args:
            results_dir: Directory containing backtest results
        """
        self.results_dir = results_dir
        self.equity_curve = None
        self.trade_log = None
        self.results_summary = None
        
        # Load data if results_dir is provided
        if results_dir and os.path.exists(results_dir):
            self._load_data()
    
    def _load_data(self):
        """Load data from results directory."""
        # Load equity curve
        equity_curve_path = os.path.join(self.results_dir, 'equity_curve.csv')
        if os.path.exists(equity_curve_path):
            self.equity_curve = pd.read_csv(equity_curve_path)
            self.equity_curve['Date'] = pd.to_datetime(self.equity_curve['Date'])
        
        # Load trade log
        trade_log_path = os.path.join(self.results_dir, 'trade_log.csv')
        if os.path.exists(trade_log_path):
            self.trade_log = pd.read_csv(trade_log_path)
            if 'date' in self.trade_log.columns:
                self.trade_log['date'] = pd.to_datetime(self.trade_log['date'])
        
        # Load results summary
        results_path = os.path.join(self.results_dir, 'results.txt')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                self.results_summary = f.read()
    
    def plot_equity_curve(self, save_path=None):
        """
        Plot equity curve.
        
        Args:
            save_path: Path to save the plot
        """
        if self.equity_curve is None:
            print("No equity curve data available.")
            return
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.equity_curve['Date'], self.equity_curve['Value'])
        
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Equity curve plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_trade_distribution(self, save_path=None):
        """
        Plot distribution of trade results.
        
        Args:
            save_path: Path to save the plot
        """
        if self.trade_log is None:
            print("No trade log data available.")
            return
        
        # Filter for closed trades
        closed_trades = self.trade_log[self.trade_log['type'] == 'close']
        
        if len(closed_trades) == 0:
            print("No closed trades found.")
            return
        
        plt.figure(figsize=(12, 6))
        
        plt.hist(closed_trades['pnl'], bins=20, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        
        plt.title('Distribution of Trade P&L')
        plt.xlabel('P&L ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Trade distribution plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_trade_pnl_by_symbol(self, save_path=None):
        """
        Plot P&L by symbol.
        
        Args:
            save_path: Path to save the plot
        """
        if self.trade_log is None:
            print("No trade log data available.")
            return
        
        # Filter for closed trades
        closed_trades = self.trade_log[self.trade_log['type'] == 'close']
        
        if len(closed_trades) == 0:
            print("No closed trades found.")
            return
        
        # Group by symbol
        symbol_pnl = closed_trades.groupby('symbol')['pnl'].sum()
        
        plt.figure(figsize=(12, 6))
        
        symbol_pnl.plot(kind='bar')
        
        plt.title('Total P&L by Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('P&L ($)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"P&L by symbol plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_trade_pnl_over_time(self, save_path=None):
        """
        Plot P&L over time.
        
        Args:
            save_path: Path to save the plot
        """
        if self.trade_log is None:
            print("No trade log data available.")
            return
        
        # Filter for closed trades
        closed_trades = self.trade_log[self.trade_log['type'] == 'close']
        
        if len(closed_trades) == 0:
            print("No closed trades found.")
            return
        
        # Sort by date
        closed_trades = closed_trades.sort_values('date')
        
        # Calculate cumulative P&L
        closed_trades['cumulative_pnl'] = closed_trades['pnl'].cumsum()
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(pd.to_datetime(closed_trades['date']), closed_trades['cumulative_pnl'])
        
        plt.title('Cumulative P&L Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative P&L ($)')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()
        
        if save_path:
            plt.savefig(save_path)
            print(f"P&L over time plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_drawdown(self, save_path=None):
        """
        Plot drawdown.
        
        Args:
            save_path: Path to save the plot
        """
        if self.equity_curve is None:
            print("No equity curve data available.")
            return
        
        # Calculate drawdown
        equity_curve = self.equity_curve.copy()
        equity_curve['Peak'] = equity_curve['Value'].cummax()
        equity_curve['Drawdown'] = (equity_curve['Value'] - equity_curve['Peak']) / equity_curve['Peak'] * 100
        
        plt.figure(figsize=(12, 6))
        
        plt.fill_between(equity_curve['Date'], equity_curve['Drawdown'], 0, alpha=0.3, color='r')
        plt.plot(equity_curve['Date'], equity_curve['Drawdown'], color='r')
        
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Drawdown plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_monthly_returns(self, save_path=None):
        """
        Plot monthly returns.
        
        Args:
            save_path: Path to save the plot
        """
        if self.equity_curve is None:
            print("No equity curve data available.")
            return
        
        # Calculate daily returns
        equity_curve = self.equity_curve.copy()
        equity_curve['Return'] = equity_curve['Value'].pct_change()
        
        # Group by month and calculate monthly returns
        equity_curve['Year'] = equity_curve['Date'].dt.year
        equity_curve['Month'] = equity_curve['Date'].dt.month
        monthly_returns = equity_curve.groupby(['Year', 'Month'])['Return'].apply(
            lambda x: (1 + x).prod() - 1
        ).reset_index()
        
        # Create month-year labels
        monthly_returns['MonthYear'] = monthly_returns.apply(
            lambda row: f"{row['Year']}-{row['Month']:02d}", axis=1
        )
        
        plt.figure(figsize=(14, 6))
        
        plt.bar(monthly_returns['MonthYear'], monthly_returns['Return'] * 100)
        
        plt.title('Monthly Returns')
        plt.xlabel('Month')
        plt.ylabel('Return (%)')
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=90)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Monthly returns plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_trade_duration_vs_pnl(self, save_path=None):
        """
        Plot trade duration vs P&L.
        
        Args:
            save_path: Path to save the plot
        """
        if self.trade_log is None:
            print("No trade log data available.")
            return
        
        # Filter for closed trades
        closed_trades = self.trade_log[self.trade_log['type'] == 'close']
        
        if len(closed_trades) == 0:
            print("No closed trades found.")
            return
        
        # Group by symbol and date to match open and close trades
        open_trades = self.trade_log[self.trade_log['type'] == 'open']
        
        # Merge open and close trades
        trades = pd.merge(
            open_trades, 
            closed_trades,
            on=['symbol'],
            suffixes=('_open', '_close')
        )
        
        # Calculate trade duration in days
        trades['duration'] = (pd.to_datetime(trades['date_close']) - pd.to_datetime(trades['date_open'])).dt.days
        
        plt.figure(figsize=(12, 6))
        
        plt.scatter(trades['duration'], trades['pnl'], alpha=0.7)
        
        plt.title('Trade Duration vs P&L')
        plt.xlabel('Duration (days)')
        plt.ylabel('P&L ($)')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(trades) > 1:
            z = np.polyfit(trades['duration'], trades['pnl'], 1)
            p = np.poly1d(z)
            plt.plot(trades['duration'], p(trades['duration']), "r--", alpha=0.7)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Trade duration vs P&L plot saved to {save_path}")
        else:
            plt.show()
    
    def create_all_plots(self, output_dir=None):
        """
        Create all plots and save them to output directory.
        
        Args:
            output_dir: Directory to save plots
        """
        if output_dir is None and self.results_dir is not None:
            output_dir = os.path.join(self.results_dir, 'plots')
        
        if output_dir is None:
            print("No output directory specified.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create all plots
        self.plot_equity_curve(save_path=os.path.join(output_dir, 'equity_curve.png'))
        self.plot_trade_distribution(save_path=os.path.join(output_dir, 'trade_distribution.png'))
        self.plot_trade_pnl_by_symbol(save_path=os.path.join(output_dir, 'pnl_by_symbol.png'))
        self.plot_trade_pnl_over_time(save_path=os.path.join(output_dir, 'pnl_over_time.png'))
        self.plot_drawdown(save_path=os.path.join(output_dir, 'drawdown.png'))
        self.plot_monthly_returns(save_path=os.path.join(output_dir, 'monthly_returns.png'))
        self.plot_trade_duration_vs_pnl(save_path=os.path.join(output_dir, 'duration_vs_pnl.png'))
        
        print(f"All plots saved to {output_dir}")
    
    def print_summary(self):
        """Print summary of backtest results."""
        if self.results_summary is not None:
            print("\n" + "="*50)
            print("BACKTEST RESULTS SUMMARY")
            print("="*50)
            print(self.results_summary)
            print("="*50)
        else:
            print("No results summary available.")
        
        if self.trade_log is not None:
            # Calculate trade statistics
            closed_trades = self.trade_log[self.trade_log['type'] == 'close']
            
            if len(closed_trades) > 0:
                total_trades = len(closed_trades)
                winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
                losing_trades = len(closed_trades[closed_trades['pnl'] <= 0])
                win_rate = winning_trades / total_trades * 100
                
                total_profit = closed_trades[closed_trades['pnl'] > 0]['pnl'].sum()
                total_loss = closed_trades[closed_trades['pnl'] <= 0]['pnl'].sum()
                
                avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
                avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
                
                profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
                
                print("\n" + "="*50)
                print("TRADE STATISTICS")
                print("="*50)
                print(f"Total Trades: {total_trades}")
                print(f"Winning Trades: {winning_trades} ({win_rate:.2f}%)")
                print(f"Losing Trades: {losing_trades} ({100 - win_rate:.2f}%)")
                print(f"Total Profit: ${total_profit:.2f}")
                print(f"Total Loss: ${total_loss:.2f}")
                print(f"Net Profit: ${total_profit + total_loss:.2f}")
                print(f"Average Profit: ${avg_profit:.2f}")
                print(f"Average Loss: ${avg_loss:.2f}")
                print(f"Profit Factor: {profit_factor:.2f}")
                print("="*50)
            else:
                print("No closed trades found.")

def main():
    """Main function to run the visualization."""
    parser = argparse.ArgumentParser(description="Visualize Auction Market Theory backtest results.")
    parser.add_argument('--results_dir', type=str, required=True,
                        help="Directory containing backtest results")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Directory to save plots (default: results_dir/plots)")
    parser.add_argument('--show_summary', action='store_true',
                        help="Print summary of backtest results")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = AuctionMarketVisualizer(results_dir=args.results_dir)
    
    # Create plots
    visualizer.create_all_plots(output_dir=args.output_dir)
    
    # Print summary
    if args.show_summary:
        visualizer.print_summary()

if __name__ == "__main__":
    main() 