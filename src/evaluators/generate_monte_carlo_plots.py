#!/usr/bin/env python3
# generate_monte_carlo_plots.py - Simplified script to generate Monte Carlo stock path plots

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import random
import sys

# Set base directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)  # src directory
PROJECT_ROOT = os.path.dirname(SRC_DIR)  # project root
DATA_DIR = os.path.join(PROJECT_ROOT, 'input')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

def load_stock_data(csv_path=None):
    """Load stock data from CSV file."""
    if csv_path is None:
        # Try a few possible locations
        possible_paths = [
            os.path.join(DATA_DIR, 'stock_data.csv'),
            os.path.join(PROJECT_ROOT, 'input', 'stock_data.csv'),
            os.path.join(SRC_DIR, 'data_preprocessing', 'input', 'stock_data.csv')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            print(f"Stock data file not found in any of: {', '.join(possible_paths)}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Directory contents: {os.listdir('.')}")
            
            # Try to find stock_data.csv anywhere in the project
            found_files = []
            for root, dirs, files in os.walk(os.path.dirname(os.path.dirname(PROJECT_ROOT))):
                if 'stock_data.csv' in files:
                    found_files.append(os.path.join(root, 'stock_data.csv'))
            
            if found_files:
                print(f"Found potential stock data files: {found_files}")
                csv_path = found_files[0]
                print(f"Using: {csv_path}")
            else:
                raise FileNotFoundError(f"Stock data file not found in any location")
    
    print(f"Loading stock data from {csv_path}...")
    
    try:
        data = pd.read_csv(csv_path)
        
        # Convert Date column to datetime
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
        
        print(f"Loaded stock data with {len(data)} rows and {len(data.columns)} columns.")
        print(f"Columns: {', '.join(data.columns)}")
        return data
    except Exception as e:
        print(f"Error loading stock data: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_simple_plot(output_dir=None):
    """Create a simple test plot to verify matplotlib is working."""
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'monte_carlo_plots')
    
    print(f"Creating simple test plot in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple plot
    plt.figure(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    plt.plot(x, np.sin(x), 'r-', label='Sine wave')
    plt.plot(x, np.cos(x), 'b--', label='Cosine wave')
    plt.title('Simple Test Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    test_plot_path = os.path.join(output_dir, 'test_plot.png')
    try:
        plt.savefig(test_plot_path)
        print(f"Test plot saved to {test_plot_path}")
    except Exception as e:
        print(f"Error saving test plot: {e}")
    
    plt.close()

def create_permuted_data(data, ticker, out_sample_start, out_sample_end, random_seed=42):
    """Create permuted data by shuffling returns within the out-of-sample period."""
    # Make a copy of the original data
    permuted_data = data.copy()
    
    # Filter for out-of-sample period
    out_sample_mask = (permuted_data.index >= out_sample_start) & (permuted_data.index <= out_sample_end)
    out_sample_data = permuted_data[out_sample_mask]
    
    # Get price column for this ticker
    close_col = f"{ticker}_Close"
    
    if close_col in out_sample_data.columns:
        # Calculate returns
        returns = out_sample_data[close_col].pct_change().dropna()
        
        # Shuffle returns
        np.random.seed(random_seed)
        shuffled_returns = np.random.permutation(returns.values)
        
        # Reconstruct prices from shuffled returns
        initial_price = out_sample_data[close_col].iloc[0]
        shuffled_prices = [initial_price]
        
        for ret in shuffled_returns:
            next_price = shuffled_prices[-1] * (1 + ret)
            shuffled_prices.append(next_price)
        
        # Replace prices in the permuted data
        permuted_data.loc[out_sample_mask, close_col] = shuffled_prices[:len(out_sample_data)]
    
    return permuted_data

def save_data_to_csv(original_data, ticker, out_sample_start, out_sample_end, num_simulations=5, output_dir=None):
    """
    Save the actual stock prices and simulated paths to CSV files for external plotting.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'monte_carlo_data')
    
    print(f"Saving data to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for out-of-sample period
    out_sample_mask = (original_data.index >= out_sample_start) & (original_data.index <= out_sample_end)
    original_out_sample = original_data[out_sample_mask]
    
    close_col = f"{ticker}_Close"
    
    if close_col not in original_data.columns:
        print(f"Error: {close_col} not found in data columns.")
        print(f"Available columns: {original_data.columns.tolist()}")
        return None
    
    # Check if we have enough data
    if len(original_out_sample) < 5:
        print(f"Error: Not enough data points for {ticker} in the specified date range.")
        return None
    
    # Create dataframe for combined data
    combined_data = pd.DataFrame(index=original_out_sample.index)
    
    # Ensure we have valid price data (no NaNs)
    if original_out_sample[close_col].isna().any():
        print(f"Warning: {ticker} price data contains NaN values. Filling with forward fill method.")
        original_out_sample[close_col] = original_out_sample[close_col].fillna(method='ffill')
        # If still have NaNs at the beginning, fill with backward fill
        original_out_sample[close_col] = original_out_sample[close_col].fillna(method='bfill')
    
    # Add actual price data
    combined_data['actual'] = original_out_sample[close_col]
    print(f"Added actual price data for {ticker}: {len(combined_data)} rows, first price: {combined_data['actual'].iloc[0] if not combined_data.empty else 'N/A'}")
    
    # Generate simulated paths
    successful_sims = 0
    for i in range(num_simulations):
        try:
            # Create permuted data
            permuted_data = create_permuted_data(original_data, ticker, out_sample_start, out_sample_end, random_seed=42+i)
            
            # Filter for out-of-sample period
            permuted_out_sample = permuted_data[out_sample_mask]
            
            # Validate the permuted data
            if not permuted_out_sample.empty and close_col in permuted_out_sample.columns:
                # Check for NaN values
                if permuted_out_sample[close_col].isna().any():
                    print(f"Warning: Simulated path {i} for {ticker} contains NaN values. Fixing...")
                    permuted_out_sample[close_col] = permuted_out_sample[close_col].fillna(method='ffill').fillna(method='bfill')
                
                # Add to combined data
                combined_data[f'sim_{i}'] = permuted_out_sample[close_col]
                successful_sims += 1
            else:
                print(f"Warning: Simulated path {i} for {ticker} produced invalid data.")
        except Exception as e:
            print(f"Error generating simulation {i} for {ticker}: {e}")
    
    # Check if we have any successful simulations
    if successful_sims == 0:
        print(f"Error: No valid simulations could be generated for {ticker}")
        return None
    
    print(f"Successfully generated {successful_sims} simulations for {ticker}")
    
    # Verify combined data is not empty
    if combined_data.empty:
        print(f"Error: Combined data for {ticker} is empty")
        return None
    
    # Print data summary
    print(f"Data summary for {ticker}:")
    print(f"  Rows: {len(combined_data)}")
    print(f"  Columns: {combined_data.columns.tolist()}")
    print(f"  Date range: {combined_data.index.min()} to {combined_data.index.max()}")
    print(f"  Actual price range: {combined_data['actual'].min():.2f} to {combined_data['actual'].max():.2f}")
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f'{ticker}_price_paths.csv')
    combined_data.to_csv(csv_path)
    print(f"Price data for {ticker} saved to {csv_path}")
    
    return combined_data

def plot_simulated_stock_prices(data, ticker, output_dir=None):
    """
    Plot the actual stock prices versus simulated paths from data frame.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'monte_carlo_plots')
    
    print(f"Generating plot in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Verify data is not empty
        if data.empty:
            print(f"Error: Data for {ticker} is empty. Cannot generate plot.")
            return False
        
        # Verify 'actual' column exists and contains valid data
        if 'actual' not in data.columns:
            print(f"Error: 'actual' column not found in data for {ticker}")
            return False
        
        if data['actual'].isna().all():
            print(f"Error: 'actual' column for {ticker} contains only NaN values")
            return False
        
        # Create figure with high quality settings
        plt.figure(figsize=(12, 7), dpi=100)
        
        # Format dates for x-axis
        dates = data.index.to_numpy()
        
        # Count simulation columns
        sim_columns = [col for col in data.columns if col.startswith('sim_')]
        print(f"Found {len(sim_columns)} simulation columns for {ticker}")
        
        # Plot simulated paths
        for col in sim_columns:
            # Verify simulation data is valid
            if data[col].isna().all():
                print(f"Warning: Simulation column {col} contains only NaN values. Skipping.")
                continue
                
            # Convert data to numpy array for plotting
            prices = data[col].to_numpy()
            plt.plot(dates, prices, 
                    alpha=0.3, linewidth=0.8, color='gray', label='_nolegend_')
        
        # Plot actual path in red with higher visibility
        actual_prices = data['actual'].to_numpy()
        plt.plot(dates, actual_prices, 
                color='red', linewidth=2.5, label=f'Actual {ticker}')
        
        # Add legend with clear background
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        
        # Format the x-axis to show dates properly
        plt.gcf().autofmt_xdate()
        
        # Set plot attributes
        plt.title(f'{ticker} Stock Price with Monte Carlo Simulations', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add annotation for context
        plt.figtext(0.02, 0.02, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                    fontsize=8, alpha=0.7)
        
        # Add price range annotation
        min_price = data['actual'].min()
        max_price = data['actual'].max()
        plt.figtext(0.02, 0.05, f"Price range: ${min_price:.2f} - ${max_price:.2f}", 
                    fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        
        # Save the plot with high DPI for quality
        plot_path = os.path.join(output_dir, f'{ticker}_monte_carlo_price_paths.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        # Verify the plot file was created and has content
        if os.path.exists(plot_path) and os.path.getsize(plot_path) > 1000:
            print(f"Plot saved to {plot_path} ({os.path.getsize(plot_path)/1024:.0f}KB)")
        else:
            print(f"Warning: Plot file may be empty or corrupt: {plot_path}")
        
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error plotting data for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return False

def plot_combined_simulations(data_dict, output_dir=None):
    """
    Create a combined plot showing simulations for multiple tickers.
    
    Args:
        data_dict: Dictionary with ticker symbols as keys and their dataframes as values
        output_dir: Directory to save the plot
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'monte_carlo_plots')
    
    print(f"Generating combined plot in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Verify we have data
        if not data_dict:
            print("Error: No data provided for combined plot")
            return False
        
        # Create high quality figure
        plt.figure(figsize=(14, 8), dpi=100)
        
        # Create subplots for each ticker
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        color_index = 0
        valid_tickers = 0
        
        # Plot only actual price lines for each ticker for clarity
        for ticker, data in data_dict.items():
            try:
                # Validate data
                if data.empty or 'actual' not in data.columns or data['actual'].isna().all():
                    print(f"Warning: Invalid data for {ticker} in combined plot. Skipping.")
                    continue
                
                # Normalize to percentage change from first day for better comparison
                first_price = data['actual'].iloc[0]
                if first_price <= 0:
                    print(f"Warning: First price for {ticker} is invalid: {first_price}. Skipping.")
                    continue
                    
                normalized_prices = (data['actual'] / first_price - 1) * 100
                
                # Convert DatetimeIndex to numpy array
                dates_array = data.index.to_numpy()
                prices_array = normalized_prices.to_numpy()
                
                plt.plot(dates_array, prices_array, 
                        linewidth=2.5, 
                        label=f'{ticker} Actual',
                        color=colors[color_index % len(colors)])
                color_index += 1
                valid_tickers += 1
                
                print(f"Added {ticker} to combined plot: change range {normalized_prices.min():.2f}% to {normalized_prices.max():.2f}%")
            except Exception as e:
                print(f"Error adding {ticker} to combined plot: {e}")
        
        # If no valid tickers were plotted, return False
        if valid_tickers == 0:
            print("Error: No valid ticker data for combined plot")
            return False
        
        # Add a horizontal line at 0%
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Format the plot
        plt.title('Comparative Stock Price Performance', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Percentage Change (%)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format dates on x-axis
        plt.gcf().autofmt_xdate()
        
        # Add annotation
        plt.figtext(0.02, 0.02, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                    fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        
        # Save the plot with high DPI
        combined_path = os.path.join(output_dir, 'simulated_stock_prices.png')
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        
        # Verify the plot file was created and has content
        if os.path.exists(combined_path) and os.path.getsize(combined_path) > 1000:
            print(f"Combined plot saved to {combined_path} ({os.path.getsize(combined_path)/1024:.0f}KB)")
        else:
            print(f"Warning: Combined plot file may be empty or corrupt: {combined_path}")
        
        plt.close()
        return True
        
    except Exception as e:
        print(f"Error creating combined plot: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Monte Carlo plot generation...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Base directory: {CURRENT_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create a simple test plot first to verify plotting works
    create_simple_plot()
    
    try:
        # Load stock data
        data = load_stock_data()
        
        # Print data summary
        print(f"\nStock data summary:")
        print(f"Rows: {len(data)}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        
        # Define parameters
        tickers = ['AAPL', 'MSFT']
        out_sample_start = '2021-05-26'
        out_sample_end = '2021-12-31'
        
        # Create output directories
        data_dir = os.path.join(OUTPUT_DIR, 'monte_carlo_data')
        plot_dir = os.path.join(OUTPUT_DIR, 'monte_carlo_plots')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        
        # Dictionary to store data for combined plot
        all_data = {}
        
        # Process each ticker
        for ticker in tickers:
            try:
                print(f"\nProcessing {ticker}...")
                # Save data to CSV
                combined_data = save_data_to_csv(original_data=data, 
                                               ticker=ticker, 
                                               out_sample_start=out_sample_start, 
                                               out_sample_end=out_sample_end, 
                                               num_simulations=10,  # Increased for better visualization
                                               output_dir=data_dir)
                
                # Create plot from data
                if combined_data is not None and not combined_data.empty:
                    # Ensure index is properly converted to datetime
                    if not isinstance(combined_data.index, pd.DatetimeIndex):
                        print(f"Converting index to DatetimeIndex for {ticker}")
                        combined_data.index = pd.to_datetime(combined_data.index)
                    
                    # Store for combined plot
                    all_data[ticker] = combined_data
                    
                    # Check if there's any valid data for plotting
                    if 'actual' in combined_data.columns and not combined_data['actual'].isna().all():
                        success = plot_simulated_stock_prices(data=combined_data, 
                                                           ticker=ticker, 
                                                           output_dir=plot_dir)
                        if success:
                            print(f"Successfully processed {ticker}")
                        else:
                            print(f"Failed to create plot for {ticker}")
                    else:
                        print(f"No valid data found for {ticker}")
                else:
                    print(f"No data generated for {ticker}")
            
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create combined plot for all tickers
        if all_data:
            print("\nCreating combined performance plot...")
            plot_combined_simulations(all_data, plot_dir)
        else:
            print("No data available for combined plot")
        
        print("\nAll processing completed!")
    
    except Exception as e:
        print(f"Error in main process: {e}")
        import traceback
        traceback.print_exc() 