import pandas as pd
import yfinance as yf
import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_manager import path_manager, get_input_dir

def fetch_stock_data(tickers, start_date, end_date, output_path='input/stock_data.csv', force_refresh=False):
    """
    Fetch historical stock data for multiple tickers and S&P 500, saving them into a single CSV file.
    
    Args:
        tickers (list or str): List of stock ticker symbols (e.g., ['MSFT', 'AAPL']) or comma-separated string.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        output_path (str, optional): Path to save the CSV file. Default is 'input/stock_data.csv'.
        force_refresh (bool, optional): If True, always fetch fresh data for all tickers. Default is False.
        
    Returns:
        str: Path to the saved CSV file
    """
    # Ensure tickers is properly formatted as a list
    if tickers is None:
        tickers = ["SPY"]
    elif isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(',') if t.strip()]
        
    print(f"Fetching stock data for tickers: {tickers}")
    # Create input directory if it doesn't exist
    if output_path.startswith('input/'):
        # Use the path relative to the input directory
        rel_path = output_path.replace('input/', '')
        full_output_path = path_manager.input_dir / rel_path
    else:
        # Use the provided path as is, but ensure it's a Path object
        full_output_path = Path(output_path)
    
    # Ensure parent directory exists
    path_manager.ensure_dir(full_output_path.parent)
    
    # If force_refresh is True, always fetch all data
    if force_refresh:
        existing_data = None
        print(f"Force refresh is enabled - fetching fresh data for all tickers")
    else:
        # Check if we need to update existing file or create a new one
        existing_data = None
        
        if os.path.exists(full_output_path):
            try:
                existing_data = pd.read_csv(full_output_path)
                print(f"Found existing data file with {len(existing_data)} rows and columns: {', '.join(existing_data.columns)}")
                
                # Extract existing tickers from column names
                existing_tickers = set()
                for col in existing_data.columns:
                    if '_' in col:
                        ticker = col.split('_')[0]
                        existing_tickers.add(ticker)
                
                print(f"Existing tickers: {', '.join(existing_tickers)}")
            except Exception as e:
                print(f"Error reading existing file: {e}. Will create a new file.")
                existing_data = None
    
    # Prepare list of tickers to fetch
    tickers_to_fetch = tickers.copy() if isinstance(tickers, list) else tickers.split(',')
    tickers_to_fetch = [t.strip() for t in tickers_to_fetch]
    
    # Determine which tickers need to be fetched
    if not force_refresh and existing_data is not None:
        # Filter out tickers that are already in the file
        new_tickers = []
        for ticker in tickers_to_fetch:
            # Check if this ticker's data columns exist
            if f"{ticker}_Close" not in existing_data.columns:
                new_tickers.append(ticker)
        
        # If all tickers already exist, return the existing file
        if not new_tickers:
            print("All requested tickers already exist in the data file.")
            return full_output_path
        
        print(f"Need to fetch data for new tickers: {', '.join(new_tickers)}")
        tickers_to_fetch = new_tickers
    
    # Add S&P 500 ticker if it doesn't exist in the data and isn't in the tickers to fetch
    if force_refresh or existing_data is None or ('SPY_Close' not in existing_data.columns and 
                                '^GSPC' not in tickers_to_fetch and 
                                'SPY' not in tickers_to_fetch):
        if '^GSPC' not in tickers_to_fetch and 'SPY' not in tickers_to_fetch:
            tickers_to_fetch.append('^GSPC')
    
    # If no tickers to fetch, return the existing file path
    if not tickers_to_fetch:
        print("No new tickers to fetch.")
        return full_output_path
    
    # Fetch data for the tickers with retry for rate limiting
    print(f"Fetching data for {', '.join(tickers_to_fetch)}...")
    max_retries = 3
    retry_delay = 10  # seconds
    
    for retry in range(max_retries):
        try:
            # Add a longer timeout to handle rate limiting
            new_data = yf.download(tickers_to_fetch, start=start_date, end=end_date, timeout=30)
            break
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Attempt {retry+1} failed: {e}. Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"All {max_retries} attempts failed. Last error: {e}")
                raise
    
    if new_data.empty:
        print("No data fetched. Please check the tickers or date range.")
        return full_output_path if existing_data is not None else None
    
    # Process the new data
    new_data = new_data.reset_index()
    
    # Flatten MultiIndex columns to '{ticker}_{field}' format
    if isinstance(new_data.columns, pd.MultiIndex):
        new_data.columns = [f"{col[1]}_{col[0]}" if col[0] != 'Date' else col[0] for col in new_data.columns]
    
    # Rename '^GSPC' to 'SPY' for clarity
    new_data.columns = [col.replace('^GSPC', 'SPY') for col in new_data.columns]
    
    # Format 'Date' column to string (YYYY-MM-DD)
    new_data['Date'] = pd.to_datetime(new_data['Date']).dt.strftime('%Y-%m-%d')
    
    # Merge with existing data if available and not doing a force refresh
    if not force_refresh and existing_data is not None:
        # Ensure date format is consistent
        existing_data['Date'] = pd.to_datetime(existing_data['Date']).dt.strftime('%Y-%m-%d')
        
        # Merge on the Date column
        merged_data = pd.merge(existing_data, new_data, on='Date', how='outer')
        
        # For duplicate columns (like Date), use the _x version
        duplicate_cols = [col for col in merged_data.columns if col.endswith('_x') or col.endswith('_y')]
        if duplicate_cols:
            # Create a mapping to rename columns
            rename_map = {}
            for col in duplicate_cols:
                if col.endswith('_x'):
                    base_col = col[:-2]
                    rename_map[col] = base_col
                elif col.endswith('_y'):
                    # Drop the _y columns as we prefer the existing data
                    merged_data = merged_data.drop(columns=[col])
            
            # Rename the _x columns back to their original names
            merged_data = merged_data.rename(columns=rename_map)
        
        # Sort by date
        merged_data = merged_data.sort_values('Date')
        
        # Save the merged data
        merged_data.to_csv(full_output_path, index=False)
        print(f"Updated data saved to {full_output_path} with columns: {', '.join(merged_data.columns)}")
    else:
        # Save the new data directly
        new_data.to_csv(full_output_path, index=False)
        print(f"New data saved to {full_output_path} with columns: {', '.join(new_data.columns)}")
    
    return full_output_path

def auto_setup(tickers=None, force_refresh=True):
    """
    Auto setup function for when this module is imported by others.
    Uses default values if not provided.
    
    Args:
        tickers (str or list, optional): Ticker symbols. Default is ['SPY', 'AAPL', 'MSFT', 'GOOGL'].
        force_refresh (bool, optional): If True, always fetch fresh data for all tickers. Default is True.
        
    Returns:
        str: Path to the saved CSV file
    """
    if tickers is None:
        # Default tickers to use
        tickers = ['SPY', 'AAPL', 'MSFT', 'GOOGL']
    
    if isinstance(tickers, str):
        # Convert comma-separated string to list
        tickers = [t.strip() for t in tickers.split(',')]
    
    # Default date range (5 years of data)
    start_date = "2020-01-01"
    end_date = "2025-01-01"
    
    return fetch_stock_data(tickers, start_date, end_date, force_refresh=force_refresh)

if __name__ == "__main__":
    # Get user input
    tickers_input = input("Enter stock tickers separated by commas (e.g., MSFT,AAPL): ").strip()
    tickers = [ticker.strip() for ticker in tickers_input.split(',')]
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    force_refresh = input("Force refresh all data? (y/n): ").strip().lower() == 'y'
    
    # Fetch and save the data
    fetch_stock_data(tickers, start_date, end_date, force_refresh=force_refresh)