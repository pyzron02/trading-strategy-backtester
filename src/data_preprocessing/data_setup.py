import pandas as pd
import yfinance as yf
import os
import sys

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
project_root = os.path.dirname(src_dir)  # Go up to project root
if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

def fetch_stock_data(tickers, start_date, end_date, output_path='input/stock_data.csv'):
    """
    Fetch historical stock data for multiple tickers and S&P 500, saving them into a single CSV file.
    
    Args:
        tickers (list): List of stock ticker symbols (e.g., ['MSFT', 'AAPL']).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        output_path (str, optional): Path to save the CSV file. Default is 'input/stock_data.csv'.
        
    Returns:
        str: Path to the saved CSV file
    """
    # Create input directory if it doesn't exist
    input_dir = os.path.dirname(os.path.join(project_root, output_path))
    os.makedirs(input_dir, exist_ok=True)
    
    # Add S&P 500 ticker to the list
    tickers_with_sp500 = tickers.copy() if isinstance(tickers, list) else tickers.split(',')
    tickers_with_sp500 = [t.strip() for t in tickers_with_sp500]
    
    if '^GSPC' not in tickers_with_sp500:
        tickers_with_sp500.append('^GSPC')
    
    # Fetch data for all tickers including S&P 500
    print(f"Fetching data for {', '.join(tickers_with_sp500)}...")
    data = yf.download(tickers_with_sp500, start=start_date, end=end_date)
    
    if data.empty:
        print("No data fetched. Please check the tickers or date range.")
        return None
    
    # Reset index to make 'Date' a column
    data = data.reset_index()
    
    # Flatten MultiIndex columns to '{ticker}_{field}' format (e.g., 'MSFT_Close')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [f"{col[1]}_{col[0]}" if col[0] != 'Date' else col[0] for col in data.columns]
    
    # Rename '^GSPC' to 'SPY' for clarity
    data.columns = [col.replace('^GSPC', 'SPY') for col in data.columns]
    
    # Format 'Date' column to string (YYYY-MM-DD)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    full_output_path = os.path.join(project_root, output_path)
    data.to_csv(full_output_path, index=False)
    print(f"Data saved to {full_output_path} with columns: {', '.join(data.columns)}")
    
    return full_output_path

def auto_setup(tickers=None):
    """
    Auto setup function for when this module is imported by others.
    Uses default values if not provided.
    
    Args:
        tickers (str or list, optional): Ticker symbols. Default is ['SPY', 'AAPL', 'MSFT', 'GOOGL'].
        
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
    
    return fetch_stock_data(tickers, start_date, end_date)

if __name__ == "__main__":
    # Get user input
    tickers_input = input("Enter stock tickers separated by commas (e.g., MSFT,AAPL): ").strip()
    tickers = [ticker.strip() for ticker in tickers_input.split(',')]
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    
    # Fetch and save the data
    fetch_stock_data(tickers, start_date, end_date)