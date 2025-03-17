import pandas as pd
import yfinance as yf
import os

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock data for multiple tickers and S&P 500, saving them into a single CSV file.
    
    Args:
        tickers (list): List of stock ticker symbols (e.g., ['MSFT', 'AAPL']).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    """
    # Create input directory if it doesn’t exist
    os.makedirs('input', exist_ok=True)
    
    # Add S&P 500 ticker to the list
    tickers_with_sp500 = tickers + ['^GSPC']
    
    # Fetch data for all tickers including S&P 500
    print(f"Fetching data for {', '.join(tickers_with_sp500)}...")
    data = yf.download(tickers_with_sp500, start=start_date, end=end_date)
    
    if data.empty:
        print("No data fetched. Please check the tickers or date range.")
        return
    
    # Reset index to make 'Date' a column
    data = data.reset_index()
    
    # Flatten MultiIndex columns to '{ticker}_{field}' format (e.g., 'MSFT_Close')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[1] + '_' + col[0] if col[0] != 'Date' else col[0] for col in data.columns]
    
    # Rename '^GSPC' to 'SP500' for clarity
    data.columns = [col.replace('^GSPC', 'SP500') for col in data.columns]
    
    # Format 'Date' column to string (YYYY-MM-DD)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    output_path = 'input/stock_data.csv'
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path} with columns: {', '.join(data.columns)}")

if __name__ == "__main__":
    # Get user input
    tickers_input = input("Enter stock tickers separated by commas (e.g., MSFT,AAPL): ").strip()
    tickers = [ticker.strip() for ticker in tickers_input.split(',')]
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    
    # Fetch and save the data
    fetch_stock_data(tickers, start_date, end_date)