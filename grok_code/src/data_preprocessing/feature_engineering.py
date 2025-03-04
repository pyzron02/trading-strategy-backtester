#!/usr/bin/env python3
# feature_engineering.py - Extract and create features from stock data

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
import argparse
import warnings

# Filter warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class StockFeatureEngineer:
    """
    A class for performing feature engineering on stock data.
    Extracts technical indicators and other features useful for trading strategies.
    """
    
    def __init__(self, input_file=None, output_dir=None):
        """
        Initialize the feature engineer with input and output paths.
        
        Args:
            input_file (str): Path to the input CSV file containing stock data.
                             If None, will use the default path.
            output_dir (str): Directory to save processed data.
                             If None, will use the default path.
        """
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        
        # Set default input file if not provided
        if input_file is None:
            input_file = os.path.join(project_root, 'input', 'stock_data.csv')
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(project_root, 'output', 'features')
        
        self.input_file = input_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data containers
        self.raw_data = None
        self.processed_data = {}
        
    def load_data(self):
        """Load the stock data from the input file."""
        print(f"Loading data from {self.input_file}...")
        
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        self.raw_data = pd.read_csv(self.input_file)
        print(f"Loaded {len(self.raw_data)} rows of data.")
        print(f"Columns: {self.raw_data.columns.tolist()}")
        
        # Convert date column to datetime
        if 'Date' in self.raw_data.columns:
            self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])
        
        return self.raw_data
    
    def detect_tickers(self):
        """
        Auto-detect ticker symbols from column names.
        
        Returns:
            list: List of detected ticker symbols.
        """
        if self.raw_data is None:
            self.load_data()
        
        all_tickers = set()
        ticker_pattern = re.compile(r'([A-Z]+)_(?:Open|High|Low|Close|Volume)')
        
        for col in self.raw_data.columns:
            match = ticker_pattern.match(col)
            if match:
                ticker = match.group(1)
                if ticker != 'SP500':  # Exclude SP500 if needed
                    all_tickers.add(ticker)
        
        tickers = sorted(list(all_tickers))
        print(f"Auto-detected tickers: {tickers}")
        return tickers
    
    def extract_ticker_data(self, ticker):
        """
        Extract data for a specific ticker from the combined dataset.
        
        Args:
            ticker (str): The ticker symbol to extract (e.g., 'AAPL', 'MSFT').
            
        Returns:
            pd.DataFrame: DataFrame containing the data for the specified ticker.
        """
        if self.raw_data is None:
            self.load_data()
        
        # Check if ticker columns exist
        required_cols = [f'{ticker}_{field}' for field in ['Open', 'High', 'Low', 'Close', 'Volume']]
        missing_cols = [col for col in required_cols if col not in self.raw_data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing columns for ticker {ticker}: {missing_cols}")
        
        # Extract and rename columns for the ticker
        ticker_data = self.raw_data[['Date'] + required_cols].copy()
        ticker_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Store in processed_data dictionary
        self.processed_data[ticker] = ticker_data
        
        return ticker_data
    
    def add_price_features(self, ticker):
        """
        Add price-based features for a specific ticker.
        
        Args:
            ticker (str): The ticker symbol to process.
            
        Returns:
            pd.DataFrame: DataFrame with added price features.
        """
        if ticker not in self.processed_data:
            self.extract_ticker_data(ticker)
        
        df = self.processed_data[ticker].copy()
        
        # Calculate returns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Calculate moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # Calculate price momentum
        for period in [1, 5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
            df[f'Return_{period}d'] = df['Close'].pct_change(period)
        
        # Calculate volatility
        for period in [5, 10, 20]:
            df[f'Volatility_{period}d'] = df['Daily_Return'].rolling(window=period).std() * np.sqrt(252)
        
        # Calculate price channels
        for period in [10, 20, 50]:
            df[f'Upper_Channel_{period}'] = df['High'].rolling(window=period).max()
            df[f'Lower_Channel_{period}'] = df['Low'].rolling(window=period).min()
            df[f'Channel_Width_{period}'] = df[f'Upper_Channel_{period}'] - df[f'Lower_Channel_{period}']
            df[f'Channel_Position_{period}'] = (df['Close'] - df[f'Lower_Channel_{period}']) / df[f'Channel_Width_{period}']
        
        # Calculate Bollinger Bands
        for period in [20]:
            df[f'BB_Middle_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'BB_Std_{period}'] = df['Close'].rolling(window=period).std()
            df[f'BB_Upper_{period}'] = df[f'BB_Middle_{period}'] + (df[f'BB_Std_{period}'] * 2)
            df[f'BB_Lower_{period}'] = df[f'BB_Middle_{period}'] - (df[f'BB_Std_{period}'] * 2)
            df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / df[f'BB_Middle_{period}']
            df[f'BB_Position_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'])
        
        # Calculate price gaps
        df['Gap_Up'] = (df['Open'] > df['High'].shift(1)).astype(int)
        df['Gap_Down'] = (df['Open'] < df['Low'].shift(1)).astype(int)
        
        # Calculate true range and ATR
        df['High_Low'] = df['High'] - df['Low']
        df['High_PrevClose'] = abs(df['High'] - df['Close'].shift(1))
        df['Low_PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
        df['True_Range'] = df[['High_Low', 'High_PrevClose', 'Low_PrevClose']].max(axis=1)
        df['ATR_14'] = df['True_Range'].rolling(window=14).mean()
        
        # Update processed data
        self.processed_data[ticker] = df
        
        return df
    
    def add_volume_features(self, ticker):
        """
        Add volume-based features for a specific ticker.
        
        Args:
            ticker (str): The ticker symbol to process.
            
        Returns:
            pd.DataFrame: DataFrame with added volume features.
        """
        if ticker not in self.processed_data:
            self.extract_ticker_data(ticker)
            self.add_price_features(ticker)
        
        df = self.processed_data[ticker].copy()
        
        # Calculate volume moving averages
        for period in [5, 10, 20, 50]:
            df[f'Volume_SMA_{period}'] = df['Volume'].rolling(window=period).mean()
        
        # Calculate volume ratios
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Calculate on-balance volume (OBV)
        df['OBV'] = 0
        df.loc[df['Daily_Return'] > 0, 'OBV'] = df['Volume']
        df.loc[df['Daily_Return'] < 0, 'OBV'] = -df['Volume']
        df['OBV'] = df['OBV'].cumsum()
        
        # Calculate volume-weighted average price (VWAP)
        df['Price_Volume'] = df['Close'] * df['Volume']
        df['Cumulative_Volume'] = df['Volume'].cumsum()
        df['Cumulative_Price_Volume'] = df['Price_Volume'].cumsum()
        df['VWAP'] = df['Cumulative_Price_Volume'] / df['Cumulative_Volume']
        
        # Calculate volume-price relationship
        df['Volume_Price_Ratio'] = df['Volume'] / df['Close']
        
        # Calculate volume momentum
        for period in [1, 5, 10]:
            df[f'Volume_Momentum_{period}'] = df['Volume'] - df['Volume'].shift(period)
        
        # Calculate volume volatility
        for period in [5, 10, 20]:
            df[f'Volume_Volatility_{period}d'] = df['Volume'].rolling(window=period).std() / df['Volume'].rolling(window=period).mean()
        
        # Update processed data
        self.processed_data[ticker] = df
        
        return df
    
    def add_technical_indicators(self, ticker):
        """
        Add technical indicators for a specific ticker.
        
        Args:
            ticker (str): The ticker symbol to process.
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators.
        """
        if ticker not in self.processed_data:
            self.extract_ticker_data(ticker)
            self.add_price_features(ticker)
            self.add_volume_features(ticker)
        
        df = self.processed_data[ticker].copy()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Calculate Stochastic Oscillator
        for period in [14]:
            df[f'Lowest_Low_{period}'] = df['Low'].rolling(window=period).min()
            df[f'Highest_High_{period}'] = df['High'].rolling(window=period).max()
            df[f'Stochastic_%K_{period}'] = 100 * ((df['Close'] - df[f'Lowest_Low_{period}']) / 
                                                 (df[f'Highest_High_{period}'] - df[f'Lowest_Low_{period}']))
            df[f'Stochastic_%D_{period}'] = df[f'Stochastic_%K_{period}'].rolling(window=3).mean()
        
        # Calculate Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = ((df['Close'] / df['Close'].shift(period)) - 1) * 100
        
        # Calculate Average Directional Index (ADX)
        # +DM, -DM
        df['Plus_DM'] = 0
        df['Minus_DM'] = 0
        
        # Calculate +DM and -DM
        for i in range(1, len(df)):
            up_move = df['High'].iloc[i] - df['High'].iloc[i-1]
            down_move = df['Low'].iloc[i-1] - df['Low'].iloc[i]
            
            if up_move > down_move and up_move > 0:
                df['Plus_DM'].iloc[i] = up_move
            else:
                df['Plus_DM'].iloc[i] = 0
                
            if down_move > up_move and down_move > 0:
                df['Minus_DM'].iloc[i] = down_move
            else:
                df['Minus_DM'].iloc[i] = 0
        
        # Calculate +DI and -DI
        df['Plus_DI_14'] = 100 * (df['Plus_DM'].rolling(window=14).mean() / df['ATR_14'])
        df['Minus_DI_14'] = 100 * (df['Minus_DM'].rolling(window=14).mean() / df['ATR_14'])
        
        # Calculate DX and ADX
        df['DX_14'] = 100 * (abs(df['Plus_DI_14'] - df['Minus_DI_14']) / (df['Plus_DI_14'] + df['Minus_DI_14']))
        df['ADX_14'] = df['DX_14'].rolling(window=14).mean()
        
        # Calculate Commodity Channel Index (CCI)
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['TP_SMA_20'] = df['Typical_Price'].rolling(window=20).mean()
        df['TP_Deviation'] = abs(df['Typical_Price'] - df['TP_SMA_20'])
        df['TP_Deviation_SMA'] = df['TP_Deviation'].rolling(window=20).mean()
        df['CCI_20'] = (df['Typical_Price'] - df['TP_SMA_20']) / (0.015 * df['TP_Deviation_SMA'])
        
        # Calculate Williams %R
        for period in [14]:
            df[f'Williams_%R_{period}'] = -100 * ((df[f'Highest_High_{period}'] - df['Close']) / 
                                                (df[f'Highest_High_{period}'] - df[f'Lowest_Low_{period}']))
        
        # Calculate Parabolic SAR (simplified)
        df['PSAR'] = df['SMA_10']  # Simplified approximation
        
        # Calculate Ichimoku Cloud components (simplified)
        df['Tenkan_Sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
        df['Kijun_Sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
        df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
        df['Senkou_Span_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
        df['Chikou_Span'] = df['Close'].shift(-26)
        
        # Update processed data
        self.processed_data[ticker] = df
        
        return df
    
    def add_pattern_features(self, ticker):
        """
        Add candlestick pattern features for a specific ticker.
        
        Args:
            ticker (str): The ticker symbol to process.
            
        Returns:
            pd.DataFrame: DataFrame with added pattern features.
        """
        if ticker not in self.processed_data:
            self.extract_ticker_data(ticker)
        
        df = self.processed_data[ticker].copy()
        
        # Calculate basic candlestick properties
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Range'] = df['High'] - df['Low']
        df['Body_Ratio'] = df['Body'] / df['Range']
        df['Upper_Shadow_Ratio'] = df['Upper_Shadow'] / df['Range']
        df['Lower_Shadow_Ratio'] = df['Lower_Shadow'] / df['Range']
        
        # Identify bullish and bearish candles
        df['Bullish'] = (df['Close'] > df['Open']).astype(int)
        df['Bearish'] = (df['Close'] < df['Open']).astype(int)
        
        # Doji pattern (open and close are almost equal)
        df['Doji'] = (df['Body_Ratio'] < 0.1).astype(int)
        
        # Hammer pattern (small body, long lower shadow, small upper shadow)
        df['Hammer'] = ((df['Body_Ratio'] < 0.3) & 
                        (df['Lower_Shadow_Ratio'] > 0.6) & 
                        (df['Upper_Shadow_Ratio'] < 0.1)).astype(int)
        
        # Shooting Star pattern (small body, long upper shadow, small lower shadow)
        df['Shooting_Star'] = ((df['Body_Ratio'] < 0.3) & 
                              (df['Upper_Shadow_Ratio'] > 0.6) & 
                              (df['Lower_Shadow_Ratio'] < 0.1)).astype(int)
        
        # Engulfing patterns
        df['Bullish_Engulfing'] = ((df['Bullish'] == 1) & 
                                  (df['Open'] < df['Close'].shift(1)) & 
                                  (df['Close'] > df['Open'].shift(1))).astype(int)
        
        df['Bearish_Engulfing'] = ((df['Bearish'] == 1) & 
                                  (df['Open'] > df['Close'].shift(1)) & 
                                  (df['Close'] < df['Open'].shift(1))).astype(int)
        
        # Initialize Morning Star and Evening Star columns
        df['Morning_Star'] = 0
        df['Evening_Star'] = 0
        
        # Morning Star (3-candle bullish reversal pattern)
        for i in range(2, len(df)):
            if (df['Bearish'].iloc[i-2] == 1 and 
                df['Body'].iloc[i-1] < df['Body'].iloc[i-2] * 0.3 and 
                df['Bullish'].iloc[i] == 1 and 
                df['Close'].iloc[i] > (df['Open'].iloc[i-2] + df['Close'].iloc[i-2]) / 2):
                # Use loc instead of iloc to avoid SettingWithCopyWarning
                df.loc[df.index[i], 'Morning_Star'] = 1
        
        # Evening Star (3-candle bearish reversal pattern)
        for i in range(2, len(df)):
            if (df['Bullish'].iloc[i-2] == 1 and 
                df['Body'].iloc[i-1] < df['Body'].iloc[i-2] * 0.3 and 
                df['Bearish'].iloc[i] == 1 and 
                df['Close'].iloc[i] < (df['Open'].iloc[i-2] + df['Close'].iloc[i-2]) / 2):
                # Use loc instead of iloc to avoid SettingWithCopyWarning
                df.loc[df.index[i], 'Evening_Star'] = 1
        
        # Update processed data
        self.processed_data[ticker] = df
        
        return df
    
    def add_market_features(self, ticker):
        """
        Add market-related features for a specific ticker.
        
        Args:
            ticker (str): The ticker symbol to process.
            
        Returns:
            pd.DataFrame: DataFrame with added market features.
        """
        if ticker not in self.processed_data:
            self.extract_ticker_data(ticker)
        
        df = self.processed_data[ticker].copy()
        
        # Check if SP500 data is available
        if 'SP500_Close' in self.raw_data.columns:
            # Extract SP500 data
            sp500_data = self.raw_data[['Date', 'SP500_Close']].copy()
            
            # Merge with ticker data
            df = pd.merge(df, sp500_data, on='Date', how='left')
            
            # Calculate market returns
            df['SP500_Return'] = df['SP500_Close'].pct_change()
            
            # Calculate beta (market sensitivity)
            # First, ensure we have returns calculated
            if 'Daily_Return' not in df.columns:
                df['Daily_Return'] = df['Close'].pct_change()
            
            # Calculate rolling beta
            for period in [20, 60]:
                cov = df['Daily_Return'].rolling(window=period).cov(df['SP500_Return'])
                market_var = df['SP500_Return'].rolling(window=period).var()
                df[f'Beta_{period}d'] = cov / market_var
            
            # Calculate relative strength
            for period in [5, 10, 20, 50]:
                df[f'RS_{period}d'] = ((df['Close'] / df['Close'].shift(period)) / 
                                      (df['SP500_Close'] / df['SP500_Close'].shift(period)))
            
            # Calculate correlation with market
            for period in [20, 60]:
                df[f'Market_Corr_{period}d'] = df['Daily_Return'].rolling(window=period).corr(df['SP500_Return'])
        
        # Update processed data
        self.processed_data[ticker] = df
        
        return df
    
    def add_target_variables(self, ticker, forward_periods=[1, 5, 10, 20]):
        """
        Add target variables for machine learning.
        
        Args:
            ticker (str): The ticker symbol to process.
            forward_periods (list): List of forward periods for return prediction.
            
        Returns:
            pd.DataFrame: DataFrame with added target variables.
        """
        if ticker not in self.processed_data:
            self.extract_ticker_data(ticker)
        
        df = self.processed_data[ticker].copy()
        
        # Calculate forward returns
        for period in forward_periods:
            # Continuous target (percentage return)
            df[f'Forward_Return_{period}d'] = df['Close'].shift(-period) / df['Close'] - 1
            
            # Binary classification targets
            df[f'Target_Up_{period}d'] = (df[f'Forward_Return_{period}d'] > 0).astype(int)
            
            # Multi-class targets
            df[f'Target_Direction_{period}d'] = 0  # 0 = no change
            df.loc[df[f'Forward_Return_{period}d'] > 0.01, f'Target_Direction_{period}d'] = 1  # 1 = up
            df.loc[df[f'Forward_Return_{period}d'] < -0.01, f'Target_Direction_{period}d'] = -1  # -1 = down
        
        # Update processed data
        self.processed_data[ticker] = df
        
        return df
    
    def process_ticker(self, ticker):
        """
        Process a single ticker by applying all feature engineering steps.
        
        Args:
            ticker (str): The ticker symbol to process.
            
        Returns:
            pd.DataFrame: DataFrame with all features.
        """
        print(f"Processing {ticker}...")
        
        try:
            # Extract data
            self.extract_ticker_data(ticker)
            
            # Add features
            self.add_price_features(ticker)
            self.add_volume_features(ticker)
            self.add_technical_indicators(ticker)
            self.add_pattern_features(ticker)
            self.add_market_features(ticker)
            self.add_target_variables(ticker)
            
            # Get the final processed data
            df = self.processed_data[ticker]
            
            # Drop rows with NaN values
            df_clean = df.dropna()
            print(f"Generated {len(df.columns)} features for {ticker}. {len(df_clean)} clean rows after removing NaNs.")
            
            return df
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            return None
    
    def save_features(self, ticker):
        """
        Save the processed features for a ticker to CSV.
        
        Args:
            ticker (str): The ticker symbol to save.
        """
        if ticker in self.processed_data:
            # Create ticker directory
            ticker_dir = os.path.join(self.output_dir, ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            
            # Save full feature set
            output_file = os.path.join(ticker_dir, f"{ticker}_features.csv")
            self.processed_data[ticker].to_csv(output_file, index=False)
            print(f"Features saved to {output_file}")
            
            # Save a clean version (no NaNs)
            clean_file = os.path.join(ticker_dir, f"{ticker}_features_clean.csv")
            self.processed_data[ticker].dropna().to_csv(clean_file, index=False)
            print(f"Clean features saved to {clean_file}")
            
            # Create feature correlation heatmap
            self.plot_feature_correlation(ticker)
    
    def plot_feature_correlation(self, ticker):
        """
        Create and save a correlation heatmap for the features.
        
        Args:
            ticker (str): The ticker symbol to process.
        """
        if ticker in self.processed_data:
            # Get numeric columns only
            df = self.processed_data[ticker].select_dtypes(include=[np.number])
            
            # Calculate correlation matrix
            corr = df.corr()
            
            # Create directory for plots
            plots_dir = os.path.join(self.output_dir, ticker, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Create heatmap
            plt.figure(figsize=(20, 16))
            plt.matshow(corr, fignum=1)
            plt.title(f'{ticker} Feature Correlation Matrix', fontsize=14)
            plt.colorbar()
            
            # Save plot
            corr_plot = os.path.join(plots_dir, f"{ticker}_correlation.png")
            plt.savefig(corr_plot)
            plt.close()
            print(f"Correlation plot saved to {corr_plot}")
            
            # Save top correlations with target variables
            if 'Forward_Return_5d' in df.columns:
                target_corr = corr['Forward_Return_5d'].sort_values(ascending=False)
                target_corr_file = os.path.join(plots_dir, f"{ticker}_target_correlations.csv")
                target_corr.to_csv(target_corr_file)
                print(f"Target correlations saved to {target_corr_file}")
    
    def process_all_tickers(self):
        """
        Process all detected tickers in the dataset.
        """
        # Detect tickers
        tickers = self.detect_tickers()
        
        # Process each ticker
        for ticker in tickers:
            try:
                self.process_ticker(ticker)
                self.save_features(ticker)
                print(f"Completed processing for {ticker}\n")
            except Exception as e:
                print(f"Error processing {ticker}: {e}\n")
                continue
        
        print("Feature engineering completed for all tickers!")

    def add_plotting_functions(self, ticker):
        """
        Add methods to create various stock plots with technical indicators and features.
        This method is deprecated and will be removed in a future version.
        
        Args:
            ticker (str): The ticker symbol to plot.
        """
        print("Warning: add_plotting_functions is deprecated. Use the plotting methods directly.")
        return None

    def create_plots(self, ticker, start_date=None, end_date=None):
        """
        Create and save all plots for a specific ticker.
        
        Args:
            ticker (str): The ticker symbol to plot.
            start_date (str): Start date for the plots (format: 'YYYY-MM-DD')
            end_date (str): End date for the plots (format: 'YYYY-MM-DD')
        """
        print(f"Creating plots for {ticker}...")
        
        # Make sure ticker is processed
        if ticker not in self.processed_data:
            self.process_ticker(ticker)
        
        # Create plots
        try:
            self.plot_price_with_ma(ticker, start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"Error creating price with MA plot: {e}")
        
        try:
            self.plot_technical_indicators(ticker, start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"Error creating technical indicators plot: {e}")
        
        try:
            self.plot_pattern_analysis(ticker, start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"Error creating pattern analysis plot: {e}")
        
        try:
            self.plot_feature_importance(ticker)
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
        
        try:
            self.plot_correlation_network(ticker)
        except Exception as e:
            print(f"Error creating correlation network plot: {e}")
        
        try:
            self.plot_candlestick(ticker, start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"Error creating candlestick plot: {e}")
        
        print(f"Completed creating plots for {ticker}")

    def plot_price_with_ma(self, ticker, start_date=None, end_date=None, mas=[20, 50, 200], figsize=(14, 7)):
        """
        Plot price chart with moving averages.
        
        Args:
            ticker (str): The ticker symbol to process.
            start_date (str): Start date for the plot (format: 'YYYY-MM-DD')
            end_date (str): End date for the plot (format: 'YYYY-MM-DD')
            mas (list): List of moving average periods to plot
            figsize (tuple): Figure size
        """
        if ticker not in self.processed_data:
            self.process_ticker(ticker)
            
        plot_df = self.processed_data[ticker].copy()
        
        if start_date:
            plot_df = plot_df[plot_df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            plot_df = plot_df[plot_df['Date'] <= pd.to_datetime(end_date)]
        
        plt.figure(figsize=figsize)
        plt.plot(plot_df['Date'].values, plot_df['Close'].values, label=f'{ticker} Close')
        
        for ma in mas:
            if f'SMA_{ma}' in plot_df.columns:
                plt.plot(plot_df['Date'].values, plot_df[f'SMA_{ma}'].values, label=f'SMA {ma}')
        
        plt.title(f'{ticker} Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(self.output_dir, ticker, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f"{ticker}_price_with_ma.png"))
        plt.close()
        print(f"Price with MA plot saved to {plots_dir}/{ticker}_price_with_ma.png")
    
    def plot_technical_indicators(self, ticker, start_date=None, end_date=None, figsize=(14, 14)):
        """
        Plot price with multiple technical indicators.
        
        Args:
            ticker (str): The ticker symbol to process.
            start_date (str): Start date for the plot (format: 'YYYY-MM-DD')
            end_date (str): End date for the plot (format: 'YYYY-MM-DD')
            figsize (tuple): Figure size
        """
        if ticker not in self.processed_data:
            self.process_ticker(ticker)
            
        plot_df = self.processed_data[ticker].copy()
        
        if start_date:
            plot_df = plot_df[plot_df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            plot_df = plot_df[plot_df['Date'] <= pd.to_datetime(end_date)]
        
        # Create subplots
        fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # Price and Bollinger Bands
        axs[0].plot(plot_df['Date'].values, plot_df['Close'].values, label='Close')
        if 'BB_Upper_20' in plot_df.columns and 'BB_Lower_20' in plot_df.columns:
            axs[0].plot(plot_df['Date'].values, plot_df['BB_Upper_20'].values, 'r--', label='Upper BB')
            axs[0].plot(plot_df['Date'].values, plot_df['BB_Middle_20'].values, 'g--', label='Middle BB')
            axs[0].plot(plot_df['Date'].values, plot_df['BB_Lower_20'].values, 'r--', label='Lower BB')
        axs[0].set_title(f'{ticker} Price and Technical Indicators')
        axs[0].set_ylabel('Price')
        axs[0].legend(loc='upper left')
        axs[0].grid(True, alpha=0.3)
        
        # Volume
        axs[1].bar(plot_df['Date'].values, plot_df['Volume'].values, color='blue', alpha=0.5)
        axs[1].set_ylabel('Volume')
        axs[1].grid(True, alpha=0.3)
        
        # RSI
        if 'RSI_14' in plot_df.columns:
            axs[2].plot(plot_df['Date'].values, plot_df['RSI_14'].values, label='RSI(14)')
            axs[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
            axs[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)
            axs[2].set_ylabel('RSI')
            axs[2].set_ylim(0, 100)
            axs[2].legend(loc='upper left')
            axs[2].grid(True, alpha=0.3)
        
        # MACD
        if all(x in plot_df.columns for x in ['MACD', 'MACD_Signal']):
            axs[3].plot(plot_df['Date'].values, plot_df['MACD'].values, label='MACD')
            axs[3].plot(plot_df['Date'].values, plot_df['MACD_Signal'].values, label='Signal')
            if 'MACD_Hist' in plot_df.columns:
                axs[3].bar(plot_df['Date'].values, plot_df['MACD_Hist'].values, color='gray', alpha=0.5, label='Histogram')
            axs[3].set_ylabel('MACD')
            axs[3].legend(loc='upper left')
            axs[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(self.output_dir, ticker, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f"{ticker}_technical_indicators.png"))
        plt.close()
        print(f"Technical indicators plot saved to {plots_dir}/{ticker}_technical_indicators.png")
    
    def plot_pattern_analysis(self, ticker, start_date=None, end_date=None, figsize=(14, 10)):
        """
        Plot price chart with candlestick pattern markers.
        
        Args:
            ticker (str): The ticker symbol to process.
            start_date (str): Start date for the plot (format: 'YYYY-MM-DD')
            end_date (str): End date for the plot (format: 'YYYY-MM-DD')
            figsize (tuple): Figure size
        """
        if ticker not in self.processed_data:
            self.process_ticker(ticker)
            
        plot_df = self.processed_data[ticker].copy()
        
        if start_date:
            plot_df = plot_df[plot_df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            plot_df = plot_df[plot_df['Date'] <= pd.to_datetime(end_date)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot price
        ax.plot(plot_df['Date'].values, plot_df['Close'].values, label='Close Price')
        
        # Plot patterns
        patterns = {
            'Doji': {'color': 'black', 'marker': 'o'},
            'Hammer': {'color': 'green', 'marker': '^'},
            'Shooting_Star': {'color': 'red', 'marker': 'v'},
            'Bullish_Engulfing': {'color': 'limegreen', 'marker': 's'},
            'Bearish_Engulfing': {'color': 'tomato', 'marker': 's'},
            'Morning_Star': {'color': 'green', 'marker': '*'},
            'Evening_Star': {'color': 'red', 'marker': '*'}
        }
        
        for pattern, style in patterns.items():
            if pattern in plot_df.columns:
                # Get dates where pattern is True (1)
                pattern_dates = plot_df.loc[plot_df[pattern] == 1, 'Date'].values
                pattern_prices = plot_df.loc[plot_df[pattern] == 1, 'Close'].values
                
                if len(pattern_dates) > 0:
                    ax.scatter(pattern_dates, pattern_prices, 
                              color=style['color'], marker=style['marker'], 
                              s=100, label=pattern.replace('_', ' '))
        
        ax.set_title(f'{ticker} Price with Candlestick Patterns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(self.output_dir, ticker, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f"{ticker}_pattern_analysis.png"))
        plt.close()
        print(f"Pattern analysis plot saved to {plots_dir}/{ticker}_pattern_analysis.png")
    
    def plot_feature_importance(self, ticker, target='Forward_Return_5d', top_n=20, figsize=(12, 10)):
        """
        Plot feature importance based on correlation with target variable.
        
        Args:
            ticker (str): The ticker symbol to process.
            target (str): Target variable to use for importance
            top_n (int): Number of top features to display
            figsize (tuple): Figure size
        """
        if ticker not in self.processed_data:
            self.process_ticker(ticker)
            
        df = self.processed_data[ticker]
        
        if target not in df.columns:
            print(f"Target variable '{target}' not found in dataframe")
            return
        
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation with target
        target_corr = numeric_df.corr()[target].drop(target)
        
        # Get top features by absolute correlation
        top_features = target_corr.abs().sort_values(ascending=False).head(top_n)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot horizontal bar chart
        colors = ['g' if x >= 0 else 'r' for x in target_corr[top_features.index]]
        plt.barh(top_features.index, target_corr[top_features.index], color=colors)
        
        plt.title(f'Top {top_n} Features by Correlation with {target}')
        plt.xlabel('Correlation')
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(self.output_dir, ticker, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f"{ticker}_feature_importance.png"))
        plt.close()
        print(f"Feature importance plot saved to {plots_dir}/{ticker}_feature_importance.png")
    
    def plot_correlation_network(self, ticker, min_correlation=0.7, figsize=(14, 14)):
        """
        Create a network graph of highly correlated features.
        
        Args:
            ticker (str): The ticker symbol to process.
            min_correlation (float): Minimum absolute correlation to include in the graph
            figsize (tuple): Figure size
        """
        try:
            import networkx as nx
        except ImportError:
            print("networkx package not found. Install with: pip install networkx")
            return
            
        if ticker not in self.processed_data:
            self.process_ticker(ticker)
            
        df = self.processed_data[ticker]
        
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr = numeric_df.corr().abs()
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for col in corr.columns:
            G.add_node(col)
        
        # Add edges for correlations above threshold
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                col_i = corr.columns[i]
                col_j = corr.columns[j]
                if corr.iloc[i, j] >= min_correlation:
                    G.add_edge(col_i, col_j, weight=corr.iloc[i, j])
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Position nodes using force-directed layout
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=50, alpha=0.8)
        
        # Draw edges with varying thickness based on correlation
        for (u, v, d) in G.edges(data=True):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=d['weight']*2, alpha=0.5)
        
        # Draw labels with smaller font
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title(f'{ticker} Feature Correlation Network (min correlation: {min_correlation})')
        plt.axis('off')
        
        # Save plot
        plots_dir = os.path.join(self.output_dir, ticker, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f"{ticker}_correlation_network.png"))
        plt.close()
        print(f"Correlation network plot saved to {plots_dir}/{ticker}_correlation_network.png")
    
    def plot_candlestick(self, ticker, start_date=None, end_date=None, ma_periods=[20, 50], figsize=(14, 7)):
        """
        Create a candlestick chart with volume and moving averages.
        
        Args:
            ticker (str): The ticker symbol to process.
            start_date (str): Start date for the plot (format: 'YYYY-MM-DD')
            end_date (str): End date for the plot (format: 'YYYY-MM-DD')
            ma_periods (list): List of moving average periods to plot
            figsize (tuple): Figure size
        """
        try:
            import mplfinance as mpf
        except ImportError:
            print("mplfinance package not found. Install with: pip install mplfinance")
            return
            
        if ticker not in self.processed_data:
            self.process_ticker(ticker)
        
        plot_df = self.processed_data[ticker].copy()
        
        if start_date:
            plot_df = plot_df[plot_df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            plot_df = plot_df[plot_df['Date'] <= pd.to_datetime(end_date)]
        
        # Set Date as index for mplfinance
        plot_df = plot_df.set_index('Date')
        
        # Create moving average overlays
        ma_overlays = []
        for period in ma_periods:
            if f'SMA_{period}' in plot_df.columns:
                ma_overlays.append(mpf.make_addplot(plot_df[f'SMA_{period}'], width=1, label=f'SMA {period}'))
        
        # Create plot
        plots_dir = os.path.join(self.output_dir, ticker, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, f"{ticker}_candlestick.png")
        
        mpf.plot(plot_df, type='candle', style='yahoo', volume=True, 
                 title=f'{ticker} Candlestick Chart', 
                 figsize=figsize, 
                 addplot=ma_overlays if ma_overlays else None,
                 savefig=save_path)
        
        print(f"Candlestick chart saved to {save_path}")

def main():
    """Main function to run the feature engineering process."""
    parser = argparse.ArgumentParser(description="Perform feature engineering on stock data.")
    parser.add_argument('--input', type=str, default=None,
                        help="Path to input CSV file containing stock data.")
    parser.add_argument('--output', type=str, default=None,
                        help="Directory to save processed features.")
    parser.add_argument('--tickers', type=str, default=None,
                        help="Comma-separated list of tickers to process. If not provided, will auto-detect.")
    parser.add_argument('--plot', action='store_true',
                        help="Generate plots for the processed tickers.")
    parser.add_argument('--start-date', type=str, default=None,
                        help="Start date for plots (format: YYYY-MM-DD).")
    parser.add_argument('--end-date', type=str, default=None,
                        help="End date for plots (format: YYYY-MM-DD).")
    parser.add_argument('--plot-only', action='store_true',
                        help="Only generate plots without reprocessing features.")
    
    args = parser.parse_args()
    
    # Create feature engineer
    engineer = StockFeatureEngineer(input_file=args.input, output_dir=args.output)
    
    # Load data
    engineer.load_data()
    
    # Process specific tickers or all tickers
    if args.tickers:
        tickers = args.tickers.split(',')
    else:
        tickers = engineer.detect_tickers()
    
    if args.plot_only:
        # Only generate plots for existing processed data
        for ticker in tickers:
            try:
                # Load existing data if available
                ticker_file = os.path.join(engineer.output_dir, ticker, f"{ticker}_features.csv")
                if os.path.exists(ticker_file):
                    engineer.processed_data[ticker] = pd.read_csv(ticker_file)
                    if 'Date' in engineer.processed_data[ticker].columns:
                        engineer.processed_data[ticker]['Date'] = pd.to_datetime(engineer.processed_data[ticker]['Date'])
                    engineer.create_plots(ticker, start_date=args.start_date, end_date=args.end_date)
                else:
                    print(f"No processed data found for {ticker}. Run without --plot-only to process data first.")
            except Exception as e:
                print(f"Error processing plots for {ticker}: {e}")
                continue
    else:
        # Process features and optionally create plots
        for ticker in tickers:
            try:
                engineer.process_ticker(ticker)
                engineer.save_features(ticker)
                
                if args.plot:
                    engineer.create_plots(ticker, start_date=args.start_date, end_date=args.end_date)
                
                print(f"Completed processing for {ticker}\n")
            except Exception as e:
                print(f"Error processing {ticker}: {e}\n")
                continue
    
    print("Feature engineering completed!")

if __name__ == "__main__":
    main()