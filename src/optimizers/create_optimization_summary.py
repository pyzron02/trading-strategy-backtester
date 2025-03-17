#!/usr/bin/env python3
# create_optimization_summary.py - Create summary of parameter optimization results

import os
import sys
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
if src_dir not in sys.path:
    sys.path.append(src_dir)

def parse_results_file(file_path):
    """Parse a results.txt file and extract the key information."""
    results = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Extract strategy name
        strategy_line = lines[0].strip()
        results['strategy'] = strategy_line.split(': ')[1]
        
        # Extract parameters
        params_line = lines[1].strip()
        params_str = params_line.split(': ')[1]
        
        print(f"File: {file_path}")
        print(f"Params string: {params_str}")
        
        # Use regex to extract parameter values
        params = {}
        # Match each parameter individually
        sma_match = re.search(r"'sma_period':\s*(\d+)", params_str)
        if sma_match:
            params['sma_period'] = int(sma_match.group(1))
            
        position_match = re.search(r"'position_size':\s*(\d+)", params_str)
        if position_match:
            params['position_size'] = int(position_match.group(1))
            
        stop_loss_match = re.search(r"'stop_loss':\s*([\d\.]+)", params_str)
        if stop_loss_match:
            params['stop_loss'] = float(stop_loss_match.group(1))
            
        take_profit_match = re.search(r"'take_profit':\s*([\d\.]+)", params_str)
        if take_profit_match:
            params['take_profit'] = float(take_profit_match.group(1))
        
        print(f"Extracted params: {params}")
        
        results['parameters'] = params
        
        # Extract tickers
        tickers_line = lines[2].strip()
        results['tickers'] = tickers_line.split(': ')[1]
        
        # Extract initial value
        initial_line = lines[3].strip()
        initial_value_str = initial_line.split('$')[1].replace(',', '')
        results['initial_value'] = float(initial_value_str)
        
        # Extract final value
        final_line = lines[4].strip()
        final_value_str = final_line.split('$')[1].replace(',', '')
        results['final_value'] = float(final_value_str)
        
        # Extract total return
        return_line = lines[5].strip()
        return_str = return_line.split(': ')[1].replace('%', '')
        results['total_return'] = float(return_str)
        
        # Extract benchmark return
        benchmark_line = lines[6].strip()
        benchmark_str = benchmark_line.split(': ')[1].replace('%', '')
        results['benchmark_return'] = float(benchmark_str)
        
        # Extract alpha
        alpha_line = lines[7].strip()
        alpha_str = alpha_line.split(': ')[1].replace('%', '')
        results['alpha'] = float(alpha_str)
    
    return results

def create_optimization_summary(results_dir):
    """Create a summary report of optimization results."""
    # Find all results.txt files
    results_files = glob.glob(os.path.join(results_dir, '**/results.txt'), recursive=True)
    
    if not results_files:
        print(f"No results.txt files found in {results_dir}")
        return
    
    # Parse all results files
    all_results = []
    for file_path in results_files:
        try:
            results = parse_results_file(file_path)
            all_results.append(results)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
    
    if not all_results:
        print("No results could be parsed successfully.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by total return (descending)
    df = df.sort_values('total_return', ascending=False)
    
    # Extract parameter columns
    param_cols = []
    for params in df['parameters']:
        for param_name in params:
            if param_name not in param_cols:
                param_cols.append(param_name)
    
    # Create parameter columns
    for param_name in param_cols:
        df[param_name] = df['parameters'].apply(lambda x: x.get(param_name, None))
    
    # Create summary report
    summary_file = os.path.join(results_dir, 'optimization_summary.csv')
    df.to_csv(summary_file, index=False)
    print(f"Summary report saved to {summary_file}")
    
    # Check if we have parameter columns
    if not param_cols:
        print("No parameter columns found. Skipping visualization.")
        return df
    
    # Create visualization of parameter impact
    fig, axes = plt.subplots(len(param_cols), 1, figsize=(10, 5 * len(param_cols)))
    
    if len(param_cols) == 1:
        axes = [axes]
    
    for i, param_name in enumerate(param_cols):
        ax = axes[i]
        param_values = sorted(df[param_name].unique())
        
        # Calculate average return for each parameter value
        avg_returns = []
        for value in param_values:
            avg_return = df[df[param_name] == value]['total_return'].mean()
            avg_returns.append(avg_return)
        
        # Plot
        ax.bar([str(v) for v in param_values], avg_returns)
        ax.set_title(f'Impact of {param_name} on Total Return')
        ax.set_xlabel(param_name)
        ax.set_ylabel('Average Total Return (%)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = os.path.join(results_dir, 'parameter_impact.png')
    plt.savefig(viz_file)
    print(f"Parameter impact visualization saved to {viz_file}")
    
    # Print top 5 strategies
    print("\nTop 5 Strategies:")
    for i, row in df.head(5).iterrows():
        print(f"{i+1}. Total Return: {row['total_return']:.2f}%, Parameters: {row['parameters']}")
    
    # Print bottom 5 strategies
    print("\nBottom 5 Strategies:")
    for i, row in df.tail(5).iterrows():
        print(f"{i+1}. Total Return: {row['total_return']:.2f}%, Parameters: {row['parameters']}")
    
    return df

def main():
    """Main function to parse arguments and create summary report."""
    parser = argparse.ArgumentParser(description='Create a summary report of optimization results')
    parser.add_argument('results_dir', type=str, help='Directory containing optimization results')
    
    args = parser.parse_args()
    
    # Create summary report
    create_optimization_summary(args.results_dir)
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 