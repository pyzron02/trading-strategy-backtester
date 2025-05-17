#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frontend application for the trading strategy backtester.
"""
import os
import sys
import json
import subprocess
import glob
import re
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def load_dotenv():
        print("Warning: python-dotenv package not installed. Environment variables from .env file will not be loaded.")
        pass
    load_dotenv()

# Add the trading-strategy-backtester to the path
# Support both Docker and local development paths
project_root = os.getenv(
    'BACKTESTER_ROOT',
    '/home/pyzron02/trading-strategy-backtester')

# In Docker environment, the backtester will be mounted at
# /trading-strategy-backtester
if os.path.exists(
        '/trading-strategy-backtester') and not os.path.exists(project_root):
    project_root = '/trading-strategy-backtester'
    print(f"Using Docker project root: {project_root}")

sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# Import the registry directly from its file to avoid module resolution issues
try:
    from src.strategies.registry import get_registered_strategies
except ImportError:
    def get_registered_strategies():
        return [{"name": "MACrossover", "version": "1.0"},
                {"name": "AuctionMarket", "version": "1.0"},
                {"name": "MultiPosition", "version": "1.0"}]

app = Flask(__name__)
app.secret_key = os.getenv(
    'SECRET_KEY',
    'trading_strategy_backtester_secret_key')

# Default parameters from environment variables
DEFAULT_START_DATE = os.getenv('DEFAULT_START_DATE', '2020-01-01')
DEFAULT_END_DATE = os.getenv('DEFAULT_END_DATE', '2025-01-01')
DEFAULT_NUM_SIMULATIONS = int(os.getenv('DEFAULT_NUM_SIMULATIONS', '100'))
DEFAULT_WALK_FORWARD_WINDOWS = int(
    os.getenv('DEFAULT_WALK_FORWARD_WINDOWS', '5'))
DEFAULT_IN_SAMPLE_PCT = float(os.getenv('DEFAULT_IN_SAMPLE_PCT', '0.7'))

# Configure template folder
app.template_folder = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    'templates')


def get_workflow_type_from_folder(folder_name):
    """Extract workflow type from a folder name

    Folder names typically follow pattern: StrategyName_workflow-type_timestamp_hash
    For example: MultiPosition_complete_20250421_204646_6a8f39e2

    Returns:
        str: Workflow type (simple, optimization, monte_carlo, walk_forward, complete)
             or None if not detected
    """
    # Common workflow types with their variants
    workflow_mapping = {
        "simple": ["simple"],
        "optimization": ["optimization", "optimize", "opt"],
        "monte_carlo": ["monte_carlo", "montecarlo", "monte-carlo"],
        "walkforward": ["walkforward", "walk-forward", "wf"],
        "complete": ["complete", "full", "all"]
    }

    # Normalize the folder name to lowercase for case-insensitive matching
    folder_lower = folder_name.lower()

    # First attempt: look for exact matches with underscore patterns
    for workflow_type, variants in workflow_mapping.items():
        for variant in variants:
            # Check for pattern like "_variant_" (standard format)
            if f"_{variant}_" in folder_lower:
                return workflow_type

    # Second attempt: look for the variants anywhere in the name
    for workflow_type, variants in workflow_mapping.items():
        for variant in variants:
            if variant in folder_lower:
                return workflow_type

    # Third attempt: check for directory structure to infer the type
    # Especially useful for complete workflow runs that have subdirectories
    if os.path.isdir(os.path.join(project_root, 'output', folder_name)):
        folder_path = os.path.join(project_root, 'output', folder_name)

        # Look for subdirectories that indicate workflow type
        if os.path.exists(os.path.join(folder_path, '03_walkforward')):
            return "complete"  # Complete workflow with walk forward analysis
        elif os.path.exists(os.path.join(folder_path, 'walkforward_summary.txt')) or os.path.exists(os.path.join(folder_path, 'walk_forward_summary.txt')):
            return "walkforward"
        elif os.path.exists(os.path.join(folder_path, '04_monte_carlo')) or os.path.exists(os.path.join(folder_path, '03_monte_carlo')):
            return "complete"  # Complete workflow with Monte Carlo analysis
        elif os.path.exists(os.path.join(folder_path, 'monte_carlo_summary.txt')):
            return "monte_carlo"
        elif os.path.exists(os.path.join(folder_path, '02_optimization')) or any('optimization_summary' in f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))):
            return "optimization"

    # If no match, return None
    return None


def get_output_folders():
    """Get all output folders from the backtester"""
    output_dir = os.path.join(project_root, 'output')
    folders = []

    for folder in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder)
        if os.path.isdir(folder_path) and not folder.startswith('.'):
            # Extract strategy and workflow type from folder name
            folder_parts = folder.split('_')
            if len(folder_parts) >= 2:
                strategy_name = folder_parts[0]
                workflow_type = get_workflow_type_from_folder(folder)

                # Extract timestamp if available
                timestamp_match = re.search(r'(\d{8}_\d{6})', folder)
                timestamp = timestamp_match.group(
                    1) if timestamp_match else None

                # Get creation time as fallback
                creation_time = datetime.fromtimestamp(
                    os.path.getctime(folder_path))
                display_time = timestamp if timestamp else creation_time.strftime(
                    '%Y%m%d_%H%M%S')

                folders.append({
                    'name': folder,
                    'path': folder_path,
                    'strategy': strategy_name,
                    'workflow': workflow_type,
                    'timestamp': display_time,
                    'full_path': os.path.abspath(folder_path)
                })

    # Sort by most recent first
    folders.sort(key=lambda x: x['timestamp'], reverse=True)
    return folders


def read_log_file(folder_path):
    """Read log file from a results folder"""
    logs = []
    for file in os.listdir(folder_path):
        if file.endswith('.log'):
            log_path = os.path.join(folder_path, file)
            try:
                with open(log_path, 'r') as f:
                    logs.append({
                        'name': file,
                        'content': f.read(),
                        'path': log_path
                    })
            except Exception as e:
                logs.append({
                    'name': file,
                    'content': f"Error reading log file: {str(e)}",
                    'path': log_path
                })
    return logs


def read_summary_files(folder_path):
    """Read summary files from a results folder"""
    summaries = []
    for file in os.listdir(folder_path):
        if ('summary' in file.lower() or 'error_report' in file.lower() or 'error_summary' in file.lower()) and file.endswith('.txt') and 'stage_error_report' not in file.lower():
            summary_path = os.path.join(folder_path, file)
            try:
                with open(summary_path, 'r') as f:
                    content = f.read()

                # Process Monte Carlo summary files to enhance display
                if 'monte_carlo_summary' in file.lower():
                    # Extract key sections for better formatting on the
                    # frontend
                    sections = {}
                    current_section = "header"
                    sections[current_section] = []

                    # Parse the summary into sections
                    for line in content.splitlines():
                        if line.startswith(
                                '=====') and current_section == "header":
                            current_section = "strategy_info"
                            continue
                        elif line.startswith('=====') and "SIMULATION PARAMETERS" in line:
                            current_section = "simulation_parameters"
                            continue
                        elif line.startswith('-----') and "Portfolio Statistics" in line:
                            current_section = "portfolio_statistics"
                            continue
                        elif line.startswith('-----') and "Simulation Results" in line:
                            current_section = "simulation_results"
                            continue
                        elif line.startswith('-----') and "Confidence Intervals" in line:
                            current_section = "confidence_intervals"
                            continue
                        elif line.startswith('-----') and "Risk Metrics" in line:
                            current_section = "risk_metrics"
                            continue
                        elif line.startswith('-----') and "Probability Metrics" in line:
                            current_section = "probability_metrics"
                            continue
                        elif line.startswith('=====') and "WORKFLOW STATUS" in line:
                            current_section = "workflow_status"
                            continue

                        # Store line in current section
                        if current_section not in sections:
                            sections[current_section] = []
                        sections[current_section].append(line)

                    # Add the formatted summary with additional metadata
                    summaries.append({
                        'name': file,
                        'content': content,
                        'path': summary_path,
                        'type': 'monte_carlo',
                        'sections': sections
                    })
                # Process error summary and error report files with special styling
                elif 'error_summary' in file.lower() or 'error_report' in file.lower():
                    summaries.append({
                        'name': file,
                        'content': content,
                        'path': summary_path,
                        'type': 'error'
                    })
                else:
                    # Regular summary file
                    summaries.append({
                        'name': file,
                        'content': content,
                        'path': summary_path,
                        'type': 'standard'
                    })
            except Exception as e:
                summaries.append({
                    'name': file,
                    'content': f"Error reading summary file: {str(e)}",
                    'path': summary_path,
                    'type': 'error'
                })
    return summaries


def get_equity_curve(output_path):
    """
    Get equity curve data from output folder, prioritizing by workflow type.

    Different workflows store equity curves in different locations:
    - simple: Looks for equity_curve.csv in main folder
    - optimization: Checks 02_optimization or for best parameters
    - monte_carlo: Searches in 03_monte_carlo/backtest
    - complete: Looks in 04_optimized_backtest for final result

    Args:
        output_path: Path to the output folder

    Returns:
        dict: Dictionary containing dates, equity values, moving averages and DataFrame,
              or None if not found
    """
    # Get the workflow type from the folder name
    folder_name = os.path.basename(output_path)
    workflow_type = get_workflow_type_from_folder(folder_name)

    # Define priority folders based on workflow type
    priority_folders = {
        "simple": [""],  # Just the main folder
        "optimization": ["02_optimization", ""],
        "monte_carlo": ["03_monte_carlo/backtest", "03_monte_carlo", ""],
        # Check out-of-sample first for walkforward
        "walkforward": ["out_sample", "combined", ""],
        "complete": ["05_optimized_backtest", "04_optimized_backtest", "03_walkforward/out_sample", "01_simple_backtest", ""],
        None: [""]  # Default to main folder if workflow type not detected
    }

    # Try the priority folders first
    folders_to_check = priority_folders.get(workflow_type, [""])

    print(f"Detected workflow type: {workflow_type} for {output_path}")
    print(f"Checking priority folders: {folders_to_check}")

    # First check priority folders based on workflow type
    for folder in folders_to_check:
        equity_curve_path = os.path.join(
            output_path, folder, "equity_curve.csv")
        print(f"Checking for equity curve at: {equity_curve_path}")
        if os.path.exists(equity_curve_path):
            try:
                print(f"Found equity curve at: {equity_curve_path}")
                df = pd.read_csv(equity_curve_path)

                # Process data for chart display
                # Convert Date column if it exists
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    # Format dates for display
                    dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
                else:
                    # Use index as dates if no Date column
                    dates = [str(i) for i in range(len(df))]

                # Get equity values
                if 'Value' in df.columns:
                    equity = df['Value'].tolist()
                elif 'Equity' in df.columns:
                    equity = df['Equity'].tolist()
                else:
                    # Try to find any column that might contain equity values
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        equity = df[numeric_cols[0]].tolist()
                    else:
                        print(
                            f"No suitable numeric column found in {equity_curve_path}")
                        continue  # Try next folder

                # Calculate moving averages
                ma_periods = [20, 50, 200]  # Short, medium, long-term MAs
                ma_data = {}

                for period in ma_periods:
                    if len(equity) > period:
                        try:
                            # Calculate simple moving average and properly handle NaNs at the beginning
                            # Use forward fill (ffill) method instead of None
                            series = pd.Series(equity)
                            ma_series = series.rolling(window=period).mean()
                            # Replace NaN values with the first valid value
                            # (forward fill)
                            ma_values = ma_series.fillna(
                                method='ffill').tolist()
                            # If there are still NaNs at the beginning, replace
                            # with the first valid value
                            if pd.isna(
                                    ma_values[0]) and any(
                                    not pd.isna(x) for x in ma_values):
                                first_valid = next(
                                    x for x in ma_values if not pd.isna(x))
                                ma_values = [
                                    first_valid if pd.isna(x) else x for x in ma_values]
                            ma_data[f'MA{period}'] = ma_values
                        except Exception as e:
                            print(f"Error calculating {period}-day MA: {e}")
                            # Skip this MA period but continue with others

                return {
                    'dates': dates,
                    'equity': equity,
                    'moving_averages': ma_data,
                    'df': df  # Return the original dataframe for potential further analysis
                }
            except Exception as e:
                print(
                    f"Error loading equity curve from {equity_curve_path}: {e}")
                # Continue to next folder rather than returning None

    # If still not found, do a recursive search
    print("Priority folders search failed, doing recursive search...")
    for root, dirs, files in os.walk(output_path):
        if "equity_curve.csv" in files:
            equity_curve_path = os.path.join(root, "equity_curve.csv")
            try:
                print(
                    f"Found equity curve during recursive search at: {equity_curve_path}")
                df = pd.read_csv(equity_curve_path)

                # Check if file is empty or malformed
                if df.empty:
                    print(
                        f"Empty equity curve file found at {equity_curve_path}")
                    continue

                # Process data for chart display
                # Convert Date column if it exists
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    # Format dates for display
                    dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
                else:
                    # Use index as dates if no Date column
                    dates = [str(i) for i in range(len(df))]

                # Get equity values
                if 'Value' in df.columns:
                    equity = df['Value'].tolist()
                elif 'Equity' in df.columns:
                    equity = df['Equity'].tolist()
                else:
                    # Try to find any column that might contain equity values
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        equity = df[numeric_cols[0]].tolist()
                    else:
                        print(
                            f"No suitable numeric column found in {equity_curve_path}")
                        continue  # Try next file

                # Calculate moving averages
                ma_periods = [20, 50, 200]  # Short, medium, long-term MAs
                ma_data = {}

                for period in ma_periods:
                    if len(equity) > period:
                        try:
                            # Calculate simple moving average and properly handle NaNs at the beginning
                            # Use forward fill (ffill) method instead of None
                            series = pd.Series(equity)
                            ma_series = series.rolling(window=period).mean()
                            # Replace NaN values with the first valid value
                            # (forward fill)
                            ma_values = ma_series.fillna(
                                method='ffill').tolist()
                            # If there are still NaNs at the beginning, replace
                            # with the first valid value
                            if pd.isna(
                                    ma_values[0]) and any(
                                    not pd.isna(x) for x in ma_values):
                                first_valid = next(
                                    x for x in ma_values if not pd.isna(x))
                                ma_values = [
                                    first_valid if pd.isna(x) else x for x in ma_values]
                            ma_data[f'MA{period}'] = ma_values
                        except Exception as e:
                            print(f"Error calculating {period}-day MA: {e}")
                            # Skip this MA period but continue with others

                return {
                    'dates': dates,
                    'equity': equity,
                    'moving_averages': ma_data,
                    'df': df  # Return the original dataframe for potential further analysis
                }
            except Exception as e:
                print(
                    f"Error loading equity curve from {equity_curve_path}: {e}")
                # Continue search

    print(f"No equity curve found for {output_path}")
    return None


def get_trade_log(folder_path):
    """Get trade log CSV if available"""
    for file in os.listdir(folder_path):
        if file == 'trade_log.csv':
            try:
                trade_path = os.path.join(folder_path, file)
                if os.path.exists(trade_path) and os.path.isfile(trade_path):
                    df = pd.read_csv(trade_path)
                    return {
                        'path': trade_path,
                        'data': df.to_dict(orient='records')
                    }
            except Exception as e:
                print(f"Error loading trade log: {str(e)}")
                pass

    # Check subdirectories
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file == 'trade_log.csv':
                    try:
                        trade_path = os.path.join(subdir_path, file)
                        if os.path.exists(
                                trade_path) and os.path.isfile(trade_path):
                            df = pd.read_csv(trade_path)
                            return {
                                'path': trade_path,
                                'data': df.to_dict(orient='records')
                            }
                    except Exception as e:
                        print(
                            f"Error loading trade log from subdirectory: {str(e)}")
                        pass

    return None


def get_optimization_results(folder_path):
    """Get optimization results if available"""
    # First check for trials CSV
    for file in os.listdir(folder_path):
        if 'trials' in file.lower() and file.endswith('.csv'):
            try:
                trials_path = os.path.join(folder_path, file)
                df = pd.read_csv(trials_path)
                return {
                    'type': 'trials',
                    'path': trials_path,
                    'data': df.to_dict(orient='records')
                }
            except Exception:
                pass

    # Check for best parameters
    for file in os.listdir(folder_path):
        if 'best_params' in file.lower() and (
                file.endswith('.json') or file.endswith('.txt')):
            best_params_path = os.path.join(folder_path, file)
            try:
                if file.endswith('.json'):
                    with open(best_params_path, 'r') as f:
                        params = json.load(f)
                    return {
                        'type': 'best_params',
                        'path': best_params_path,
                        'data': params
                    }
                else:  # .txt file
                    with open(best_params_path, 'r') as f:
                        content = f.read()
                    return {
                        'type': 'best_params_text',
                        'path': best_params_path,
                        'data': content
                    }
            except Exception:
                pass

    # Check subdirectories
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path) and (
                'optimization' in subdir.lower() or '02_optimization' in subdir.lower()):
            return get_optimization_results(subdir_path)

    return None


def get_monte_carlo_charts(folder_path):
    """Get Monte Carlo charts if available"""
    print(f"Looking for Monte Carlo charts in {folder_path}")
    charts = []

    # Function to check if a file is a monte carlo related chart
    def is_monte_carlo_chart(filename):
        # Check for drawdowns_comparison.html specifically and exclude it
        if 'drawdowns_comparison' in filename.lower():
            return False

        # Include other monte carlo related charts
        return (
            filename.endswith('.html') or filename.endswith('.png')) and any(
            x in filename.lower() for x in [
                'monte_carlo',
                'return_distribution',
                'dashboard'])

    # Check main directory first
    for file in os.listdir(folder_path):
        if is_monte_carlo_chart(file):
            file_path = os.path.join(folder_path, file)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                print(f"Found chart: {file_path}")
                charts.append({
                    'name': file,
                    'path': file_path,
                    'type': 'html' if file.endswith('.html') else 'png'
                })

    # Look for specific directory names that might contain charts
    chart_dirs = ['03_monte_carlo', 'monte_carlo']
    for chart_dir in chart_dirs:
        subdir_path = os.path.join(folder_path, chart_dir)
        if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
            print(f"Checking specific directory: {subdir_path}")
            for file in os.listdir(subdir_path):
                if is_monte_carlo_chart(file):
                    file_path = os.path.join(subdir_path, file)
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        print(f"Found chart: {file_path}")
                        charts.append({
                            'name': file,
                            'path': file_path,
                            'type': 'html' if file.endswith('.html') else 'png'
                        })

    # Check all subdirectories (recursive)
    if not charts:
        for subdir in os.listdir(folder_path):
            subdir_path = os.path.join(folder_path, subdir)
            if os.path.isdir(subdir_path):
                print(f"Checking subdirectory: {subdir_path}")
                # Look in this subdirectory
                for file in os.listdir(subdir_path):
                    if is_monte_carlo_chart(file):
                        file_path = os.path.join(subdir_path, file)
                        if os.path.exists(
                                file_path) and os.path.isfile(file_path):
                            print(f"Found chart: {file_path}")
                            charts.append({
                                'name': file,
                                'path': file_path,
                                'type': 'html' if file.endswith('.html') else 'png'
                            })

                # Check one level deeper (for nested output directories)
                for nested_dir in os.listdir(subdir_path):
                    nested_path = os.path.join(subdir_path, nested_dir)
                    if os.path.isdir(nested_path):
                        print(f"Checking nested directory: {nested_path}")
                        for file in os.listdir(nested_path):
                            if is_monte_carlo_chart(file):
                                file_path = os.path.join(nested_path, file)
                                if os.path.exists(
                                        file_path) and os.path.isfile(file_path):
                                    print(f"Found chart: {file_path}")
                                    charts.append({
                                        'name': file,
                                        'path': file_path,
                                        'type': 'html' if file.endswith('.html') else 'png'
                                    })

    # First separate charts by their base name (without extension)
    chart_groups = {}
    for chart in charts:
        base_name = chart['name'].replace('.html', '').replace('.png', '')
        if base_name not in chart_groups:
            chart_groups[base_name] = []
        chart_groups[base_name].append(chart)

    # Then prioritize HTML files over PNG for each base name
    prioritized_charts = []
    for base_name, chart_list in chart_groups.items():
        # Find HTML files first - always include ALL HTML charts
        html_charts = [c for c in chart_list if c['type'] == 'html']
        if html_charts:
            # Include all HTML versions
            for html_chart in html_charts:
                prioritized_charts.append(html_chart)
                print(f"Added HTML chart: {html_chart['name']}")
        else:
            # Otherwise use PNG version (just one)
            png_charts = [c for c in chart_list if c['type'] == 'png']
            if png_charts:
                prioritized_charts.append(png_charts[0])
                print(
                    f"Selected PNG version of {base_name} (HTML not available)")

    print(
        f"Found {len(charts)} total charts, selected {len(prioritized_charts)} prioritized charts")
    for chart in prioritized_charts:
        print(f"Selected chart: {chart['name']} ({chart['type']})")

    return prioritized_charts


def get_equity_curve_description(workflow_type):
    """Get description of equity curve based on workflow type"""
    base_description = ""
    if workflow_type == "simple":
        base_description = """
        <h5>Simple Workflow Equity Curve</h5>
        <p>This equity curve represents the actual balance of your trading account over time.
        Each point corresponds to a day/timestamp in the backtest period, showing how your capital
        would have grown or declined based on the trading strategy with fixed parameters.</p>
        """
    elif workflow_type == "optimization":
        base_description = """
        <h5>Optimization Workflow Equity Curve</h5>
        <p>This equity curve shows the performance of your strategy with the <strong>best parameters</strong>
        found during optimization. After testing many parameter combinations, the optimizer selected
        the parameters that maximize your chosen metric (Sharpe ratio, total return, etc.). This curve
        demonstrates how those optimal parameters would have performed historically.</p>
        """
    elif workflow_type == "monte_carlo":
        base_description = """
        <h5>Monte Carlo Workflow Equity Curve</h5>
        <p>This equity curve represents the <strong>average or median</strong> equity path across
        multiple simulations. In Monte Carlo analysis, the price data is randomized/permuted to test
        strategy robustness. This gives a more realistic picture of expected performance by accounting
        for market randomness. Check the Monte Carlo tab for the distribution of possible outcomes.</p>
        """
    elif workflow_type == "walkforward":
        base_description = """
        <h5>Walk Forward Workflow Equity Curve</h5>
        <p>This equity curve shows performance using <strong>walk forward analysis</strong>. In this method,
        the strategy is optimized on an in-sample period and then tested on an out-of-sample period repeatedly
        across the entire time series. This helps prevent overfitting by continually validating the strategy's
        performance on unseen data. The resulting equity curve represents the performance on out-of-sample data only.</p>
        <p>Walk forward analysis can be performed in two modes:</p>
        <ul>
            <li><strong>Rolling Walk Forward</strong>: Both in-sample and out-of-sample windows move forward in time,
            with new data being incorporated into each window.</li>
            <li><strong>Anchored Walk Forward</strong>: The in-sample period starts at a fixed date, but grows larger
            with each iteration, while the out-of-sample window moves forward.</li>
        </ul>
        <p>This methodology attempts to simulate how a strategy would perform in real-time trading, where you would
        optimize on historical data and then trade on new, unseen market conditions. Performance metrics calculated
        only on the out-of-sample periods provide a more realistic estimation of future performance.</p>
        """
    elif workflow_type == "complete":
        base_description = """
        <h5>Complete Workflow Equity Curve</h5>
        <p>This equity curve combines the benefits of optimization, Monte Carlo testing, and walk forward analysis.
        It shows the performance of optimized parameters that have been validated through both Monte Carlo simulations
        and walk forward testing. This is the most comprehensive view of how your strategy might perform in real market
        conditions, accounting for both market randomness and parameter stability.</p>
        """
    else:
        base_description = """
        <h5>Equity Curve</h5>
        <p>This chart shows the growth of your trading account over the backtest period based on
        the strategy's trades. It's a visual representation of cumulative performance over time.</p>
        """

    return base_description


def get_walk_forward_results(folder_path):
    """Get walk forward analysis results if available"""
    print(f"Looking for walk forward results in {folder_path}")

    # Extract folder name from path
    folder_name = os.path.basename(folder_path)
    workflow_type = get_workflow_type_from_folder(folder_name)

    # STANDARDIZED APPROACH: Check all possible walkforward locations in a
    # consistent way
    possible_wf_dirs = [
        os.path.join(folder_path, "03_walkforward"),
        os.path.join(folder_path, "03_walk_forward"),
        os.path.join(folder_path, "walkforward"),
        os.path.join(folder_path, "walk_forward"),
        folder_path  # For standalone walkforward runs
    ]

    walk_forward_dir = None
    # First pass: look for directories with walkforward content
    for potential_dir in possible_wf_dirs:
        if os.path.exists(potential_dir):
            # Check if this directory has walkforward files using a more
            # inclusive approach
            has_wf_files = False
            if os.path.isdir(potential_dir):
                has_wf_files = any(
                    'walkforward' in f.lower() or 'walk_forward' in f.lower() or
                    'window' in f.lower() or 'in_sample' in f.lower() or
                    'out_sample' in f.lower()
                    for f in os.listdir(potential_dir)
                    if os.path.isfile(os.path.join(potential_dir, f)) and
                    (f.endswith('.txt') or f.endswith('.html') or f.endswith('.csv'))
                )

                # Also check for walkforward subdirectories
                has_wf_dirs = any(
                    'walkforward' in d.lower() or 'walk_forward' in d.lower() or
                    'window' in d.lower() or 'in_sample' in d.lower() or
                    'out_sample' in d.lower()
                    for d in os.listdir(potential_dir)
                    if os.path.isdir(os.path.join(potential_dir, d))
                )

                if has_wf_files or has_wf_dirs:
                    walk_forward_dir = potential_dir
                    print(
                        f"Found walk forward directory at {walk_forward_dir}")
                    break

    # If not found in predefined locations, search recursively for any
    # directory with walkforward naming
    if not walk_forward_dir:
        for root, dirs, files in os.walk(folder_path):
            dir_name = os.path.basename(root).lower()
            if ('walk_forward' in dir_name or 'walkforward' in dir_name):
                walk_forward_dir = root
                print(
                    f"Found walk forward directory from recursive search: {walk_forward_dir}")
                break

    if not walk_forward_dir:
        print(f"No walk forward directory found in {folder_path}")
        return None

    results = {
        'summary': None,
        'performance_metrics': None,
        'in_sample_data': None,
        'out_sample_data': None,
        'window_results': [],
        'charts': [],
        'comparison_files': []
    }

    # STANDARDIZED DATA COLLECTION

    # 1. Look for walkforward summary files across all possible locations
    summary_files = []
    # Check main walkforward directory
    for file in os.listdir(walk_forward_dir):
        if ('summary' in file.lower() or 'results' in file.lower()
                or 'metrics' in file.lower()) and file.endswith('.txt'):
            summary_files.append(os.path.join(walk_forward_dir, file))

    # Also check for summary files in the parent directory for complete
    # workflows
    if workflow_type == 'complete' and walk_forward_dir != folder_path:
        for file in os.listdir(folder_path):
            if 'walkforward' in file.lower() and 'summary' in file.lower() and file.endswith('.txt'):
                summary_files.append(os.path.join(folder_path, file))

    # Process the first valid summary file found
    for summary_path in summary_files:
        try:
            with open(summary_path, 'r') as f:
                content = f.read()

            # Parse summary to extract performance metrics in a standardized
            # way
            metrics = {}
            reading_metrics = False

            for line in content.splitlines():
                if any(
                    header in line for header in [
                        'Performance Metrics:',
                        'Out-of-Sample Performance:',
                        'Summary Statistics:']):
                    reading_metrics = True
                    continue
                elif reading_metrics and ('---' in line or '===' in line or len(line.strip()) == 0):
                    reading_metrics = False
                    continue
                elif reading_metrics and ':' in line:
                    try:
                        key, value = line.split(':', 1)
                        metrics[key.strip()] = value.strip()
                    except Exception as e:
                        print(f"Error parsing metric line: {line}, {e}")

            results['summary'] = {
                'path': summary_path,
                'content': content,
                'name': os.path.basename(summary_path),
                'parsed_metrics': metrics
            }
            print(
                f"Found walkforward summary file: {os.path.basename(summary_path)}")
            break
        except Exception as e:
            print(f"Error reading walkforward summary file: {e}")

    # 2. Standardized approach to finding comparison files
    comparison_locations = [walk_forward_dir]
    if workflow_type == 'complete':
        # Also check parent folder for complete workflows
        comparison_locations.append(folder_path)

    for location in comparison_locations:
        if os.path.exists(location):
            for file in os.listdir(location):
                if any(
                    term in file.lower() for term in [
                        'comparison',
                        'combined',
                        'metrics']) and file.endswith('.csv'):
                    comparison_path = os.path.join(location, file)
                    try:
                        # Read CSV into pandas DataFrame
                        df = pd.read_csv(comparison_path)
                        results['comparison_files'].append({
                            'path': comparison_path,
                            'name': file,
                            'data': df.to_dict(orient='records')
                        })
                        print(f"Found comparison file: {file}")
                    except Exception as e:
                        print(f"Error reading comparison file {file}: {e}")

    # 3. Look for in-sample and out-of-sample data consistently across all
    # workflow types

    # Function to find data directories across workflow types
    def find_data_dirs(base_dir, dir_type):
        """Find all directories of a specific type across the folder structure"""
        found_dirs = []

        # Check direct subdirectory
        direct_path = os.path.join(base_dir, dir_type)
        if os.path.exists(direct_path) and os.path.isdir(direct_path):
            found_dirs.append(direct_path)

        # Look in walk_forward directory
        if base_dir != walk_forward_dir:
            wf_path = os.path.join(walk_forward_dir, dir_type)
            if os.path.exists(wf_path) and os.path.isdir(wf_path):
                found_dirs.append(wf_path)

        # Check window subdirectories
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and (
                    'window' in item.lower() or 'period' in item.lower()):
                window_path = os.path.join(item_path, dir_type)
                if os.path.exists(window_path) and os.path.isdir(window_path):
                    found_dirs.append(window_path)

        return found_dirs

    # Find in-sample and out-of-sample directories
    in_sample_dirs = find_data_dirs(walk_forward_dir, 'in_sample')
    out_sample_dirs = find_data_dirs(walk_forward_dir, 'out_sample')

    # Process in-sample data
    for in_dir in in_sample_dirs:
        if not results['in_sample_data']:  # Only use the first valid data found
            results_path = os.path.join(in_dir, 'results.txt')
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        content = f.read()
                    results['in_sample_data'] = {
                        'path': results_path,
                        'content': content,
                        'name': 'results.txt'
                    }
                    print(f"Found in-sample results file: {results_path}")
                except Exception as e:
                    print(f"Error reading in-sample results: {e}")

            # Try to get equity curve
            equity_path = os.path.join(in_dir, 'equity_curve.csv')
            if os.path.exists(equity_path):
                try:
                    df = pd.read_csv(equity_path)
                    results['in_sample_equity'] = {
                        'path': equity_path,
                        'data': df.to_dict(orient='records')
                    }
                    print(f"Found in-sample equity curve file: {equity_path}")
                except Exception as e:
                    print(f"Error reading in-sample equity curve: {e}")

    # Process out-sample data
    for out_dir in out_sample_dirs:
        if not results['out_sample_data']:  # Only use the first valid data found
            results_path = os.path.join(out_dir, 'results.txt')
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        content = f.read()
                    results['out_sample_data'] = {
                        'path': results_path,
                        'content': content,
                        'name': 'results.txt'
                    }
                    print(f"Found out-sample results file: {results_path}")
                except Exception as e:
                    print(f"Error reading out-sample results: {e}")

            # Try to get equity curve
            equity_path = os.path.join(out_dir, 'equity_curve.csv')
            if os.path.exists(equity_path):
                try:
                    df = pd.read_csv(equity_path)
                    results['out_sample_equity'] = {
                        'path': equity_path,
                        'data': df.to_dict(orient='records')
                    }
                    print(f"Found out-sample equity curve file: {equity_path}")
                except Exception as e:
                    print(f"Error reading out-sample equity curve: {e}")

    # 4. STANDARDIZED WINDOW PROCESSING: Look for individual window results
    # consistently
    window_dirs = []

    # First, check for direct window folders
    for item in os.listdir(walk_forward_dir):
        item_path = os.path.join(walk_forward_dir, item)
        if os.path.isdir(item_path) and (
                'window' in item.lower() or 'period' in item.lower()):
            window_dirs.append(item_path)

    # If in a complete workflow but no windows found directly, try more
    # thorough search
    if not window_dirs and workflow_type == 'complete':
        # Look for deeper nested window folders
        for root, dirs, files in os.walk(walk_forward_dir):
            for dir_name in dirs:
                if 'window' in dir_name.lower() or 'period' in dir_name.lower():
                    window_path = os.path.join(root, dir_name)
                    if window_path not in window_dirs:
                        window_dirs.append(window_path)

    # If we found window directories, process them
    if window_dirs:
        print(f"Found {len(window_dirs)} window directories")

        # Sort window directories by name/number if possible
        window_dirs.sort(
            key=lambda x: int(
                re.search(
                    r'(\d+)',
                    os.path.basename(x)).group(1)) if re.search(
                r'(\d+)',
                os.path.basename(x)) else os.path.basename(x))

        # Process each window directory
        for window_dir in window_dirs:
            window_name = os.path.basename(window_dir)
            window_result = {
                'name': window_name,
                'path': window_dir,
                'in_sample': None,
                'out_sample': None,
                'charts': []
            }

            # Check for in-sample and out-of-sample subdirectories
            in_sample_dir = os.path.join(window_dir, 'in_sample')
            out_sample_dir = os.path.join(window_dir, 'out_sample')

            # More thorough search for in-sample data
            if os.path.exists(in_sample_dir):
                # Look for results across multiple file types
                in_sample_files = [
                    os.path.join(in_sample_dir, 'results.txt'),
                    os.path.join(in_sample_dir, 'summary.txt'),
                    os.path.join(in_sample_dir, 'metrics.txt'),
                    os.path.join(in_sample_dir, 'optimization_results.txt'),
                    os.path.join(in_sample_dir, 'best_params.json'),
                    os.path.join(in_sample_dir, 'best_params.txt')
                ]

                for file_path in in_sample_files:
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                            window_result['in_sample'] = {
                                'path': file_path,
                                'content': content,
                                'name': os.path.basename(file_path)
                            }
                            break
                        except Exception as e:
                            print(f"Error reading in-sample window file: {e}")

            # More thorough search for out-of-sample data
            if os.path.exists(out_sample_dir):
                # Look for results across multiple file types
                out_sample_files = [
                    os.path.join(out_sample_dir, 'results.txt'),
                    os.path.join(out_sample_dir, 'summary.txt'),
                    os.path.join(out_sample_dir, 'metrics.txt'),
                    os.path.join(out_sample_dir, 'performance.txt')
                ]

                for file_path in out_sample_files:
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                            window_result['out_sample'] = {
                                'path': file_path,
                                'content': content,
                                'name': os.path.basename(file_path)
                            }
                            break
                        except Exception as e:
                            print(f"Error reading out-sample window file: {e}")

            # Look for equity curves and other charts in a standardized way
            for subdir in [window_dir, in_sample_dir, out_sample_dir]:
                if os.path.exists(subdir):
                    for file in os.listdir(subdir):
                        if file.endswith(('.png', '.jpg', '.jpeg', '.html')) and (
                            'curve' in file.lower() or 'chart' in file.lower() or
                                'plot' in file.lower() or 'equity' in file.lower()):
                            chart_path = os.path.join(subdir, file)
                            window_result['charts'].append({
                                'path': chart_path,
                                'name': file,
                                'type': 'html' if file.endswith('.html') else 'image'
                            })

            results['window_results'].append(window_result)

    # 5. STANDARDIZED CHART COLLECTION - Gather all visualization files in a
    # consistent way

    # Define a function to find all visualization files
    def find_visualization_files(base_dir, max_depth=2, current_depth=0):
        """Find all visualization files recursively up to a max depth"""
        found_files = []

        if current_depth > max_depth:
            return found_files

        if os.path.exists(base_dir) and os.path.isdir(base_dir):
            for file in os.listdir(base_dir):
                file_path = os.path.join(base_dir, file)

                # Check if it's a visualization file
                if os.path.isfile(file_path) and file.endswith(('.html', '.png', '.jpg', '.jpeg')) and (
                    'equity' in file.lower() or 'chart' in file.lower() or
                    'plot' in file.lower() or 'curve' in file.lower() or
                    'performance' in file.lower() or 'comparison' in file.lower() or
                        'visualization' in file.lower() or 'dashboard' in file.lower()):

                    found_files.append({
                        'path': file_path,
                        'name': file,
                        'type': 'html' if file.endswith('.html') else 'image'
                    })

                # Recursively search directories
                elif os.path.isdir(file_path) and not file.startswith('.'):
                    sub_files = find_visualization_files(
                        file_path, max_depth, current_depth + 1)
                    found_files.extend(sub_files)

        return found_files

    # First find all visualization files in the walk_forward directory
    all_viz_files = find_visualization_files(walk_forward_dir)

    # If in complete workflow, also check the main folder for visualization
    # files
    if workflow_type == 'complete' and walk_forward_dir != folder_path:
        main_viz_files = find_visualization_files(
            folder_path, max_depth=1)  # Only search 1 level deep in main folder
        all_viz_files.extend(main_viz_files)

    # Prioritize HTML files over images for each visualization type
    viz_by_type = {}
    for viz in all_viz_files:
        # Create a key based on the visualization type (removing extension and
        # normalizing name)
        base_name = viz['name'].lower()
        for ext in ['.html', '.png', '.jpg', '.jpeg']:
            base_name = base_name.replace(ext, '')

        # Categorize by visualization type
        viz_type = None
        if 'equity' in base_name:
            viz_type = 'equity'
        elif 'drawdown' in base_name:
            viz_type = 'drawdown'
        elif 'return' in base_name:
            viz_type = 'returns'
        elif 'performance' in base_name or 'metric' in base_name:
            viz_type = 'performance'
        elif 'comparison' in base_name:
            viz_type = 'comparison'
        elif 'dashboard' in base_name:
            viz_type = 'dashboard'
        else:
            viz_type = 'other'

        # Add visualization to its type category
        if viz_type not in viz_by_type:
            viz_by_type[viz_type] = []
        viz_by_type[viz_type].append(viz)

    # For each type, prioritize HTML over images
    for viz_type, vizs in viz_by_type.items():
        # First try to find HTML versions
        html_vizs = [v for v in vizs if v['type'] == 'html']
        if html_vizs:
            results['charts'].extend(html_vizs)
        else:
            # If no HTML, include the image version
            image_vizs = [v for v in vizs if v['type'] == 'image']
            if image_vizs:
                # Just include one image per type
                results['charts'].extend(image_vizs[:1])

    # If no charts found, perform an even more thorough search for any HTML
    # files
    if not results['charts']:
        for root, dirs, files in os.walk(walk_forward_dir):
            for file in files:
                if file.endswith('.html'):
                    chart_path = os.path.join(root, file)
                    if not any(
                            chart['path'] == chart_path for chart in results['charts']):
                        results['charts'].append({
                            'path': chart_path,
                            'name': file,
                            'type': 'html'
                        })

    # Sort charts in a consistent way: dashboards first, then equity, then
    # others
    def chart_sort_key(chart):
        name = chart['name'].lower()
        if 'dashboard' in name:
            return (0, name)
        elif 'equity' in name:
            return (1, name)
        elif 'drawdown' in name:
            return (2, name)
        elif 'return' in name:
            return (3, name)
        elif 'performance' in name:
            return (4, name)
        elif 'comparison' in name:
            return (5, name)
        else:
            return (99, name)

    results['charts'].sort(key=chart_sort_key)

    if results['charts']:
        print(f"Total visualization files found: {len(results['charts'])}")
        for chart in results['charts']:
            print(f"  - {chart['name']} ({chart['type']})")
    else:
        print("No visualization files found")

    has_results = (
        results['summary'] is not None or
        results['performance_metrics'] is not None or
        results['window_results'] or
        results['charts'] or
        results['in_sample_data'] is not None or
        results['out_sample_data'] is not None or
        results['comparison_files']
    )

    return results if has_results else None


@app.route('/')
def index():
    """Landing page with form to execute backtests"""
    # Get list of registered strategies
    strategies = get_registered_strategies()

    # Define available workflow types
    workflow_types = [{"id": "simple",
                       "name": "Simple Backtest",
                       "description": "Basic backtest with fixed parameters"},
                      {"id": "optimization",
                       "name": "Parameter Optimization",
                       "description": "Find optimal strategy parameters"},
                      {"id": "monte_carlo",
                       "name": "Monte Carlo Analysis",
                       "description": "Test strategy robustness with multiple simulations"},
                      {"id": "walkforward",
                       "name": "Walk Forward Analysis",
                       "description": "Test strategy across multiple time windows"},
                      {"id": "complete",
                       "name": "Complete Workflow",
                       "description": "Run all analysis steps (optimization, walk forward, monte carlo)"}]

    # Define optimization metrics with descriptions
    optimization_metrics = [
        {"id": "sharpe_ratio", "name": "Sharpe Ratio", "description": "Return per unit of risk (higher is better)", "direction": "maximize"},
        {"id": "sortino_ratio", "name": "Sortino Ratio", "description": "Return per unit of downside risk only (higher is better)", "direction": "maximize"},
        {"id": "calmar_ratio", "name": "Calmar Ratio", "description": "Return relative to maximum drawdown (higher is better)", "direction": "maximize"},
        {"id": "total_return", "name": "Total Return", "description": "Overall percentage return (higher is better)", "direction": "maximize"},
        {"id": "profit_factor", "name": "Profit Factor", "description": "Gross profit divided by gross loss (higher is better)", "direction": "maximize"},
        {"id": "max_drawdown", "name": "Maximum Drawdown", "description": "Maximum peak-to-trough decline (lower is better)", "direction": "minimize"},
        {"id": "average_trade", "name": "Average Trade", "description": "Mean profit/loss per trade (higher is better)", "direction": "maximize"},
        {"id": "win_rate", "name": "Win Rate", "description": "Percentage of winning trades (higher is better)", "direction": "maximize"},
        {"id": "ulcer_index", "name": "Ulcer Index", "description": "Measure of drawdown severity (lower is better)", "direction": "minimize"},
        {"id": "cagr", "name": "CAGR", "description": "Compound Annual Growth Rate (higher is better)", "direction": "maximize"}
    ]

    # Define walk forward modes with descriptions
    walk_forward_modes = [{"id": "rolling",
                           "name": "Rolling Window",
                           "description": "Both in-sample and out-of-sample windows move forward through time"},
                          {"id": "anchored",
                           "name": "Anchored Window",
                           "description": "In-sample starts from a fixed date and grows, out-of-sample moves forward"}]

    # Define reoptimization settings for walk forward
    reoptimization_settings = [{"id": "always",
                                "name": "Always",
                                "description": "Reoptimize parameters for each window"},
                               {"id": "on_degradation",
                                "name": "On Degradation",
                                "description": "Only reoptimize when performance degrades below threshold"},
                               {"id": "never",
                                "name": "Never",
                                "description": "Use initial optimal parameters for all windows"}]

    return render_template(
        'index.html',
        strategies=strategies,
        workflow_types=workflow_types,
        optimization_metrics=optimization_metrics,
        walk_forward_modes=walk_forward_modes,
        reoptimization_settings=reoptimization_settings,
        default_start_date=DEFAULT_START_DATE,
        default_end_date=DEFAULT_END_DATE,
        default_num_simulations=DEFAULT_NUM_SIMULATIONS,
        default_walk_forward_windows=DEFAULT_WALK_FORWARD_WINDOWS,
        default_in_sample_pct=DEFAULT_IN_SAMPLE_PCT)


@app.route('/strategy/<strategy_name>')
def strategy_params(strategy_name):
    """Get parameters for a specific strategy"""
    strategies = get_registered_strategies()

    # Find the strategy in the list
    strategy_info = None
    for strategy in strategies:
        if strategy['name'] == strategy_name:
            strategy_info = strategy
            break

    if not strategy_info:
        return jsonify({"error": f"Strategy {strategy_name} not found"}), 404

    # Return default parameters for the strategy
    if strategy_name == "MACrossover":
        return jsonify(
            {
                "parameters": {
                    "fast_period": {
                        "type": "int",
                        "default": 10,
                        "min": 2,
                        "max": 50,
                        "description": "Fast MA period"},
                    "slow_period": {
                        "type": "int",
                        "default": 50,
                        "min": 5,
                        "max": 200,
                        "description": "Slow MA period"},
                    "position_size": {
                        "type": "int",
                        "default": 100,
                        "min": 1,
                        "max": 1000,
                        "description": "Position size"},
                    "entry_threshold": {
                        "type": "float",
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.1,
                        "description": "Entry threshold"},
                    "exit_threshold": {
                        "type": "float",
                        "default": 0.0,
                        "min": 0.0,
                                "max": 0.1,
                        "description": "Exit threshold"}}})
    elif strategy_name == "AuctionMarket":
        return jsonify(
            {
                "parameters": {
                    "value_area": {
                        "type": "float",
                        "default": 0.7,
                        "min": 0.5,
                        "max": 0.9,
                        "description": "Value area percentage"},
                    "position_size": {
                        "type": "int",
                        "default": 100,
                        "min": 1,
                        "max": 1000,
                        "description": "Position size"},
                    "use_vwap": {
                        "type": "bool",
                        "default": True,
                        "description": "Use VWAP"},
                    "use_volume_profile": {
                        "type": "bool",
                        "default": True,
                        "description": "Use volume profile"},
                    "risk_percent": {
                        "type": "float",
                        "default": 0.01,
                        "min": 0.001,
                                "max": 0.05,
                                "description": "Risk percentage"},
                    "use_atr_sizing": {
                        "type": "bool",
                        "default": True,
                        "description": "Use ATR for position sizing"},
                    "atr_period": {
                        "type": "int",
                        "default": 14,
                        "min": 5,
                        "max": 30,
                        "description": "ATR period"}}})
    elif strategy_name == "MultiPosition":
        return jsonify(
            {
                "parameters": {
                    "sma_period": {
                        "type": "int",
                        "default": 20,
                        "min": 5,
                        "max": 200,
                        "description": "SMA period"},
                    "position_size": {
                        "type": "int",
                        "default": 100,
                        "min": 10,
                        "max": 1000,
                        "description": "Position size"},
                    "max_positions": {
                        "type": "int",
                        "default": 5,
                        "min": 1,
                        "max": 10,
                        "description": "Maximum positions"}}})
    else:
        return jsonify(
            {
                "parameters": {
                    "position_size": {
                        "type": "int",
                        "default": 100,
                        "min": 1,
                        "max": 1000,
                        "description": "Position size"}}})


@app.route('/results')
def results():
    """Results page showing all backtest results"""
    folders = get_output_folders()
    return render_template('results.html', folders=folders)


@app.route('/results/<folder_name>')
def view_result(folder_name):
    """View a specific backtest result"""
    folder_path = os.path.join(project_root, 'output', folder_name)

    if not os.path.exists(folder_path):
        flash(f"Results folder not found: {folder_name}")
        return redirect(url_for('results'))

    try:
        workflow_type = get_workflow_type_from_folder(folder_name)
        strategy_name = folder_name.split('_')[0]

        print(
            f"Processing result view for {folder_name}, workflow: {workflow_type}")

        # Get various result components
        logs = read_log_file(folder_path)
        summaries = read_summary_files(folder_path)
        
        # Look for stage error report files in main folder and subdirectories
        stage_error_report = None
        
        # First check the main folder
        for file in os.listdir(folder_path):
            if "stage_error_report" in file.lower() and file.endswith('.txt'):
                try:
                    report_path = os.path.join(folder_path, file)
                    with open(report_path, 'r') as f:
                        stage_error_report = {
                            'name': file,
                            'content': f.read(),
                            'path': report_path
                        }
                    print(f"Found stage error report in main folder: {file}")
                    break
                except Exception as e:
                    print(f"Error reading stage error report: {str(e)}")
        
        # If not found in main folder, check subdirectories
        if not stage_error_report:
            for root, dirs, files in os.walk(folder_path):
                if root == folder_path:
                    continue  # Skip the main folder as we already checked it
                    
                for file in files:
                    if "stage_error_report" in file.lower() and file.endswith('.txt'):
                        try:
                            report_path = os.path.join(root, file)
                            with open(report_path, 'r') as f:
                                stage_error_report = {
                                    'name': file,
                                    'content': f.read(),
                                    'path': report_path
                                }
                            print(f"Found stage error report in subdirectory: {report_path}")
                            break
                        except Exception as e:
                            print(f"Error reading stage error report from subdirectory: {str(e)}")
                
                if stage_error_report:
                    break  # Stop searching if we found a report

        # Get equity curve data with additional logging
        print(f"Attempting to retrieve equity curve data for {folder_name}")
        equity_curve = get_equity_curve(folder_path)

        # Special handling for optimization results if no equity curve found
        if not equity_curve and workflow_type == 'optimization':
            print(
                "Attempting to find main equity curve file directly for optimization workflow")
            try:
                # For optimization workflows, try to find the equity curve in
                # the root folder first
                main_equity_path = os.path.join(
                    folder_path, 'equity_curve.csv')
                if os.path.exists(main_equity_path):
                    print(f"Found main equity curve at: {main_equity_path}")
                    df = pd.read_csv(main_equity_path)
                    if not df.empty:
                        # Convert dates
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                            dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
                        else:
                            dates = [str(i) for i in range(len(df))]

                        # Get equity values
                        if 'Value' in df.columns:
                            equity = df['Value'].tolist()
                        elif 'Equity' in df.columns:
                            equity = df['Equity'].tolist()
                        else:
                            numeric_cols = df.select_dtypes(
                                include=['number']).columns
                            if len(numeric_cols) > 0:
                                equity = df[numeric_cols[0]].tolist()
                            else:
                                raise ValueError(
                                    "No suitable numeric column found")

                        # Process moving averages
                        ma_data = {}
                        for period in [20, 50, 200]:
                            if len(equity) > period:
                                try:
                                    series = pd.Series(equity)
                                    ma_series = series.rolling(
                                        window=period).mean()
                                    # Use bfill (backward fill) which can work
                                    # better for the start of the series
                                    ma_values = ma_series.fillna(
                                        method='bfill').tolist()
                                    ma_data[f'MA{period}'] = ma_values
                                except Exception as e:
                                    print(f"Error calculating MA: {e}")

                        equity_curve = {
                            'dates': dates,
                            'equity': equity,
                            'moving_averages': ma_data,
                            'df': df
                        }
                        print(f"Successfully created equity curve data manually")
            except Exception as e:
                print(f"Failed to manually create equity curve: {e}")
                import traceback
                traceback.print_exc()

        if equity_curve:
            print(
                f"Successfully found equity curve data with {len(equity_curve['dates'])} data points")
        else:
            print(f"No equity curve data found for {folder_name}")

        # Find equity curve plots generated by workflows with --plot flag
        equity_plots = get_equity_curve_plots(
            folder_path) if 'get_equity_curve_plots' in globals() else []

        trade_log = get_trade_log(folder_path)
        optimization_results = get_optimization_results(folder_path)

        # STANDARDIZED TAB DISPLAY LOGIC:
        # Always look for walk_forward_results for all workflow types (not just
        # 'walk_forward' and 'complete')
        walk_forward_results = get_walk_forward_results(folder_path)

        # Always look for monte_carlo_charts for all workflow types (not just
        # 'monte_carlo' and 'complete')
        monte_carlo_charts = get_monte_carlo_charts(folder_path)

        # Enhanced debug logging for Monte Carlo charts
        if monte_carlo_charts:
            print(f"Monte Carlo charts details for {folder_name}:")
            print(f"Found {len(monte_carlo_charts)} charts")
            for i, chart in enumerate(monte_carlo_charts):
                print(f"Chart {i+1}: {chart['name']} - type: {chart['type']}")
        else:
            print(f"No Monte Carlo charts found for {folder_name}")

        # Get equity curve description based on workflow type
        equity_curve_description = get_equity_curve_description(workflow_type)

        # STANDARDIZED TAB VISIBILITY CONTROL:
        # Define which tabs should be shown for each workflow type
        tab_visibility = {
            'simple': {
                'summary': True,
                'equity_curve': True,
                'trades': True,
                'optimization': False,
                'monte_carlo': False,
                'walk_forward': False
            },
            'optimization': {
                'summary': True,
                'equity_curve': True,
                'trades': True,
                'optimization': True,
                'monte_carlo': False,
                'walk_forward': False
            },
            'monte_carlo': {
                'summary': True,
                'equity_curve': True,
                'trades': True,
                'optimization': False,
                'monte_carlo': True,
                'walk_forward': False
            },
            'walkforward': {
                'summary': True,
                'equity_curve': True,
                'trades': True,
                'optimization': False,
                'monte_carlo': False,
                'walk_forward': True
            },
            'complete': {
                'summary': True,
                'equity_curve': True,
                'trades': True,
                'optimization': True,
                'monte_carlo': True,
                'walk_forward': True
            }
        }

        # Get tab settings for current workflow (default to all tabs if
        # workflow_type not found)
        current_tabs = tab_visibility.get(workflow_type, {
            'summary': True,
            'equity_curve': True,
            'trades': True,
            'optimization': True,
            'monte_carlo': True,
            'walk_forward': True
        })

        return render_template(
            'result_detail.html',
            folder_name=folder_name,
            folder_path=folder_path,
            workflow_type=workflow_type,
            strategy_name=strategy_name,
            logs=logs,
            summaries=summaries,
            equity_curve=equity_curve,
            equity_plots=equity_plots if 'equity_plots' in locals() else [],
            trade_log=trade_log,
            optimization_results=optimization_results if current_tabs['optimization'] else None,
            monte_carlo_charts=monte_carlo_charts if current_tabs['monte_carlo'] else None,
            walk_forward_results=walk_forward_results if current_tabs['walk_forward'] else None,
            project_root=project_root,
            equity_curve_description=equity_curve_description,
            stage_error_report=stage_error_report,
            tab_visibility=current_tabs)
    except Exception as e:
        print(f"Error viewing result {folder_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f"Error viewing result: {str(e)}")
        return redirect(url_for('results'))


@app.route('/log/<folder_name>')
def view_log(folder_name):
    """View log from a backtest"""
    folder_path = os.path.join(project_root, 'output', folder_name)
    logs = read_log_file(folder_path)

    return render_template('log_view.html', folder_name=folder_name, logs=logs)


@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """Run a backtest with the specified parameters"""
    # Extract form data
    strategy = request.form.get('strategy')
    workflow_type = request.form.get('workflow_type', 'simple')
    tickers = request.form.get('tickers', 'SPY').replace(' ', '').split(',')
    start_date = request.form.get('start_date', DEFAULT_START_DATE)
    end_date = request.form.get('end_date', DEFAULT_END_DATE)
    initial_capital = float(request.form.get('initial_capital', 100000))

    # Create a timestamp for the output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create config file for the backtest
    config = {
        "workflow_type": workflow_type,
        "common_params": {
            "start_date": start_date,
            "end_date": end_date,
            "data_dir": os.path.join(
                project_root,
                "input"),
            "tickers": tickers,
            "initial_capital": initial_capital,
            "commission": 0.001,
            "plot": False,
            "enhanced_plots": True if workflow_type in [
                "monte_carlo",
                "walk_forward",
                "complete"] else False,
            "verbose": request.form.get('verbose') == 'on'},
        "strategies": {
            strategy: {}}}

    # Handle strategy parameters
    parameters = {}
    parameter_grid = {}

    # Define parameter mappings for each strategy to ensure names match what
    # the strategy expects
    parameter_mappings = {
        "MultiPosition": {
            # No mappings needed now that we've fixed the frontend parameter
            # names
        },
        "MACrossover": {
            # No mappings needed
        },
        "AuctionMarket": {
            # No mappings needed
        }
    }

    # Extract single parameters
    for key, value in request.form.items():
        if key.startswith('param_'):
            param_name = key[6:]  # Remove 'param_' prefix

            # Convert value to appropriate type
            if value.lower() in ['true', 'on', 'yes']:
                parameters[param_name] = True
            elif value.lower() in ['false', 'off', 'no']:
                parameters[param_name] = False
            elif value.isdigit():
                parameters[param_name] = int(value)
            else:
                try:
                    parameters[param_name] = float(value)
                except ValueError:
                    parameters[param_name] = value

    # Apply parameter mappings if any
    if strategy in parameter_mappings:
        for old_name, new_name in parameter_mappings[strategy].items():
            if old_name in parameters:
                parameters[new_name] = parameters.pop(old_name)
                print(f"Mapping parameter: {old_name} -> {new_name}")

    # Extract grid parameters for optimization
    for key, value in request.form.items():
        if key.startswith('grid_'):
            param_name = key[5:]  # Remove 'grid_' prefix

            # Apply parameter mappings for grid parameters if any
            if strategy in parameter_mappings:
                if param_name in parameter_mappings[strategy]:
                    param_name = parameter_mappings[strategy][param_name]
                    print(f"Mapping grid parameter: {key[5:]} -> {param_name}")

            # Split the values by comma and convert
            try:
                values = [v.strip() for v in value.split(',') if v.strip()]
                typed_values = []

                for v in values:
                    if v.lower() in ['true', 'yes']:
                        typed_values.append(True)
                    elif v.lower() in ['false', 'no']:
                        typed_values.append(False)
                    elif v.isdigit():
                        typed_values.append(int(v))
                    else:
                        try:
                            typed_values.append(float(v))
                        except ValueError:
                            typed_values.append(v)

                if typed_values:
                    parameter_grid[param_name] = typed_values
            except Exception as e:
                flash(f"Error parsing grid parameter {param_name}: {str(e)}")
                return redirect(url_for('index'))

    # Add parameters to config
    if workflow_type in ['optimization', 'walk_forward', 'complete']:
        if parameter_grid:
            config["strategies"][strategy]["parameter_grid"] = parameter_grid
            print(f"Using parameter grid: {parameter_grid}")
        else:
            # If no grid specified but optimization workflow, convert single
            # params to grid
            grid = {}
            for key, value in parameters.items():
                if isinstance(
                        value, (int, float)) and not isinstance(
                        value, bool):
                    # Create a range around the value
                    if isinstance(value, int):
                        grid[key] = [max(1, value - 5), value, value + 5]
                    else:  # float
                        grid[key] = [max(0.001, value / 2), value, value * 2]
                elif isinstance(value, bool):
                    grid[key] = [not value, value]
                else:
                    grid[key] = [value]

            config["strategies"][strategy]["parameter_grid"] = grid
            print(f"Created parameter grid: {grid}")
            flash("Created parameter grid for optimization based on single parameters.")
    else:
        # For non-optimization workflows, use single parameters
        config["strategies"][strategy]["parameters"] = parameters
        print(f"Using parameters: {parameters}")

    # Add specific parameters for different workflow types
    if workflow_type in ['optimization', 'walk_forward', 'complete']:
        # Get the optimization metric from the form
        optimization_metric = request.form.get(
            'optimization_metric', 'sharpe_ratio')

        # Map the metric name from UI to the format expected by the backtester
        metric_mapping = {
            'sharpe_ratio': 'sharpe_ratio',
            'sortino_ratio': 'sortino_ratio',
            'calmar_ratio': 'calmar_ratio',
            'total_return': 'total_return',
            'profit_factor': 'profit_factor',
            'max_drawdown': 'max_drawdown',  # Note: Minimizing drawdown
            'average_trade': 'average_trade',
            'win_rate': 'win_rate',
            'ulcer_index': 'ulcer_index',  # Note: Lower is better
            'cagr': 'cagr'  # Compound Annual Growth Rate
        }

        # Use the mapped metric or default to sharpe_ratio if unknown
        mapped_metric = metric_mapping.get(optimization_metric, 'sharpe_ratio')

        print(
            f"Using optimization metric: {mapped_metric} (from UI: {optimization_metric})")

        config["strategies"][strategy]["optimization"] = {
            "n_trials": int(request.form.get('n_trials', 50)),
            "optimization_metric": mapped_metric
        }

    if workflow_type in ['monte_carlo', 'complete']:
        config["strategies"][strategy]["monte_carlo"] = {
            "n_simulations": int(
                request.form.get(
                    'n_simulations',
                    DEFAULT_NUM_SIMULATIONS)),
            "keep_permuted_data": request.form.get('keep_permuted_data') == 'on'}

    if workflow_type in ['walk_forward', 'complete']:
        # Add walk forward specific parameters
        anchor_type = request.form.get('anchor', 'rolling')
        # Can be 'always', 'on_degradation', 'never'
        reoptimize = request.form.get('reoptimize', 'always')

        wf_params = {
            "n_windows": int(request.form.get('n_windows', DEFAULT_WALK_FORWARD_WINDOWS)),
            "in_sample_pct": float(request.form.get('in_sample_pct', DEFAULT_IN_SAMPLE_PCT)),
            "anchor": anchor_type == 'rolling',  # True for rolling, False for anchored
            "optimization_metric": mapped_metric if workflow_type in ['optimization', 'walk_forward', 'complete'] else 'sharpe_ratio',
            "reoptimize": reoptimize,
            "combine_results": request.form.get('combine_results') == 'on',
            "plot_comparisons": request.form.get('plot_comparisons') == 'on',
            "export_model_history": request.form.get('export_model_history') == 'on'
        }

        # Add advanced walk forward parameters if provided
        if request.form.get('min_window_size'):
            wf_params["min_window_size"] = int(
                request.form.get('min_window_size'))

        if request.form.get('validation_ratio'):
            wf_params["validation_ratio"] = float(
                request.form.get('validation_ratio'))

        # Get reoptimization threshold if applicable
        if reoptimize == 'on_degradation' and request.form.get(
                'reoptimization_threshold'):
            wf_params["reoptimization_threshold"] = float(
                request.form.get('reoptimization_threshold'))

        # Add walk forward params to config
        config["strategies"][strategy]["walk_forward"] = wf_params

        # Add walk forward specific parameters for display in frontend
        if workflow_type == 'walk_forward':
            # Ensure we have optimization parameters for walk forward
            if "optimization" not in config["strategies"][strategy]:
                config["strategies"][strategy]["optimization"] = {
                    "n_trials": int(request.form.get('n_trials', 50)),
                    "optimization_metric": mapped_metric
                }

    # Create config directory if it doesn't exist
    config_dir = os.path.join(project_root, 'input', 'workflow_configs')
    os.makedirs(config_dir, exist_ok=True)

    # Write config to file
    config_file = os.path.join(config_dir,
                               f"{strategy}_{workflow_type}_{timestamp}.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    # Run the backtest using subprocess
    cmd = [
        sys.executable,
        os.path.join(project_root, 'src', 'workflows', 'cli.py'),
        '--config', config_file
    ]

    try:
        # Create a subprocess to run the backtest
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )

        # Flash message with command details
        flash(
            f"Started backtest: {strategy}_{workflow_type} with config {os.path.basename(config_file)}")

        # Capture and print output (this is non-blocking)
        def read_output():
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line.strip())

        import threading
        output_thread = threading.Thread(target=read_output)
        output_thread.daemon = True
        output_thread.start()

        # Create a thread to wait for process completion and cleanup config
        # file
        def cleanup_process():
            # Wait for process to complete
            return_code = process.wait()
            print(
                f"Backtest process completed with return code: {return_code}")

            # Delete config file after backtest completes
            try:
                if os.path.exists(config_file):
                    os.remove(config_file)
                    print(f"Cleaned up config file: {config_file}")
            except Exception as e:
                print(f"Error cleaning up config file: {str(e)}")

        cleanup_thread = threading.Thread(target=cleanup_process)
        cleanup_thread.daemon = True
        cleanup_thread.start()

        # Return success and redirect to results page
        return redirect(url_for('results'))

    except Exception as e:
        # Clean up config file after error
        try:
            if os.path.exists(config_file):
                os.remove(config_file)
                print(f"Cleaned up config file after error: {config_file}")
        except Exception as cleanup_error:
            print(f"Error cleaning up config file after error: {str(cleanup_error)}")

        flash(f"Error running backtest: {str(e)}")
        return redirect(url_for('index'))


@app.route('/file/<path:file_path>')
def serve_file(file_path):
    """Serve any static file (images, HTML, etc.)"""
    full_path = os.path.join(project_root, file_path)
    if os.path.exists(full_path):
        return send_file(full_path)
    else:
        return f"File not found: {file_path}", 404


@app.route('/cleanup_configs')
def cleanup_configs():
    """Admin utility to clean up old config files"""
    if not request.remote_addr == '127.0.0.1':
        return "Access denied", 403

    config_dir = os.path.join(project_root, 'input', 'workflow_configs')
    if not os.path.exists(config_dir):
        return "Config directory not found", 404

    deleted_count = 0
    error_count = 0
    config_files = []

    # List all config files
    for file in os.listdir(config_dir):
        if file.endswith('.json'):
            file_path = os.path.join(config_dir, file)
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))

            # Keep track of files
            config_files.append({
                'name': file,
                'path': file_path,
                'age_hours': file_age.total_seconds() / 3600
            })

    # Delete old files
    for file_info in config_files:
        if file_info['age_hours'] > 24:                
            try:
                os.remove(file_info['path'])
                deleted_count += 1
            except Exception:
                error_count += 1

    return jsonify({
        'status': 'success',
        'message': f'Deleted {deleted_count} old config files, {error_count} errors.',
        'remaining_files': len(config_files) - deleted_count
    })


if __name__ == '__main__':
    # Create temp directory for temporary files
    os.makedirs(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            'temp'),
        exist_ok=True)

    # Verify output directory exists
    output_dir = os.path.join(project_root, 'output')
    if not os.path.exists(output_dir):
        print(f"WARNING: Output directory not found at {output_dir}")
        print("Creating output directory...")
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory at {output_dir}")
        except Exception as e:
            print(f"Error creating output directory: {str(e)}")
    else:
        print(f"Output directory found at {output_dir}")
        # List all result folders
        folders = [f for f in os.listdir(output_dir) if os.path.isdir(
            os.path.join(output_dir, f)) and not f.startswith('.')]
        print(f"Found {len(folders)} result folders in output directory")

    # Clean up old config files on startup
    def cleanup_old_configs():
        """Clean up old config files on startup"""
        try:
            config_dir = os.path.join(
                project_root, 'input', 'workflow_configs')
            if not os.path.exists(config_dir):
                return

            deleted_count = 0
            for file in os.listdir(config_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(config_dir, file)                    
                    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    # Delete files older than 24 hours
                    if file_age.total_seconds() > 86400:  # 24 hours
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                        except Exception as e:
                            print(f"Error deleting old config file {file_path}: {str(e)}")
            
            if deleted_count > 0:
                print(f"Startup cleanup: Removed {deleted_count} old config files")
        except Exception as e:
            print(f"Error during startup config cleanup: {str(e)}")

    # Run cleanup
    cleanup_old_configs()

    app.run(debug=True, host='0.0.0.0', port=5000)
