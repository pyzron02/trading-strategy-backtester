#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line tool to generate consolidated error reports across workflows.
This tool provides a single utility to track and report all errors in the system.
"""
import os
import sys
import argparse
import datetime

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
project_root = os.path.dirname(src_dir)  # Go up to project root
if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.error_reporting import generate_consolidated_error_report, generate_error_summary

def main():
    """Command-line interface for generating error reports."""
    
    parser = argparse.ArgumentParser(
        description="Generate error reports for trading strategy backtester workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate a consolidated error report for the past 7 days:
    python generate_error_report.py --consolidated
    
  Generate a consolidated error report for the past 14 days:
    python generate_error_report.py --consolidated --days 14
    
  Generate an error summary for a specific workflow output directory:
    python generate_error_report.py --workflow-dir /path/to/output/AuctionMarket_monte_carlo_20250426
    
  Generate both a consolidated report and detailed report for a workflow:
    python generate_error_report.py --consolidated --workflow-dir /path/to/output/AuctionMarket_monte_carlo_20250426
        """
    )
    
    parser.add_argument(
        "--consolidated", 
        action="store_true",
        help="Generate a consolidated error report across all workflows"
    )
    
    parser.add_argument(
        "--days", 
        type=int, 
        default=7,
        help="Number of days to include in the consolidated report (default: 7)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="Directory to save the generated report(s) (default: logs directory)"
    )
    
    parser.add_argument(
        "--workflow-dir", 
        type=str,
        help="Specific workflow directory to analyze"
    )
    
    parser.add_argument(
        "--format", 
        type=str,
        choices=["txt", "json", "html"],
        default="txt",
        help="Output format for the report (default: txt)"
    )
    
    args = parser.parse_args()
    
    # Set current working directory as default output directory if not specified
    if not args.output_dir:
        args.output_dir = os.getcwd()
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If no specific action is requested, show help
    if not args.consolidated and not args.workflow_dir:
        parser.print_help()
        return
    
    # Generate a consolidated report if requested
    if args.consolidated:
        report_path = generate_consolidated_error_report(args.output_dir, args.days)
        print(f"\n\033[92mConsolidated error report generated: \033[1m{report_path}\033[0m")
        print(f"Covering errors from the past {args.days} days")
    
    # Generate a specific workflow report if requested
    if args.workflow_dir:
        if not os.path.exists(args.workflow_dir):
            print(f"\n\033[91mError: Workflow directory not found: {args.workflow_dir}\033[0m")
            return
        
        # Try to determine workflow type and strategy
        dir_name = os.path.basename(args.workflow_dir)
        parts = dir_name.split('_')
        
        strategy_name = parts[0] if len(parts) > 0 else "Unknown"
        
        workflow_type = "unknown"
        for wf in ['simple', 'monte_carlo', 'optimization', 'walkforward', 'complete']:
            if wf in dir_name.lower():
                workflow_type = wf
                break
        
        # Generate summary for the specific directory
        summary_file = generate_error_summary(
            args.workflow_dir, 
            workflow_type,
            strategy_name,
            "Unknown", "Unknown",
            ["Unknown"],
            "Manual error report generation"
        )
        
        print(f"\n\033[92mWorkflow error summary generated: \033[1m{summary_file}\033[0m")
        print(f"For workflow: {workflow_type} - Strategy: {strategy_name}")

if __name__ == "__main__":
    main()