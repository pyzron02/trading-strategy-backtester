#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to check log files in the output directory for errors.
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

from workflows.workflow_utils import check_logs_for_errors, print_error_report

def main():
    parser = argparse.ArgumentParser(description="Check log files for errors")
    parser.add_argument("--output-dir", type=str, help="Directory to search for logs")
    parser.add_argument("--keywords", type=str, nargs="+", help="Keywords to search for")
    parser.add_argument("--report-file", type=str, help="File to save the report to")
    parser.add_argument("--days", type=int, default=None, help="Only check logs from the last N days")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(project_root, "output")
    
    # Filter output directories by date if days parameter is provided
    if args.days:
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=args.days)
        
        # If output_dir is the main output directory, filter subdirectories by date
        if output_dir == os.path.join(project_root, "output"):
            filtered_dirs = []
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    try:
                        # Check if folder was created within the specified time range
                        mod_time = os.path.getmtime(item_path)
                        folder_date = datetime.datetime.fromtimestamp(mod_time)
                        if folder_date >= cutoff_date:
                            filtered_dirs.append(item_path)
                    except Exception:
                        # If we can't determine the date, include it anyway
                        filtered_dirs.append(item_path)
            
            # If we have filtered directories, use them
            if filtered_dirs:
                print(f"Checking {len(filtered_dirs)} directories from the last {args.days} days")
                error_logs = {}
                for directory in filtered_dirs:
                    error_logs.update(check_logs_for_errors(directory, args.keywords))
                print_error_report(error_logs, args.report_file)
                return
    
    # Default behavior: check all logs in the output directory
    error_logs = check_logs_for_errors(output_dir, args.keywords)
    print_error_report(error_logs, args.report_file)

if __name__ == "__main__":
    main() 