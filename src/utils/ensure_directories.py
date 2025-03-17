#!/usr/bin/env python3
# ensure_directories.py - Ensure required directories exist

import os
import sys

def ensure_directories():
    """Ensure required directories exist."""
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels (src -> project root)
    
    # Create output directory
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Create strategies directory
    strategies_dir = os.path.join(os.path.dirname(current_dir), 'strategies')
    os.makedirs(strategies_dir, exist_ok=True)
    print(f"Created strategies directory: {strategies_dir}")
    
    # Create input directory
    input_dir = os.path.join(project_root, 'input')
    os.makedirs(input_dir, exist_ok=True)
    print(f"Created input directory: {input_dir}")
    
    # Create logs directory
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Created logs directory: {logs_dir}")
    
    # Create cache directory
    cache_dir = os.path.join(project_root, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Created cache directory: {cache_dir}")
    
    return 0

if __name__ == '__main__':
    sys.exit(ensure_directories()) 