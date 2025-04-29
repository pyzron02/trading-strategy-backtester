import os
import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_manager import path_manager

def create_directories(base_dir=None, strategy_name='default_strategy'):
    """
    Create directories for input files, source code, and timestamped output results.
    
    Parameters:
    - base_dir: str, path to the project base directory (default: None, will use path_manager's default)
    - strategy_name: str, name of the strategy for organizing output (default: 'default_strategy')
    """
    # Initialize path_manager with base_dir if provided
    if base_dir:
        global path_manager
        path_manager = path_manager.__class__(base_dir)
    
    # Ensure all standard directories exist
    try:
        dirs = path_manager.ensure_base_dirs()
        for name, path in dirs.items():
            print(f"Directory exists: {path}")
    except Exception as e:
        print(f"Error creating directories: {e}")
    
    # Create a timestamped strategy output directory if requested
    if strategy_name:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        strategy_output_dir = path_manager.output_dir / f"{timestamp}_{strategy_name}"
        
        try:
            path_manager.ensure_dir(strategy_output_dir)
            print(f"Created strategy output directory: {strategy_output_dir}")
            return strategy_output_dir
        except Exception as e:
            print(f"Error creating {strategy_output_dir}: {e}")
    
    return path_manager.output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up project directories for backtesting")
    parser.add_argument('--base_dir', type=str, default=None, help="Path to the project base directory")
    parser.add_argument('--strategy_name', type=str, default='CoveredCall', help="Name of the strategy")
    args = parser.parse_args()
    
    create_directories(
        base_dir=args.base_dir,
        strategy_name=args.strategy_name
    )