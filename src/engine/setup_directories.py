import os
import argparse
from datetime import datetime

def create_directories(project_root=None, strategy_name='default_strategy'):
    """
    Create directories for input files, source code, and timestamped output results.
    
    Parameters:
    - project_root: str, path to the project root directory (default: None, will be determined automatically)
    - strategy_name: str, name of the strategy for organizing output (default: 'default_strategy')
    """
    # Determine project root if not provided
    if project_root is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Define standard directories
    input_dir = os.path.join(project_root, 'input')
    output_dir = os.path.join(project_root, 'output')
    logs_dir = os.path.join(project_root, 'logs')
    cache_dir = os.path.join(project_root, 'cache')
    
    # Create the input directory
    if not os.path.exists(input_dir):
        try:
            os.makedirs(input_dir)
            print(f"Created directory: {input_dir}")
        except Exception as e:
            print(f"Error creating {input_dir}: {e}")
    else:
        print(f"Directory already exists: {input_dir}")
    
    # Create the output directory
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        except Exception as e:
            print(f"Error creating {output_dir}: {e}")
    else:
        print(f"Directory already exists: {output_dir}")
    
    # Create the logs directory
    if not os.path.exists(logs_dir):
        try:
            os.makedirs(logs_dir)
            print(f"Created directory: {logs_dir}")
        except Exception as e:
            print(f"Error creating {logs_dir}: {e}")
    else:
        print(f"Directory already exists: {logs_dir}")
    
    # Create the cache directory
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
            print(f"Created directory: {cache_dir}")
        except Exception as e:
            print(f"Error creating {cache_dir}: {e}")
    else:
        print(f"Directory already exists: {cache_dir}")
    
    # Create a timestamped strategy output directory if requested
    if strategy_name:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        strategy_output_dir = os.path.join(output_dir, f"{timestamp}_{strategy_name}")
        
        if not os.path.exists(strategy_output_dir):
            try:
                os.makedirs(strategy_output_dir)
                print(f"Created strategy output directory: {strategy_output_dir}")
                return strategy_output_dir
            except Exception as e:
                print(f"Error creating {strategy_output_dir}: {e}")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up project directories for backtesting")
    parser.add_argument('--project_root', type=str, default=None, help="Path to the project root directory")
    parser.add_argument('--strategy_name', type=str, default='CoveredCall', help="Name of the strategy")
    args = parser.parse_args()
    
    create_directories(
        project_root=args.project_root,
        strategy_name=args.strategy_name
    )