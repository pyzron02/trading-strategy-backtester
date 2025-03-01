import os
import argparse
from datetime import datetime

def create_directories(input_dir='input', src_dir='src', output_dir='output', strategy_name='default_strategy'):
    """
    Create directories for input files, source code, and timestamped output results.
    
    Parameters:
    - input_dir: str, name of the input directory (default: 'input')
    - src_dir: str, name of the source code directory (default: 'src')
    - output_dir: str, name of the output directory (default: 'output')
    - strategy_name: str, name of the strategy for organizing output (default: 'default_strategy')
    """
    # Create the input directory
    if not os.path.exists(input_dir):
        try:
            os.makedirs(input_dir)
            print(f"Created directory: {input_dir}")
        except Exception as e:
            print(f"Error creating {input_dir}: {e}")
    else:
        print(f"Directory already exists: {input_dir}")
    
    # Create the source directory
    if not os.path.exists(src_dir):
        try:
            os.makedirs(src_dir)
            print(f"Created directory: {src_dir}")
        except Exception as e:
            print(f"Error creating {src_dir}: {e}")
    else:
        print(f"Directory already exists: {src_dir}")
    
    # Create the output directory with timestamped subfolder
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_subfolder = os.path.join(output_dir, f"{timestamp}_{strategy_name}")
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        except Exception as e:
            print(f"Error creating {output_dir}: {e}")
    else:
        print(f"Directory already exists: {output_dir}")
    
    if not os.path.exists(output_subfolder):
        try:
            os.makedirs(output_subfolder)
            print(f"Created subdirectory: {output_subfolder}")
        except Exception as e:
            print(f"Error creating {output_subfolder}: {e}")
    else:
        print(f"Subdirectory already exists: {output_subfolder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up project directories for backtesting")
    parser.add_argument('--input_dir', type=str, default='input', help="Name of the input directory")
    parser.add_argument('--src_dir', type=str, default='src', help="Name of the source code directory")
    parser.add_argument('--output_dir', type=str, default='output', help="Name of the output directory")
    parser.add_argument('--strategy_name', type=str, default='CoveredCall', help="Name of the strategy")
    args = parser.parse_args()
    
    create_directories(
        input_dir=args.input_dir,
        src_dir=args.src_dir,
        output_dir=args.output_dir,
        strategy_name=args.strategy_name
    )