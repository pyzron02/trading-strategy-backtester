#!/usr/bin/env python3
# ensure_directories.py - Ensure required directories exist

import os
import sys
from pathlib import Path
from .path_manager import path_manager

def ensure_directories():
    """Ensure required directories exist."""
    # Ensure all base directories exist
    dirs = path_manager.ensure_base_dirs()
    
    # Print created directories
    for name, path in dirs.items():
        print(f"Created directory: {path}")
    
    # Create strategies directory
    strategies_dir = path_manager.src_dir / 'strategies'
    path_manager.ensure_dir(strategies_dir)
    print(f"Created strategies directory: {strategies_dir}")
    
    return 0

def ensure_output_directory(output_dir):
    """
    Ensure the specified output directory exists.
    
    Args:
        output_dir: Path to the output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")

if __name__ == '__main__':
    sys.exit(ensure_directories()) 