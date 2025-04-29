#!/usr/bin/env python3
# path_manager.py - Centralized path management for the application

import os
from pathlib import Path

class PathManager:
    """
    Centralized path management for the application.
    
    This class provides a way to manage paths in a containerization-friendly way,
    allowing the application to run in different environments without hardcoding
    absolute paths.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern to ensure only one path manager exists."""
        if cls._instance is None:
            cls._instance = super(PathManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, base_dir=None):
        """
        Initialize the PathManager.
        
        Args:
            base_dir (str, optional): Base directory for the application. 
                If not provided, uses the following in order:
                1. BASE_DIR environment variable if set
                2. Parent directory of the src folder
        """
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
        
        # Set base directory
        if base_dir is None:
            # First check for environment variable
            env_base_dir = os.environ.get('BASE_DIR')
            if env_base_dir:
                self.base_dir = Path(env_base_dir)
            else:
                # Default: Use the directory containing the 'src' folder
                current_file = Path(__file__)
                # Navigate up from utils to src to project root
                self.base_dir = current_file.parent.parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # Define standard directories relative to base_dir
        self.src_dir = self.base_dir / 'src'
        self.input_dir = self.base_dir / 'input'
        self.output_dir = self.base_dir / 'output'
        self.logs_dir = self.base_dir / 'logs'
        self.cache_dir = self.base_dir / 'cache'
        
        # Define subdirectories for input
        self.parameters_dir = self.input_dir / 'parameters'
        self.parameter_grids_dir = self.input_dir / 'parameter_grids'
        self.workflow_configs_dir = self.input_dir / 'workflow_configs'
        
        self._initialized = True
    
    def get_path(self, name):
        """
        Get a path by name.
        
        Args:
            name (str): Path name (base_dir, src_dir, input_dir, output_dir, logs_dir, cache_dir,
                        parameters_dir, parameter_grids_dir, workflow_configs_dir)
            
        Returns:
            Path: Path object for the requested directory
        """
        if hasattr(self, name):
            return getattr(self, name)
        raise ValueError(f"Unknown path name: {name}")
    
    def join_path(self, base, *paths):
        """
        Join paths relative to a base path.
        
        Args:
            base (str): Base path name (e.g., 'input_dir', 'output_dir')
            *paths: Additional path components to join
            
        Returns:
            Path: Combined path
        """
        base_path = self.get_path(base)
        return base_path.joinpath(*paths)
    
    def ensure_dir(self, dir_path):
        """
        Ensure a directory exists.
        
        Args:
            dir_path (Path or str): Directory path to ensure
            
        Returns:
            Path: Path to the ensured directory
        """
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def ensure_base_dirs(self):
        """
        Ensure all base directories exist.
        
        Returns:
            dict: Dictionary of paths that were created
        """
        dirs = {
            'input_dir': self.input_dir,
            'output_dir': self.output_dir,
            'logs_dir': self.logs_dir,
            'cache_dir': self.cache_dir,
            'parameters_dir': self.parameters_dir,
            'parameter_grids_dir': self.parameter_grids_dir,
            'workflow_configs_dir': self.workflow_configs_dir
        }
        
        for name, path in dirs.items():
            self.ensure_dir(path)
        
        return dirs
    
    def rel_path(self, path):
        """
        Get a path relative to the base directory.
        
        Args:
            path (Path or str): Path to convert
            
        Returns:
            Path: Relative path
        """
        return Path(path).relative_to(self.base_dir)
    
    def abs_path(self, rel_path):
        """
        Convert a relative path to an absolute path.
        
        Args:
            rel_path (Path or str): Relative path
            
        Returns:
            Path: Absolute path
        """
        return self.base_dir / rel_path
    
    def __str__(self):
        """String representation showing the base directory."""
        return f"PathManager(base_dir={self.base_dir})"

# Create a singleton instance
path_manager = PathManager()

# Helper functions for common path operations
def get_base_dir():
    """Get the base directory."""
    return path_manager.base_dir

def get_src_dir():
    """Get the src directory."""
    return path_manager.src_dir

def get_input_dir():
    """Get the input directory."""
    return path_manager.input_dir

def get_output_dir():
    """Get the output directory."""
    return path_manager.output_dir

def get_logs_dir():
    """Get the logs directory."""
    return path_manager.logs_dir

def get_cache_dir():
    """Get the cache directory."""
    return path_manager.cache_dir

def ensure_directory(path):
    """Ensure a directory exists."""
    return path_manager.ensure_dir(path)

def join_path(base, *paths):
    """Join paths relative to a base path."""
    return path_manager.join_path(base, *paths)