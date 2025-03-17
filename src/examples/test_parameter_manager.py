#!/usr/bin/env python3
# test_parameter_manager.py - Test script for the ParameterManager class

import os
import sys
import time
import json

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from engine.parameter_management import ParameterManager
from strategies import registry

def main():
    """Test the ParameterManager class."""
    print("\n" + "="*80)
    print("ParameterManager Test")
    print("="*80 + "\n")
    
    # Create ParameterManager instance
    param_manager = ParameterManager()
    
    # Define parameters from registry
    param_manager.define_parameters_from_registry(registry)
    
    # Print defined strategies
    print(f"Defined parameters for strategies:")
    for strategy_name in param_manager.parameter_definitions:
        print(f"  - {strategy_name}:")
        for param_name, param_def in param_manager.parameter_definitions[strategy_name].items():
            print(f"    - {param_name}: {param_def['default']} (type: {param_def['type'].__name__})")
    
    # Test parameter validation
    print("\nTesting parameter validation:")
    
    # Get a strategy name
    if param_manager.parameter_definitions:
        strategy_name = list(param_manager.parameter_definitions.keys())[0]
        
        # Get default parameters
        default_params = param_manager.get_default_parameters(strategy_name)
        print(f"Default parameters for {strategy_name}: {default_params}")
        
        # Validate default parameters
        is_valid, error = param_manager.validate_parameters(strategy_name, default_params)
        print(f"Default parameters valid: {is_valid}")
        if not is_valid:
            print(f"Error: {error}")
        
        # Create an invalid parameter set
        invalid_params = default_params.copy()
        param_name = list(invalid_params.keys())[0]
        if isinstance(invalid_params[param_name], int):
            invalid_params[param_name] = "invalid"  # Change type to make it invalid
        elif isinstance(invalid_params[param_name], str):
            invalid_params[param_name] = 123  # Change type to make it invalid
        
        # Validate invalid parameters
        is_valid, error = param_manager.validate_parameters(strategy_name, invalid_params)
        print(f"Invalid parameters valid: {is_valid}")
        if not is_valid:
            print(f"Error: {error}")
        
        # Test parameter grid creation
        print("\nTesting parameter grid creation:")
        
        # Create a parameter grid
        param_grid = {}
        for param_name, param_def in param_manager.parameter_definitions[strategy_name].items():
            if param_def['type'] == int:
                param_grid[param_name] = [10, 20, 30]
            elif param_def['type'] == float:
                param_grid[param_name] = [0.1, 0.2, 0.3]
            elif param_def['type'] == bool:
                param_grid[param_name] = [True, False]
            elif param_def['type'] == str:
                param_grid[param_name] = ["option1", "option2"]
        
        # Limit to 2 parameters for simplicity
        param_grid = {k: param_grid[k] for k in list(param_grid.keys())[:2]}
        
        print(f"Parameter grid: {param_grid}")
        
        # Create parameter combinations
        start_time = time.time()
        param_combinations = param_manager.create_parameter_grid(strategy_name, param_grid)
        grid_time = time.time() - start_time
        
        print(f"Created {len(param_combinations)} parameter combinations in {grid_time:.2f} seconds")
        print(f"First few combinations:")
        for i, params in enumerate(param_combinations[:3]):
            print(f"  {i+1}: {params}")
        
        # Test parameter sampling
        print("\nTesting parameter sampling:")
        
        # Create parameter ranges
        param_ranges = {}
        for param_name, param_def in param_manager.parameter_definitions[strategy_name].items():
            if param_def['type'] == int:
                param_ranges[param_name] = (10, 100)
            elif param_def['type'] == float:
                param_ranges[param_name] = (0.1, 1.0)
            elif param_def['type'] == bool:
                param_ranges[param_name] = [True, False]
            elif param_def['type'] == str:
                param_ranges[param_name] = ["option1", "option2", "option3"]
        
        # Limit to 2 parameters for simplicity
        param_ranges = {k: param_ranges[k] for k in list(param_ranges.keys())[:2]}
        
        print(f"Parameter ranges: {param_ranges}")
        
        # Test random sampling
        start_time = time.time()
        random_samples = param_manager.sample_parameters(
            strategy_name, param_ranges, num_samples=5, method='random')
        random_time = time.time() - start_time
        
        print(f"Generated {len(random_samples)} random samples in {random_time:.2f} seconds")
        print(f"Random samples:")
        for i, params in enumerate(random_samples):
            print(f"  {i+1}: {params}")
        
        # Test grid sampling
        start_time = time.time()
        grid_samples = param_manager.sample_parameters(
            strategy_name, param_ranges, num_samples=5, method='grid')
        grid_time = time.time() - start_time
        
        print(f"Generated {len(grid_samples)} grid samples in {grid_time:.2f} seconds")
        print(f"Grid samples:")
        for i, params in enumerate(grid_samples[:5]):
            print(f"  {i+1}: {params}")
        
        # Test Latin Hypercube sampling
        start_time = time.time()
        latin_samples = param_manager.sample_parameters(
            strategy_name, param_ranges, num_samples=5, method='latin')
        latin_time = time.time() - start_time
        
        print(f"Generated {len(latin_samples)} Latin Hypercube samples in {latin_time:.2f} seconds")
        print(f"Latin Hypercube samples:")
        for i, params in enumerate(latin_samples):
            print(f"  {i+1}: {params}")
        
        # Test parameter saving and loading
        print("\nTesting parameter saving and loading:")
        
        # Save parameters to a file
        params_file = os.path.join(os.path.dirname(current_dir), 'output', 'test_params.json')
        os.makedirs(os.path.dirname(params_file), exist_ok=True)
        
        param_manager.save_parameters(default_params, params_file)
        print(f"Saved parameters to {params_file}")
        
        # Load parameters from the file
        loaded_params = param_manager.load_parameters(params_file)
        print(f"Loaded parameters: {loaded_params}")
        
        # Check if loaded parameters match original
        match = all(loaded_params[k] == default_params[k] for k in default_params)
        print(f"Loaded parameters match original: {match}")
    
    print("\n" + "="*80)
    print("ParameterManager Test Complete")
    print("="*80 + "\n")

if __name__ == '__main__':
    main() 