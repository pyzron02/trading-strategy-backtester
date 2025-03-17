#!/usr/bin/env python3
# parameter_management.py - Centralized parameter management system

import os
import sys
import numpy as np
import itertools
import random
from typing import Dict, List, Any, Tuple, Union, Optional
import json

class ParameterManager:
    """
    Centralized parameter management system for trading strategies.
    
    This class handles parameter definitions, validation, and sampling
    for strategy optimization.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern to ensure only one parameter manager exists."""
        if cls._instance is None:
            cls._instance = super(ParameterManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the ParameterManager."""
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
            
        # Dictionary to store parameter definitions
        self.parameter_definitions = {}
        
        # Dictionary to store parameter constraints
        self.parameter_constraints = {}
        
        self._initialized = True
    
    def define_parameter(self, strategy_name: str, param_name: str, param_type: type,
                        default_value: Any, min_value: Optional[Any] = None, 
                        max_value: Optional[Any] = None, choices: Optional[List[Any]] = None,
                        description: str = ""):
        """
        Define a parameter for a strategy.
        
        Args:
            strategy_name (str): Name of the strategy
            param_name (str): Name of the parameter
            param_type (type): Type of the parameter (int, float, bool, str)
            default_value (Any): Default value for the parameter
            min_value (Any, optional): Minimum value for the parameter
            max_value (Any, optional): Maximum value for the parameter
            choices (List[Any], optional): List of valid choices for the parameter
            description (str): Description of the parameter
        """
        # Initialize strategy parameters if not already defined
        if strategy_name not in self.parameter_definitions:
            self.parameter_definitions[strategy_name] = {}
        
        # Define the parameter
        self.parameter_definitions[strategy_name][param_name] = {
            'type': param_type,
            'default': default_value,
            'min': min_value,
            'max': max_value,
            'choices': choices,
            'description': description
        }
    
    def _get_strategy_params(self, strategy_class):
        """
        Extract parameters from a strategy class.
        
        Args:
            strategy_class (class): Strategy class
            
        Returns:
            dict: Dictionary of parameter names and values
        """
        params = {}
        if hasattr(strategy_class, 'params'):
            # Handle tuple of tuples format: ((name1, value1), (name2, value2), ...)
            if isinstance(strategy_class.params, tuple):
                for param_tuple in strategy_class.params:
                    if isinstance(param_tuple, tuple) and len(param_tuple) == 2:
                        param_name, param_value = param_tuple
                        params[param_name] = param_value
            # Handle dictionary format
            elif isinstance(strategy_class.params, dict):
                params = strategy_class.params.copy()
        return params
    
    def define_parameters_from_strategy(self, strategy_name: str, strategy_class):
        """
        Define parameters from a strategy class.
        
        Args:
            strategy_name (str): Name of the strategy
            strategy_class (class): Strategy class
        """
        # Get parameters from strategy class
        try:
            params = self._get_strategy_params(strategy_class)
            
            # Define each parameter
            for param_name, param_value in params.items():
                param_type = type(param_value)
                
                # Define the parameter with default values
                self.define_parameter(
                    strategy_name=strategy_name,
                    param_name=param_name,
                    param_type=param_type,
                    default_value=param_value,
                    description=f"Parameter {param_name} for {strategy_name}"
                )
            
            # Add the strategy to parameter_definitions if it's empty
            if strategy_name not in self.parameter_definitions:
                self.parameter_definitions[strategy_name] = {}
                
        except Exception as e:
            print(f"Error defining parameters for {strategy_name}: {e}")
            # Create an empty parameter definition to avoid future errors
            if strategy_name not in self.parameter_definitions:
                self.parameter_definitions[strategy_name] = {}
    
    def define_parameters_from_registry(self, registry):
        """
        Define parameters from the strategy registry.
        
        Args:
            registry: Strategy registry instance
        """
        # Get all strategies from the registry
        for strategy_name in registry.get_strategy_names():
            # Get strategy class and metadata
            strategy_info = registry.get_strategy(strategy_name)
            strategy_class = strategy_info['class']
            
            # Define parameters from the strategy class
            self.define_parameters_from_strategy(strategy_name, strategy_class)
    
    def add_constraint(self, strategy_name: str, constraint_func, description: str = ""):
        """
        Add a constraint function for parameter validation.
        
        Args:
            strategy_name (str): Name of the strategy
            constraint_func (callable): Function that takes parameters dict and returns bool
            description (str): Description of the constraint
        """
        # Initialize strategy constraints if not already defined
        if strategy_name not in self.parameter_constraints:
            self.parameter_constraints[strategy_name] = []
        
        # Add the constraint
        self.parameter_constraints[strategy_name].append({
            'func': constraint_func,
            'description': description
        })
    
    def validate_parameters(self, strategy_name: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate parameters for a strategy.
        
        Args:
            strategy_name (str): Name of the strategy
            parameters (Dict[str, Any]): Parameters to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check if strategy exists
        if strategy_name not in self.parameter_definitions:
            print(f"Warning: Strategy '{strategy_name}' not found in parameter definitions")
            # Try to get the strategy from the registry
            try:
                from strategies import registry
                strategy_class = registry.get_strategy_class(strategy_name)
                self.define_parameters_from_strategy(strategy_name, strategy_class)
            except Exception as e:
                print(f"Error getting strategy from registry: {e}")
                # Return True to avoid errors
                return True, ""
        
        # If no parameter definitions, return True
        if not self.parameter_definitions[strategy_name]:
            return True, ""
        
        # Check required parameters
        for param_name, param_def in self.parameter_definitions[strategy_name].items():
            if param_def['required'] and param_name not in parameters:
                return False, f"Required parameter '{param_name}' missing"
        
        # Check parameter types and constraints
        for param_name, param_value in parameters.items():
            # Skip parameters not in definition
            if param_name not in self.parameter_definitions[strategy_name]:
                continue
                
            param_def = self.parameter_definitions[strategy_name][param_name]
            
            # Check type
            if not isinstance(param_value, param_def['type']):
                return False, f"Parameter '{param_name}' should be of type {param_def['type'].__name__}"
            
            # Check min/max for numeric types
            if isinstance(param_value, (int, float)):
                if param_def['min'] is not None and param_value < param_def['min']:
                    return False, f"Parameter '{param_name}' should be >= {param_def['min']}"
                if param_def['max'] is not None and param_value > param_def['max']:
                    return False, f"Parameter '{param_name}' should be <= {param_def['max']}"
            
            # Check choices
            if param_def['choices'] is not None and param_value not in param_def['choices']:
                return False, f"Parameter '{param_name}' should be one of {param_def['choices']}"
        
        # Check constraints
        if strategy_name in self.parameter_constraints:
            for constraint in self.parameter_constraints[strategy_name]:
                if not constraint['func'](parameters):
                    return False, f"Constraint failed: {constraint['description']}"
        
        return True, ""
    
    def get_default_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get default parameters for a strategy.
        
        Args:
            strategy_name (str): Name of the strategy
            
        Returns:
            Dict[str, Any]: Default parameters
        """
        # Check if strategy exists
        if strategy_name not in self.parameter_definitions:
            print(f"Warning: Strategy '{strategy_name}' not found in parameter definitions")
            # Try to get the strategy from the registry
            try:
                from strategies import registry
                strategy_class = registry.get_strategy_class(strategy_name)
                self.define_parameters_from_strategy(strategy_name, strategy_class)
            except Exception as e:
                print(f"Error getting strategy from registry: {e}")
                # Return an empty dictionary to avoid errors
                return {}
        
        # Get default parameters
        default_params = {}
        for param_name, param_def in self.parameter_definitions[strategy_name].items():
            default_params[param_name] = param_def['default']
        
        return default_params
    
    def create_parameter_grid(self, strategy_name: str, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Create a grid of parameter combinations for optimization.
        
        Args:
            strategy_name (str): Name of the strategy
            param_grid (Dict[str, List[Any]]): Grid of parameter values
            
        Returns:
            List[Dict[str, Any]]: List of parameter combinations
        """
        # Check if strategy exists
        if strategy_name not in self.parameter_definitions:
            print(f"Warning: Strategy '{strategy_name}' not found in parameter definitions")
            # Try to get the strategy from the registry
            try:
                from strategies import registry
                strategy_class = registry.get_strategy_class(strategy_name)
                self.define_parameters_from_strategy(strategy_name, strategy_class)
            except Exception as e:
                print(f"Error getting strategy from registry: {e}")
                # Create an empty parameter definition to avoid errors
                self.parameter_definitions[strategy_name] = {}
        
        # Get default parameters
        default_params = self.get_default_parameters(strategy_name)
        
        # Use param_grid directly if parameters are not defined
        if not self.parameter_definitions[strategy_name]:
            print(f"Using param_grid directly for strategy '{strategy_name}'")
            # Create all combinations of parameters
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            combinations = list(itertools.product(*param_values))
            
            # Create list of parameter dictionaries
            grid = []
            for combo in combinations:
                params = {}
                for i, param_name in enumerate(param_names):
                    params[param_name] = combo[i]
                grid.append(params)
            
            return grid
        
        # Validate parameters
        for param_name in param_grid:
            if param_name not in self.parameter_definitions[strategy_name]:
                raise ValueError(f"Parameter '{param_name}' not defined for strategy '{strategy_name}'")
        
        # Create all combinations of parameters
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        # Create list of parameter dictionaries
        grid = []
        for combo in combinations:
            params = default_params.copy()
            for i, param_name in enumerate(param_names):
                params[param_name] = combo[i]
            
            # Validate parameters
            valid, message = self.validate_parameters(strategy_name, params)
            if valid:
                grid.append(params)
            else:
                print(f"Skipping invalid parameter combination: {message}")
        
        return grid
    
    def sample_parameters(self, strategy_name: str, param_ranges: Dict[str, Union[List[Any], Tuple[Any, Any]]],
                         num_samples: int = 10, method: str = 'random') -> List[Dict[str, Any]]:
        """
        Sample parameters from ranges for optimization.
        
        Args:
            strategy_name (str): Name of the strategy
            param_ranges (Dict[str, Union[List[Any], Tuple[Any, Any]]]): Parameter ranges
            num_samples (int): Number of samples to generate
            method (str): Sampling method ('random', 'grid', 'latin')
            
        Returns:
            List[Dict[str, Any]]: List of parameter combinations
        """
        # Check if strategy exists
        if strategy_name not in self.parameter_definitions:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        # Get default parameters
        default_params = self.get_default_parameters(strategy_name)
        
        # Validate param_ranges
        for param_name in param_ranges:
            if param_name not in default_params:
                raise ValueError(f"Parameter '{param_name}' not defined for strategy '{strategy_name}'")
        
        # Sample parameters based on method
        if method == 'random':
            return self._random_sample(strategy_name, default_params, param_ranges, num_samples)
        elif method == 'grid':
            return self._grid_sample(strategy_name, default_params, param_ranges, num_samples)
        elif method == 'latin':
            return self._latin_hypercube_sample(strategy_name, default_params, param_ranges, num_samples)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def _random_sample(self, strategy_name: str, default_params: Dict[str, Any],
                      param_ranges: Dict[str, Union[List[Any], Tuple[Any, Any]]],
                      num_samples: int) -> List[Dict[str, Any]]:
        """Generate random parameter samples."""
        param_dicts = []
        
        for _ in range(num_samples):
            # Start with default parameters
            params = default_params.copy()
            
            # Sample each parameter
            for param_name, param_range in param_ranges.items():
                param_def = self.parameter_definitions[strategy_name][param_name]
                
                if isinstance(param_range, list):
                    # Sample from discrete list
                    params[param_name] = random.choice(param_range)
                elif isinstance(param_range, tuple) and len(param_range) == 2:
                    # Sample from range
                    min_val, max_val = param_range
                    
                    if param_def['type'] == int:
                        params[param_name] = random.randint(min_val, max_val)
                    elif param_def['type'] == float:
                        params[param_name] = random.uniform(min_val, max_val)
                    elif param_def['type'] == bool:
                        params[param_name] = random.choice([True, False])
                    else:
                        # For other types, use the default value
                        pass
            
            # Validate parameters
            is_valid, error = self.validate_parameters(strategy_name, params)
            if is_valid:
                param_dicts.append(params)
            else:
                # Try again if invalid
                continue
        
        return param_dicts
    
    def _grid_sample(self, strategy_name: str, default_params: Dict[str, Any],
                    param_ranges: Dict[str, Union[List[Any], Tuple[Any, Any]]],
                    num_samples: int) -> List[Dict[str, Any]]:
        """Generate grid-based parameter samples."""
        # Convert ranges to lists
        param_grid = {}
        for param_name, param_range in param_ranges.items():
            param_def = self.parameter_definitions[strategy_name][param_name]
            
            if isinstance(param_range, list):
                # Use the list as is
                param_grid[param_name] = param_range
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                # Create a grid from the range
                min_val, max_val = param_range
                
                if param_def['type'] == int:
                    # Calculate step size to get approximately num_samples points
                    step = max(1, (max_val - min_val) // int(num_samples ** (1 / len(param_ranges))))
                    param_grid[param_name] = list(range(min_val, max_val + 1, step))
                elif param_def['type'] == float:
                    # Calculate step size to get approximately num_samples points
                    step = (max_val - min_val) / int(num_samples ** (1 / len(param_ranges)))
                    param_grid[param_name] = list(np.arange(min_val, max_val + step/2, step))
                elif param_def['type'] == bool:
                    param_grid[param_name] = [True, False]
                else:
                    # For other types, use the default value
                    param_grid[param_name] = [param_def['default']]
        
        # Create grid
        return self.create_parameter_grid(strategy_name, param_grid)
    
    def _latin_hypercube_sample(self, strategy_name: str, default_params: Dict[str, Any],
                               param_ranges: Dict[str, Union[List[Any], Tuple[Any, Any]]],
                               num_samples: int) -> List[Dict[str, Any]]:
        """Generate Latin Hypercube parameter samples."""
        param_dicts = []
        
        # Get continuous parameters (int and float)
        continuous_params = {}
        discrete_params = {}
        
        for param_name, param_range in param_ranges.items():
            param_def = self.parameter_definitions[strategy_name][param_name]
            
            if isinstance(param_range, tuple) and len(param_range) == 2:
                if param_def['type'] in (int, float):
                    continuous_params[param_name] = param_range
                else:
                    discrete_params[param_name] = param_range
            else:
                discrete_params[param_name] = param_range
        
        # Generate Latin Hypercube samples for continuous parameters
        if continuous_params:
            # Number of continuous parameters
            n_params = len(continuous_params)
            
            # Generate Latin Hypercube samples
            samples = np.zeros((num_samples, n_params))
            
            # Generate samples for each parameter
            for j in range(n_params):
                # Generate random permutation of segments
                perm = np.random.permutation(num_samples)
                
                # Generate uniform samples within each segment
                samples[:, j] = (perm + np.random.uniform(0, 1, num_samples)) / num_samples
            
            # Scale samples to parameter ranges
            param_names = list(continuous_params.keys())
            for i in range(num_samples):
                # Start with default parameters
                params = default_params.copy()
                
                # Set continuous parameters
                for j, param_name in enumerate(param_names):
                    param_def = self.parameter_definitions[strategy_name][param_name]
                    min_val, max_val = continuous_params[param_name]
                    
                    # Scale sample to parameter range
                    if param_def['type'] == int:
                        params[param_name] = int(min_val + samples[i, j] * (max_val - min_val))
                    else:
                        params[param_name] = min_val + samples[i, j] * (max_val - min_val)
                
                # Set discrete parameters randomly
                for param_name, param_range in discrete_params.items():
                    if isinstance(param_range, list):
                        params[param_name] = random.choice(param_range)
                    elif isinstance(param_range, tuple) and len(param_range) == 2:
                        param_def = self.parameter_definitions[strategy_name][param_name]
                        min_val, max_val = param_range
                        
                        if param_def['type'] == bool:
                            params[param_name] = random.choice([True, False])
                
                # Validate parameters
                is_valid, error = self.validate_parameters(strategy_name, params)
                if is_valid:
                    param_dicts.append(params)
                else:
                    # Try again with random parameters
                    continue
        else:
            # Fall back to random sampling for discrete parameters
            param_dicts = self._random_sample(strategy_name, default_params, param_ranges, num_samples)
        
        return param_dicts
    
    def save_parameters(self, parameters: Dict[str, Any], filepath: str):
        """
        Save parameters to a JSON file.
        
        Args:
            parameters (Dict[str, Any]): Parameters to save
            filepath (str): Path to save the parameters
        """
        with open(filepath, 'w') as f:
            json.dump(parameters, f, indent=4)
    
    def load_parameters(self, filepath: str) -> Dict[str, Any]:
        """
        Load parameters from a JSON file.
        
        Args:
            filepath (str): Path to load the parameters from
            
        Returns:
            Dict[str, Any]: Loaded parameters
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def load_parameter_grid(self, strategy_name: str, param_file: str = None) -> Dict[str, List[Any]]:
        """
        Load parameter grid from a JSON file.
        
        Args:
            strategy_name (str): Name of the strategy
            param_file (str, optional): Path to the parameter file
            
        Returns:
            Dict[str, List[Any]]: Parameter grid
        """
        if param_file:
            # Check if param_file is a full path or just a filename
            if not os.path.isabs(param_file):
                # Check if file exists in strategies directory
                strategies_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'strategies')
                param_path = os.path.join(strategies_dir, param_file)
                if os.path.exists(param_path):
                    param_file = param_path
            
            # Load parameter grid from file
            try:
                with open(param_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading parameter grid from {param_file}: {e}")
                # Fall back to default parameters
                return self.get_default_parameters(strategy_name)
        else:
            # Use default parameters
            return self.get_default_parameters(strategy_name) 