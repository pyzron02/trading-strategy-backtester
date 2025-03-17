#!/usr/bin/env python3
# check_strategies.py - Script to check available strategies

import os
import sys

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
if src_dir not in sys.path:
    sys.path.append(src_dir)

from strategies import registry

def check_strategies():
    """Check available strategies and print information about them."""
    print("Available strategies:")
    
    # Get registered strategies
    registered_strategies = registry.get_registered_strategies()
    
    for strategy_info in registered_strategies:
        name = strategy_info['name']
        version = strategy_info['version']
        print(f"\n{name} (v{version}):")
        
        try:
            strategy_class = registry.get_strategy_class(name)
            print(f"  Class: {strategy_class.__name__}")
            print(f"  Module: {strategy_class.__module__}")
            
            # Get default parameters if available
            if hasattr(strategy_class, 'params'):
                print("  Default parameters:")
                
                # Handle different ways parameters might be stored
                if hasattr(strategy_class.params, '__dict__'):
                    for param_name, param_value in strategy_class.params.__dict__.items():
                        if not param_name.startswith('_'):
                            print(f"    {param_name}: {param_value}")
                elif isinstance(strategy_class.params, list) or isinstance(strategy_class.params, tuple):
                    for param in strategy_class.params:
                        if isinstance(param, tuple) and len(param) == 2:
                            param_name, param_value = param
                            print(f"    {param_name}: {param_value}")
        except Exception as e:
            print(f"  Error getting details: {e}")

if __name__ == '__main__':
    check_strategies() 