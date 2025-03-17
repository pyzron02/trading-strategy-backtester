#!/usr/bin/env python3
# test_strategy_registry.py - Test script for the StrategyRegistry class

import os
import sys

# Add the parent directory to the path so we can import from strategies
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from strategies import registry, strategy_info
import backtrader as bt

def main():
    """Test the StrategyRegistry class."""
    print("\n" + "="*80)
    print("StrategyRegistry Test")
    print("="*80 + "\n")
    
    # Get all registered strategies
    strategies = registry.get_all_strategies()
    strategy_names = registry.get_strategy_names()
    
    print(f"Found {len(strategies)} registered strategies:")
    for name in strategy_names:
        metadata = registry.get_strategy_metadata(name)
        print(f"  - {name} (v{metadata['version']}) by {metadata['author']}")
        print(f"    {metadata['description']}")
        
        # Print parameters
        params = registry.get_strategy_parameters(name)
        print(f"    Parameters:")
        for param_name, param_value in params.items():
            print(f"      - {param_name}: {param_value}")
        
        print()
    
    # Test creating a new strategy with the decorator
    @strategy_info(
        name="TestStrategy",
        description="A test strategy for demonstration purposes",
        version="0.1.0",
        author="Test Author",
        parameters={
            "param1": 10,
            "param2": 20
        }
    )
    class TestStrategy(bt.Strategy):
        params = (
            ('param1', 10),
            ('param2', 20),
        )
        
        def __init__(self):
            pass
            
        def next(self):
            pass
    
    # Register the test strategy manually
    registry.register_strategy("TestStrategy", TestStrategy)
    
    # Verify the test strategy was registered
    print("After registering test strategy:")
    print(f"Total strategies: {len(registry.get_strategy_names())}")
    
    # Get the test strategy
    test_strategy = registry.get_strategy("TestStrategy")
    print(f"Test strategy class: {test_strategy['class'].__name__}")
    print(f"Test strategy metadata: {test_strategy['metadata']}")
    
    print("\n" + "="*80)
    print("StrategyRegistry Test Complete")
    print("="*80 + "\n")

if __name__ == '__main__':
    main() 