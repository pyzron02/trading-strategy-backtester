#!/usr/bin/env python3
# test_imports.py - Test script to verify imports are working

import os
import sys
import importlib

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def test_import(module_name):
    """Test if a module can be imported."""
    try:
        module = importlib.import_module(module_name)
        print(f"✓ Successfully imported {module_name}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {module_name}: {e}")
        return False

def main():
    """Test all required imports."""
    print("\n=== Testing Required Imports ===\n")
    
    # Test basic Python packages
    packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "backtrader"
    ]
    
    for package in packages:
        test_import(package)
    
    print("\n=== Testing Project Modules ===\n")
    
    # Test project modules
    modules = [
        "engine.testing",
        "engine.testing.strategy_tester",
        "engine.testing.in_sample_excellence",
        "engine.testing.in_sample_monte_carlo",
        "engine.testing.walk_forward_test",
        "engine.testing.walk_forward_monte_carlo",
        "strategies.simplestock",
        "strategies.multi_position_strategy",
        "strategies.auction_market_strategy"
    ]
    
    for module in modules:
        test_import(module)
    
    print("\n=== Import Test Complete ===\n")

if __name__ == "__main__":
    main() 