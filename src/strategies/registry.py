#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Registry for managing trading strategies.

This module provides functionality to register and retrieve strategy classes.
"""

# Dictionary to store registered strategies
_registry = {}

def register_strategy(name, strategy_class, version="1.0.1"):
    """
    Register a strategy class with the registry.
    
    Args:
        name (str): The name of the strategy
        strategy_class (class): The strategy class
        version (str): The version of the strategy
    """
    _registry[name] = {
        'class': strategy_class,
        'version': version
    }
    print(f"Registered strategy: {name} (v{version})")

def get_strategy_class(name):
    """
    Get a strategy class from the registry.
    
    Args:
        name (str): The name of the strategy
        
    Returns:
        class: The strategy class
        
    Raises:
        ValueError: If the strategy is not found
    """
    if name in _registry:
        return _registry[name]['class']
    raise ValueError(f"Strategy '{name}' not found in registry")

def get_strategy_version(name):
    """
    Get the version of a strategy from the registry.
    
    Args:
        name (str): The name of the strategy
        
    Returns:
        str: The version of the strategy
        
    Raises:
        ValueError: If the strategy is not found
    """
    if name in _registry:
        return _registry[name]['version']
    raise ValueError(f"Strategy '{name}' not found in registry")

def get_registered_strategies():
    """
    Get a list of all registered strategies.
    
    Returns:
        list: A list of dictionaries containing strategy names and versions
    """
    return [{'name': name, 'version': data['version']} for name, data in _registry.items()]

# Auto-register strategies
try:
    from strategies.simplestock import SimpleStock
    register_strategy('SimpleStock', SimpleStock)
except ImportError:
    print("Could not import SimpleStock strategy")

try:
    from strategies.multi_position_strategy import MultiPositionStrategy
    register_strategy('MultiPosition', MultiPositionStrategy)
except ImportError:
    print("Could not import MultiPositionStrategy strategy")

try:
    from strategies.auction_market_strategy import AuctionMarketStrategy
    register_strategy('AuctionMarket', AuctionMarketStrategy)
except ImportError:
    print("Could not import AuctionMarketStrategy strategy")

try:
    from strategies.ma_crossover import MACrossover
    register_strategy('MACrossover', MACrossover, version="1.0.0")
except ImportError:
    print("Could not import MACrossover strategy") 
    
try:
    from strategies.pairs_trading_strategy import PairsTradingStrategy
    register_strategy('PairsTrading', PairsTradingStrategy, version="1.0.0")
except ImportError:
    try:
        # Alternative import path
        from src.strategies.pairs_trading_strategy import PairsTradingStrategy
        register_strategy('PairsTrading', PairsTradingStrategy, version="1.0.0")
    except ImportError:
        print("Could not import PairsTradingStrategy strategy")