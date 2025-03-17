#!/usr/bin/env python3
# test_logging_system.py - Test script for the LoggingSystem class and its decorators

import os
import sys
import time
import random

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from engine.logging_system import logger, log_execution_time, disable_logging

# Define some test functions
@log_execution_time('testing', 'INFO')
def slow_function(sleep_time=0.5):
    """A slow function that sleeps."""
    time.sleep(sleep_time)
    return f"Slept for {sleep_time} seconds"

@log_execution_time('testing', 'DEBUG', threshold=0.1)
def sometimes_slow_function(sleep_time=0.05):
    """A function that is sometimes slow."""
    time.sleep(sleep_time)
    return f"Slept for {sleep_time} seconds"

@disable_logging
def noisy_function():
    """A function that would normally log a lot."""
    logger.info('testing', "This message should not appear")
    logger.debug('testing', "This debug message should not appear")
    logger.warning('testing', "This warning should not appear")
    return "Noisy function completed silently"

def main():
    """Test the LoggingSystem class and its decorators."""
    print("\n" + "="*80)
    print("LoggingSystem Test")
    print("="*80 + "\n")
    
    # Test basic logging
    print("Testing basic logging:")
    
    # Log messages at different levels
    logger.debug('testing', "This is a debug message")
    logger.info('testing', "This is an info message")
    logger.warning('testing', "This is a warning message")
    logger.error('testing', "This is an error message")
    logger.critical('testing', "This is a critical message")
    
    # Test logging with formatting
    logger.info('testing', "Formatted message: %s, %d, %.2f", "string", 42, 3.14159)
    
    # Test logging with keyword arguments
    logger.info('testing', "Message with extra data", extra={'key': 'value'})
    
    # Test setting log levels
    print("\nTesting log level changes:")
    
    # Set level to DEBUG for testing component
    logger.set_level('DEBUG', 'testing')
    logger.debug('testing', "This debug message should appear")
    
    # Set level to WARNING for testing component
    logger.set_level('WARNING', 'testing')
    logger.debug('testing', "This debug message should NOT appear")
    logger.info('testing', "This info message should NOT appear")
    logger.warning('testing', "This warning message should appear")
    
    # Reset level to INFO
    logger.set_level('INFO', 'testing')
    
    # Test performance tracking
    print("\nTesting performance tracking:")
    
    # Start tracking performance
    logger.performance_start('testing', 'slow_operation')
    
    # Simulate work
    time.sleep(0.2)
    
    # End tracking performance (should log because it's above threshold)
    logger.performance_end('testing', 'slow_operation')
    
    # Start tracking performance
    logger.performance_start('testing', 'fast_operation')
    
    # Simulate work
    time.sleep(0.05)
    
    # End tracking performance (should not log because it's below threshold)
    logger.performance_end('testing', 'fast_operation')
    
    # Test execution time decorator
    print("\nTesting execution time decorator:")
    
    # Call slow function (should log execution time)
    result = slow_function()
    print(f"Slow function result: {result}")
    
    # Call sometimes slow function with fast execution (should not log)
    result = sometimes_slow_function(0.05)
    print(f"Sometimes slow function result (fast): {result}")
    
    # Call sometimes slow function with slow execution (should log)
    result = sometimes_slow_function(0.15)
    print(f"Sometimes slow function result (slow): {result}")
    
    # Test disable logging decorator
    print("\nTesting disable logging decorator:")
    
    # Log a message before
    logger.info('testing', "This message should appear")
    
    # Call noisy function (should not log anything)
    result = noisy_function()
    print(f"Noisy function result: {result}")
    
    # Log a message after
    logger.info('testing', "This message should also appear")
    
    # Test async logging
    print("\nTesting async logging:")
    
    # Log a bunch of messages quickly
    for i in range(100):
        logger.debug('testing', f"Async message {i}")
    
    # Wait for async logging to complete
    time.sleep(0.5)
    
    print("\n" + "="*80)
    print("LoggingSystem Test Complete")
    print("="*80 + "\n")

if __name__ == '__main__':
    main() 