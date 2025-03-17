#!/usr/bin/env python3
# test_smart_cache.py - Test script for the SmartCache class and cached decorator

import os
import sys
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from engine.smart_cache import cache, cached

# Define some test functions
@cached()
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

@cached(dependencies=['fibonacci'])
def fibonacci_sum(n):
    """Calculate the sum of Fibonacci numbers up to n."""
    return sum(fibonacci(i) for i in range(n+1))

@cached(ttl=5)
def current_time():
    """Return the current time."""
    return datetime.now().isoformat()

@cached()
def expensive_calculation(size=1000):
    """Simulate an expensive calculation."""
    time.sleep(0.1)  # Simulate work
    return np.random.normal(0, 1, size).mean()

@cached()
def create_dataframe(rows=100, cols=5):
    """Create a random DataFrame."""
    time.sleep(0.1)  # Simulate work
    return pd.DataFrame(
        np.random.normal(0, 1, (rows, cols)),
        columns=[f'col_{i}' for i in range(cols)]
    )

def main():
    """Test the SmartCache class and cached decorator."""
    print("\n" + "="*80)
    print("SmartCache Test")
    print("="*80 + "\n")
    
    # Clear the cache
    cache.clear()
    
    # Test basic caching
    print("Testing basic caching:")
    
    # Calculate Fibonacci numbers
    start_time = time.time()
    fib_30 = fibonacci(30)
    first_run_time = time.time() - start_time
    
    print(f"Fibonacci(30) = {fib_30}")
    print(f"First run time: {first_run_time:.4f} seconds")
    
    # Calculate again (should be cached)
    start_time = time.time()
    fib_30_cached = fibonacci(30)
    cached_run_time = time.time() - start_time
    
    print(f"Fibonacci(30) from cache = {fib_30_cached}")
    print(f"Cached run time: {cached_run_time:.4f} seconds")
    print(f"Speedup: {first_run_time / max(cached_run_time, 0.0001):.1f}x")
    
    # Test dependency tracking
    print("\nTesting dependency tracking:")
    
    # Calculate Fibonacci sum
    fib_sum_10 = fibonacci_sum(10)
    print(f"Fibonacci sum up to 10 = {fib_sum_10}")
    
    # Invalidate fibonacci(5)
    print("Invalidating fibonacci(5)...")
    for key in list(cache.memory_cache.keys()):
        if 'fibonacci' in key and '5' in key:
            cache.invalidate(key)
    
    # Calculate Fibonacci sum again (should recalculate dependencies)
    start_time = time.time()
    fib_sum_10_new = fibonacci_sum(10)
    recalc_time = time.time() - start_time
    
    print(f"Fibonacci sum up to 10 (after invalidation) = {fib_sum_10_new}")
    print(f"Recalculation time: {recalc_time:.4f} seconds")
    
    # Test TTL
    print("\nTesting TTL:")
    
    # Get current time
    time1 = current_time()
    print(f"Current time: {time1}")
    
    # Get current time again (should be cached)
    time2 = current_time()
    print(f"Cached time: {time2}")
    print(f"Same time: {time1 == time2}")
    
    # Wait for TTL to expire
    print("Waiting for TTL to expire (5 seconds)...")
    time.sleep(6)
    
    # Get current time again (should be updated)
    time3 = current_time()
    print(f"New time after TTL: {time3}")
    print(f"Different time: {time2 != time3}")
    
    # Test with different data types
    print("\nTesting with different data types:")
    
    # Test with numeric calculation
    result1 = expensive_calculation(1000)
    print(f"Expensive calculation result: {result1:.4f}")
    
    # Test with DataFrame
    df = create_dataframe(100, 5)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame head:\n{df.head(3)}")
    
    # Test cache statistics
    print("\nCache statistics:")
    stats = cache.get_stats()
    for stat, value in stats.items():
        print(f"  {stat}: {value}")
    
    # Test cache size
    size_info = cache.get_size()
    print("\nCache size:")
    for key, value in size_info.items():
        if key == 'disk_size_bytes' and value > 0:
            # Convert to KB or MB for readability
            if value > 1024 * 1024:
                print(f"  {key}: {value / (1024 * 1024):.2f} MB")
            else:
                print(f"  {key}: {value / 1024:.2f} KB")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("SmartCache Test Complete")
    print("="*80 + "\n")

if __name__ == '__main__':
    main() 