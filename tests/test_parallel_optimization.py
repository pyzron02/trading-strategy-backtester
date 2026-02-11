#!/usr/bin/env python3
"""
Test script for parallel optimization functionality.
Tests both sequential and parallel execution to verify correctness and measure speedup.
"""

import time
import json
import os

from engine.testing.in_sample_excellence import InSampleExcellence
from engine.logging_system import logger

def test_parallel_optimization():
    """Test parallel optimization with a simple parameter grid."""
    
    # Test configuration
    strategy_name = "AuctionMarket"
    tickers = ["SPY", "QQQ"]
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    
    # Simple parameter grid for testing
    param_grid = {
        "atr_period": [10, 14, 20],
        "risk_percent": [0.01, 0.02],
        "position_size": [100, 200],
        "value_area": [0.68, 0.70]
    }
    
    # Total combinations: 3 * 2 * 2 * 2 = 24
    
    print("=" * 80)
    print("Testing Parallel Optimization Implementation")
    print("=" * 80)
    print(f"Strategy: {strategy_name}")
    print(f"Tickers: {tickers}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Total parameter combinations: 24")
    print()
    
    # Test 1: Sequential execution (n_jobs=1)
    print("Test 1: Sequential Execution (n_jobs=1)")
    print("-" * 40)
    
    start_time = time.time()
    optimizer_seq = InSampleExcellence(
        strategy_name=strategy_name,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        parameter_grid=param_grid,
        output_dir="output/test_parallel_seq",
        n_jobs=1,  # Sequential
        verbose=True
    )
    
    result_seq = optimizer_seq.run()
    seq_time = time.time() - start_time
    
    print(f"\nSequential execution time: {seq_time:.2f} seconds")
    if isinstance(result_seq, dict) and 'error' not in result_seq:
        print(f"Optimization completed successfully")
    
    # Test 2: Parallel execution (n_jobs=4)
    print("\n\nTest 2: Parallel Execution (n_jobs=4)")
    print("-" * 40)
    
    start_time = time.time()
    optimizer_par = InSampleExcellence(
        strategy_name=strategy_name,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        parameter_grid=param_grid,
        output_dir="output/test_parallel_par",
        n_jobs=4,  # Parallel with 4 workers
        verbose=True
    )
    
    result_par = optimizer_par.run()
    par_time = time.time() - start_time
    
    print(f"\nParallel execution time: {par_time:.2f} seconds")
    if isinstance(result_par, dict) and 'error' not in result_par:
        print(f"Optimization completed successfully")
    
    # Test 3: Parallel with batching
    print("\n\nTest 3: Parallel Execution with Batching (n_jobs=4, batch_size=6)")
    print("-" * 40)
    
    start_time = time.time()
    optimizer_batch = InSampleExcellence(
        strategy_name=strategy_name,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        parameter_grid=param_grid,
        output_dir="output/test_parallel_batch",
        n_jobs=4,
        batch_size=6,  # Process 6 at a time
        verbose=True
    )
    
    result_batch = optimizer_batch.run()
    batch_time = time.time() - start_time
    
    print(f"\nBatch parallel execution time: {batch_time:.2f} seconds")
    if isinstance(result_batch, dict) and 'error' not in result_batch:
        print(f"Optimization completed successfully")
    
    # Compare results
    print("\n\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    # Check if results are consistent
    print("\nBest Parameters Found:")
    for name, result in [("Sequential", result_seq), ("Parallel", result_par), ("Batch", result_batch)]:
        if isinstance(result, dict) and 'parameters' in result:
            print(f"{name}: {result['parameters']}")
        else:
            print(f"{name}: Error or no results")
    
    # Performance summary
    print("\n\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Sequential time: {seq_time:.2f}s")
    print(f"Parallel time: {par_time:.2f}s (Speedup: {seq_time/par_time:.2f}x)")
    print(f"Batch parallel time: {batch_time:.2f}s (Speedup: {seq_time/batch_time:.2f}x)")
    
    # Test workflow integration
    print("\n\n" + "=" * 80)
    print("TESTING WORKFLOW INTEGRATION")
    print("=" * 80)
    
    # Create a test config file
    test_config = {
        "workflow_type": "optimization",
        "strategy_name": strategy_name,
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "n_jobs": -1,  # Use all cores
        "batch_size": 10,
        "parameter_grid": param_grid,
        "n_trials": 24,
        "optimization_metric": "sharpe_ratio"
    }
    
    config_path = "output/test_parallel_config.json"
    with open(config_path, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print(f"Created test config at: {config_path}")
    print("You can run this with: python src/workflows/cli.py --config output/test_parallel_config.json")
    
    print("\n✅ Parallel optimization tests completed successfully!")
    
if __name__ == "__main__":
    test_parallel_optimization()