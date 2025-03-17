#!/usr/bin/env python3
# test_parallel_tester.py - Test script for the ParallelTester class

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

from engine.parallel_testing import ParallelTester

# Define some test functions
def test_function_fast(x, y=1):
    """A fast test function."""
    return x + y

def test_function_slow(x, y=1, sleep_time=0.1):
    """A slow test function that sleeps."""
    time.sleep(sleep_time)
    return x + y

def test_function_error(x, y=1):
    """A test function that raises an error."""
    if x == 0:
        raise ValueError("x cannot be zero")
    return x + y

def test_function_dataframe(rows=10, cols=3):
    """A test function that returns a DataFrame."""
    return pd.DataFrame(
        np.random.normal(0, 1, (rows, cols)),
        columns=[f'col_{i}' for i in range(cols)]
    )

def main():
    """Test the ParallelTester class."""
    print("\n" + "="*80)
    print("ParallelTester Test")
    print("="*80 + "\n")
    
    # Create ParallelTester instance
    tester = ParallelTester(max_workers=4)
    
    # Test adding individual tasks
    print("Testing adding individual tasks:")
    
    # Add some fast tasks
    for i in range(5):
        tester.add_task(f"fast_{i}", test_function_fast, (i,), {'y': i*2})
    
    # Add some slow tasks
    for i in range(3):
        tester.add_task(f"slow_{i}", test_function_slow, (i,), {'y': i*2, 'sleep_time': 0.5})
    
    # Add a task that will raise an error
    tester.add_task("error_task", test_function_error, (0,))
    
    # Add a task that returns a DataFrame
    tester.add_task("df_task", test_function_dataframe, (), {'rows': 100, 'cols': 5})
    
    # Print task queue
    print(f"Task queue: {len(tester.task_queue)} tasks")
    for task in tester.task_queue:
        print(f"  {task['id']}: {task['func'].__name__}({task['args']}, {task['kwargs']})")
    
    # Test parameter sweep
    print("\nTesting parameter sweep:")
    
    # Clear tasks
    tester.clear_tasks()
    
    # Add parameter sweep
    tester.add_parameter_sweep(
        base_task_id="sweep",
        task_func=test_function_slow,
        param_name="sleep_time",
        param_values=[0.1, 0.2, 0.3, 0.4, 0.5],
        task_args=(10,),
        task_kwargs={'y': 5}
    )
    
    # Print task queue
    print(f"Task queue: {len(tester.task_queue)} tasks")
    for task in tester.task_queue:
        print(f"  {task['id']}: {task['func'].__name__}({task['args']}, {task['kwargs']})")
    
    # Test grid search
    print("\nTesting grid search:")
    
    # Clear tasks
    tester.clear_tasks()
    
    # Add grid search
    tester.add_grid_search(
        base_task_id="grid",
        task_func=test_function_slow,
        param_grid={
            'sleep_time': [0.1, 0.2],
            'y': [1, 2, 3]
        },
        task_args=(10,)
    )
    
    # Print task queue
    print(f"Task queue: {len(tester.task_queue)} tasks")
    for task in tester.task_queue:
        print(f"  {task['id']}: {task['func'].__name__}({task['args']}, {task['kwargs']})")
    
    # Test running tasks
    print("\nTesting running tasks:")
    
    # Run tasks
    results = tester.run_tasks()
    
    # Print results
    print(f"Results: {len(results)} results")
    for task_id, result in results.items():
        print(f"  {task_id}: {result}")
    
    # Print errors
    errors = tester.get_errors()
    if errors:
        print(f"Errors: {len(errors)} errors")
        for task_id, error in errors.items():
            print(f"  {task_id}: {error['exception']}")
    
    # Print execution times
    execution_times = tester.get_execution_times()
    print(f"Execution times: {len(execution_times)} tasks")
    for task_id, time_taken in execution_times.items():
        print(f"  {task_id}: {time_taken:.2f} seconds")
    
    # Print average execution time
    avg_time = tester.get_average_execution_time()
    print(f"Average execution time: {avg_time:.2f} seconds")
    
    # Print total execution time
    total_time = tester.get_total_execution_time()
    print(f"Total execution time: {total_time:.2f} seconds")
    
    # Print speedup
    speedup = tester.get_speedup()
    print(f"Speedup: {speedup:.2f}x")
    
    # Test mixed tasks
    print("\nTesting mixed tasks:")
    
    # Clear tasks
    tester.clear_tasks()
    
    # Add some fast tasks
    for i in range(10):
        tester.add_task(f"fast_{i}", test_function_fast, (i,), {'y': i*2})
    
    # Add some slow tasks
    for i in range(5):
        tester.add_task(f"slow_{i}", test_function_slow, (i,), {'y': i*2, 'sleep_time': 0.2})
    
    # Add some error tasks
    for i in range(3):
        tester.add_task(f"error_{i}", test_function_error, (0 if i == 1 else i,))
    
    # Add some DataFrame tasks
    for i in range(2):
        tester.add_task(f"df_{i}", test_function_dataframe, (), {'rows': 50 * (i+1), 'cols': 5})
    
    # Run tasks
    results = tester.run_tasks()
    
    # Print summary
    print(f"Results: {len(results)} results")
    print(f"Errors: {len(tester.get_errors())} errors")
    print(f"Average execution time: {tester.get_average_execution_time():.2f} seconds")
    print(f"Total execution time: {tester.get_total_execution_time():.2f} seconds")
    print(f"Speedup: {tester.get_speedup():.2f}x")
    
    print("\n" + "="*80)
    print("ParallelTester Test Complete")
    print("="*80 + "\n")

if __name__ == '__main__':
    main() 