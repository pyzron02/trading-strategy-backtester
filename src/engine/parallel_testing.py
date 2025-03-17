#!/usr/bin/env python3
# parallel_testing.py - Parallel testing framework

import os
import sys
import time
import multiprocessing as mp
from multiprocessing import Pool, Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Union, Optional, Callable
import traceback
from tqdm import tqdm

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from engine.results_management import ResultsManager

class ParallelTester:
    """
    Parallel testing framework for running multiple tests concurrently.
    
    This class provides functionality for running tests in parallel using
    multiprocessing, with progress tracking and result aggregation.
    """
    
    def __init__(self, max_workers=None, use_tqdm=True):
        """
        Initialize the ParallelTester.
        
        Args:
            max_workers (int, optional): Maximum number of worker processes
            use_tqdm (bool): Whether to use tqdm for progress tracking
        """
        # Set maximum number of workers
        self.max_workers = max_workers or mp.cpu_count()
        
        # Set whether to use tqdm
        self.use_tqdm = use_tqdm
        
        # Get results manager
        self.results_manager = ResultsManager()
        
        # Initialize task queue
        self.task_queue = []
        
        # Initialize results
        self.results = {}
        
        # Initialize errors
        self.errors = {}
        
        # Initialize progress tracking
        self.progress = {}
    
    def add_task(self, task_id: str, task_func: Callable, task_args: Tuple = (), 
                task_kwargs: Dict[str, Any] = None):
        """
        Add a task to the queue.
        
        Args:
            task_id (str): Unique identifier for the task
            task_func (callable): Function to execute
            task_args (tuple): Positional arguments for the function
            task_kwargs (dict): Keyword arguments for the function
        """
        if task_kwargs is None:
            task_kwargs = {}
            
        self.task_queue.append({
            'id': task_id,
            'func': task_func,
            'args': task_args,
            'kwargs': task_kwargs
        })
        
        # Initialize progress tracking for this task
        self.progress[task_id] = {
            'status': 'queued',
            'progress': 0.0,
            'start_time': None,
            'end_time': None
        }
    
    def add_parameter_sweep(self, base_task_id: str, task_func: Callable, 
                           param_name: str, param_values: List[Any],
                           task_args: Tuple = (), task_kwargs: Dict[str, Any] = None):
        """
        Add multiple tasks with different parameter values.
        
        Args:
            base_task_id (str): Base identifier for the tasks
            task_func (callable): Function to execute
            param_name (str): Name of the parameter to sweep
            param_values (list): List of parameter values
            task_args (tuple): Positional arguments for the function
            task_kwargs (dict): Keyword arguments for the function
        """
        if task_kwargs is None:
            task_kwargs = {}
            
        for i, param_value in enumerate(param_values):
            # Create task ID with parameter value
            task_id = f"{base_task_id}_{param_name}_{i}"
            
            # Create kwargs with parameter value
            task_kwargs_copy = task_kwargs.copy()
            task_kwargs_copy[param_name] = param_value
            
            # Add task
            self.add_task(task_id, task_func, task_args, task_kwargs_copy)
    
    def add_grid_search(self, base_task_id: str, task_func: Callable,
                       param_grid: Dict[str, List[Any]],
                       task_args: Tuple = (), task_kwargs: Dict[str, Any] = None):
        """
        Add multiple tasks with different parameter combinations.
        
        Args:
            base_task_id (str): Base identifier for the tasks
            task_func (callable): Function to execute
            param_grid (dict): Dictionary of parameter names and values
            task_args (tuple): Positional arguments for the function
            task_kwargs (dict): Keyword arguments for the function
        """
        if task_kwargs is None:
            task_kwargs = {}
            
        # Get parameter names and values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate all combinations of parameter values
        import itertools
        combinations = list(itertools.product(*param_values))
        
        # Add a task for each combination
        for i, combo in enumerate(combinations):
            # Create task ID with combination index
            task_id = f"{base_task_id}_combo_{i}"
            
            # Create kwargs with parameter values
            task_kwargs_copy = task_kwargs.copy()
            for j, param_name in enumerate(param_names):
                task_kwargs_copy[param_name] = combo[j]
            
            # Add task
            self.add_task(task_id, task_func, task_args, task_kwargs_copy)
    
    def clear_tasks(self):
        """Clear all tasks from the queue."""
        self.task_queue = []
        self.progress = {}
    
    def _task_wrapper(self, task, progress_dict=None):
        """
        Wrapper function for executing a task and tracking progress.
        
        Args:
            task (dict): Task to execute
            progress_dict (dict): Shared dictionary for progress tracking
            
        Returns:
            tuple: (task_id, result, error)
        """
        task_id = task['id']
        
        # Update progress
        if progress_dict is not None:
            progress_dict[task_id] = {
                'status': 'running',
                'progress': 0.0,
                'start_time': time.time(),
                'end_time': None
            }
        
        try:
            # Execute the task
            result = task['func'](*task['args'], **task['kwargs'])
            
            # Update progress
            if progress_dict is not None:
                progress_dict[task_id] = {
                    'status': 'completed',
                    'progress': 1.0,
                    'start_time': progress_dict[task_id]['start_time'],
                    'end_time': time.time()
                }
            
            return task_id, result, None
        except Exception as e:
            # Get traceback
            error = {
                'exception': str(e),
                'traceback': traceback.format_exc()
            }
            
            # Update progress
            if progress_dict is not None:
                progress_dict[task_id] = {
                    'status': 'failed',
                    'progress': 0.0,
                    'start_time': progress_dict[task_id]['start_time'],
                    'end_time': time.time()
                }
            
            return task_id, None, error
    
    def run_tasks(self, store_results=True, result_prefix='task_result_'):
        """
        Run all tasks in parallel.
        
        Args:
            store_results (bool): Whether to store results in the ResultsManager
            result_prefix (str): Prefix for result IDs
            
        Returns:
            dict: Dictionary of results
        """
        if not self.task_queue:
            print("No tasks to run")
            return {}
        
        # Create a manager for sharing progress information
        manager = Manager()
        progress_dict = manager.dict()
        
        # Initialize progress dictionary
        for task in self.task_queue:
            progress_dict[task['id']] = {
                'status': 'queued',
                'progress': 0.0,
                'start_time': None,
                'end_time': None
            }
        
        # Create a pool of workers
        print(f"Running {len(self.task_queue)} tasks with {self.max_workers} workers")
        start_time = time.time()
        
        # Use ProcessPoolExecutor for better exception handling
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._task_wrapper, task, progress_dict): task
                for task in self.task_queue
            }
            
            # Process results as they complete
            if self.use_tqdm:
                # Use tqdm for progress tracking
                with tqdm(total=len(self.task_queue), desc="Running tasks") as pbar:
                    for future in as_completed(future_to_task):
                        task_id, result, error = future.result()
                        
                        # Store result or error
                        if error is None:
                            self.results[task_id] = result
                            
                            # Store result in ResultsManager
                            if store_results:
                                result_id = f"{result_prefix}{task_id}"
                                self.results_manager.store_result(
                                    result_id=result_id,
                                    result_data=result,
                                    metadata={
                                        'task_id': task_id,
                                        'task_args': future_to_task[future]['args'],
                                        'task_kwargs': future_to_task[future]['kwargs'],
                                        'execution_time': progress_dict[task_id]['end_time'] - progress_dict[task_id]['start_time']
                                    }
                                )
                        else:
                            self.errors[task_id] = error
                        
                        # Update progress bar
                        pbar.update(1)
            else:
                # Process results without progress bar
                for future in as_completed(future_to_task):
                    task_id, result, error = future.result()
                    
                    # Store result or error
                    if error is None:
                        self.results[task_id] = result
                        
                        # Store result in ResultsManager
                        if store_results:
                            result_id = f"{result_prefix}{task_id}"
                            self.results_manager.store_result(
                                result_id=result_id,
                                result_data=result,
                                metadata={
                                    'task_id': task_id,
                                    'task_args': future_to_task[future]['args'],
                                    'task_kwargs': future_to_task[future]['kwargs'],
                                    'execution_time': progress_dict[task_id]['end_time'] - progress_dict[task_id]['start_time']
                                }
                            )
                    else:
                        self.errors[task_id] = error
                    
                    # Print progress
                    completed = len(self.results) + len(self.errors)
                    total = len(self.task_queue)
                    print(f"Completed {completed}/{total} tasks ({completed/total:.1%})")
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        
        # Print summary
        print(f"Completed {len(self.results)}/{len(self.task_queue)} tasks in {execution_time:.2f} seconds")
        if self.errors:
            print(f"Failed tasks: {len(self.errors)}")
            for task_id, error in self.errors.items():
                print(f"  {task_id}: {error['exception']}")
        
        # Update progress tracking
        for task_id, progress in progress_dict.items():
            self.progress[task_id] = dict(progress)
        
        return self.results
    
    def get_results(self):
        """
        Get all results.
        
        Returns:
            dict: Dictionary of results
        """
        return self.results
    
    def get_errors(self):
        """
        Get all errors.
        
        Returns:
            dict: Dictionary of errors
        """
        return self.errors
    
    def get_progress(self):
        """
        Get progress information.
        
        Returns:
            dict: Dictionary of progress information
        """
        return self.progress
    
    def get_task_result(self, task_id):
        """
        Get the result for a specific task.
        
        Args:
            task_id (str): Task identifier
            
        Returns:
            Any: Task result
        """
        if task_id not in self.results:
            raise KeyError(f"No result for task '{task_id}'")
        
        return self.results[task_id]
    
    def get_task_error(self, task_id):
        """
        Get the error for a specific task.
        
        Args:
            task_id (str): Task identifier
            
        Returns:
            dict: Task error
        """
        if task_id not in self.errors:
            raise KeyError(f"No error for task '{task_id}'")
        
        return self.errors[task_id]
    
    def get_task_progress(self, task_id):
        """
        Get progress information for a specific task.
        
        Args:
            task_id (str): Task identifier
            
        Returns:
            dict: Task progress information
        """
        if task_id not in self.progress:
            raise KeyError(f"No progress information for task '{task_id}'")
        
        return self.progress[task_id]
    
    def get_execution_times(self):
        """
        Get execution times for all completed tasks.
        
        Returns:
            dict: Dictionary of execution times
        """
        execution_times = {}
        
        for task_id, progress in self.progress.items():
            if progress['status'] == 'completed' and progress['start_time'] is not None and progress['end_time'] is not None:
                execution_times[task_id] = progress['end_time'] - progress['start_time']
        
        return execution_times
    
    def get_average_execution_time(self):
        """
        Get the average execution time for completed tasks.
        
        Returns:
            float: Average execution time
        """
        execution_times = self.get_execution_times()
        
        if not execution_times:
            return 0.0
        
        return sum(execution_times.values()) / len(execution_times)
    
    def get_total_execution_time(self):
        """
        Get the total execution time for all tasks.
        
        Returns:
            float: Total execution time
        """
        execution_times = self.get_execution_times()
        
        if not execution_times:
            return 0.0
        
        return sum(execution_times.values())
    
    def get_speedup(self):
        """
        Get the speedup factor from parallelization.
        
        Returns:
            float: Speedup factor
        """
        total_time = self.get_total_execution_time()
        
        if total_time == 0.0:
            return 0.0
        
        # Get the time from the first task start to the last task end
        start_times = [p['start_time'] for p in self.progress.values() if p['start_time'] is not None]
        end_times = [p['end_time'] for p in self.progress.values() if p['end_time'] is not None]
        
        if not start_times or not end_times:
            return 0.0
        
        parallel_time = max(end_times) - min(start_times)
        
        if parallel_time == 0.0:
            return 0.0
        
        return total_time / parallel_time 