#!/usr/bin/env python3
# logging_system.py - Configurable logging system with performance awareness

import os
import sys
import logging
import time
import json
import threading
import queue
import atexit
from datetime import datetime
from typing import Dict, List, Any, Tuple, Union, Optional, Callable
from functools import wraps

class LoggingSystem:
    """
    Configurable logging system with different verbosity levels and performance awareness.
    
    This class provides a centralized logging system with configurable verbosity,
    performance-aware logging, and structured log format.
    """
    
    _instance = None
    
    # Define log levels
    LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
        'NONE': logging.CRITICAL + 10  # Higher than any standard level
    }
    
    # Define components
    COMPONENTS = [
        'engine',
        'strategies',
        'data',
        'testing',
        'results',
        'cache',
        'parallel'
    ]
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern to ensure only one logging system exists."""
        if cls._instance is None:
            cls._instance = super(LoggingSystem, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_dir=None, default_level='INFO', 
                 console_output=True, file_output=True,
                 async_logging=True, max_queue_size=1000,
                 performance_threshold=0.1):
        """
        Initialize the LoggingSystem.
        
        Args:
            log_dir (str): Directory for log files
            default_level (str): Default log level
            console_output (bool): Whether to output logs to console
            file_output (bool): Whether to output logs to file
            async_logging (bool): Whether to use asynchronous logging
            max_queue_size (int): Maximum size of the log queue
            performance_threshold (float): Threshold in seconds for performance logging
        """
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
            
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
        # Set log directory
        self.log_dir = log_dir or os.path.join(project_root, 'logs')
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set default level
        self.default_level = self.LEVELS.get(default_level.upper(), logging.INFO)
        
        # Set output options
        self.console_output = console_output
        self.file_output = file_output
        
        # Set performance threshold
        self.performance_threshold = performance_threshold
        
        # Set async logging options
        self.async_logging = async_logging
        self.max_queue_size = max_queue_size
        
        # Initialize component loggers
        self.loggers = {}
        
        # Initialize component levels
        self.component_levels = {component: self.default_level for component in self.COMPONENTS}
        
        # Initialize log queue for async logging
        self.log_queue = queue.Queue(maxsize=max_queue_size) if async_logging else None
        self.log_thread = None
        self.stop_event = threading.Event() if async_logging else None
        
        # Initialize performance tracking
        self.performance_tracking = {}
        
        # Set up logging
        self._setup_logging()
        
        # Start async logging thread if enabled
        if self.async_logging:
            self._start_async_logging()
        
        # Register exit handler
        atexit.register(self.shutdown)
        
        self._initialized = True
    
    def _setup_logging(self):
        """Set up logging handlers and formatters."""
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create handlers
        handlers = []
        
        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            handlers.append(console_handler)
        
        if self.file_output:
            # Create a log file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(self.log_dir, f'backtester_{timestamp}.log')
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
        
        # Create loggers for each component
        for component in self.COMPONENTS:
            logger = logging.getLogger(f'backtester.{component}')
            logger.setLevel(self.component_levels[component])
            logger.propagate = False
            
            # Add handlers
            for handler in handlers:
                logger.addHandler(handler)
            
            self.loggers[component] = logger
        
        # Create a root logger
        root_logger = logging.getLogger('backtester')
        root_logger.setLevel(self.default_level)
        root_logger.propagate = False
        
        # Add handlers to root logger
        for handler in handlers:
            root_logger.addHandler(handler)
        
        self.loggers['root'] = root_logger
        
        # Log initialization
        root_logger.info(f"Logging system initialized with default level: {logging.getLevelName(self.default_level)}")
        root_logger.info(f"Log directory: {self.log_dir}")
        root_logger.info(f"Console output: {self.console_output}")
        root_logger.info(f"File output: {self.file_output}")
        root_logger.info(f"Async logging: {self.async_logging}")
    
    def _start_async_logging(self):
        """Start the asynchronous logging thread."""
        if not self.async_logging:
            return
        
        # Define the worker function
        def log_worker():
            while not self.stop_event.is_set() or not self.log_queue.empty():
                try:
                    # Get a log record from the queue with timeout
                    record = self.log_queue.get(timeout=0.1)
                    
                    # Process the record
                    logger_name, level, msg, args, kwargs = record
                    
                    # Get the logger
                    logger = self.get_logger(logger_name)
                    
                    # Log the message
                    logger.log(level, msg, *args, **kwargs)
                    
                    # Mark the task as done
                    self.log_queue.task_done()
                except queue.Empty:
                    # Queue is empty, continue waiting
                    continue
                except Exception as e:
                    # Log the error to stderr
                    sys.stderr.write(f"Error in log worker: {e}\n")
        
        # Create and start the thread
        self.log_thread = threading.Thread(target=log_worker, daemon=True)
        self.log_thread.start()
        
        # Log thread start
        self.get_logger('root').info("Async logging thread started")
    
    def get_logger(self, component='root'):
        """
        Get a logger for a component.
        
        Args:
            component (str): Component name
            
        Returns:
            logging.Logger: Logger for the component
        """
        if component not in self.loggers:
            # Create a new logger for the component
            logger = logging.getLogger(f'backtester.{component}')
            logger.setLevel(self.default_level)
            logger.propagate = False
            
            # Add handlers from root logger
            for handler in self.loggers['root'].handlers:
                logger.addHandler(handler)
            
            self.loggers[component] = logger
        
        return self.loggers[component]
    
    def set_level(self, level, component=None):
        """
        Set the log level for a component or all components.
        
        Args:
            level (str): Log level name
            component (str, optional): Component name
        """
        # Convert level name to level value
        level_value = self.LEVELS.get(level.upper(), self.default_level)
        
        if component is None:
            # Set level for all components
            for comp in self.COMPONENTS:
                self.component_levels[comp] = level_value
                if comp in self.loggers:
                    self.loggers[comp].setLevel(level_value)
            
            # Set level for root logger
            self.loggers['root'].setLevel(level_value)
            self.default_level = level_value
            
            self.loggers['root'].info(f"Set log level for all components to {level}")
        elif component in self.COMPONENTS or component == 'root':
            # Set level for specific component
            self.component_levels[component] = level_value
            if component in self.loggers:
                self.loggers[component].setLevel(level_value)
            
            self.loggers['root'].info(f"Set log level for {component} to {level}")
        else:
            self.loggers['root'].warning(f"Unknown component: {component}")
    
    def log(self, component, level, msg, *args, **kwargs):
        """
        Log a message.
        
        Args:
            component (str): Component name
            level (str): Log level name
            msg (str): Log message
            *args: Arguments for message formatting
            **kwargs: Keyword arguments for logging
        """
        # Convert level name to level value
        level_value = self.LEVELS.get(level.upper(), self.default_level)
        
        # Check if we should log at this level
        if component in self.component_levels and level_value < self.component_levels[component]:
            return
        
        if self.async_logging:
            # Add to queue for async logging
            try:
                self.log_queue.put((component, level_value, msg, args, kwargs), block=False)
            except queue.Full:
                # Queue is full, log directly
                self.get_logger(component).log(level_value, msg, *args, **kwargs)
        else:
            # Log directly
            self.get_logger(component).log(level_value, msg, *args, **kwargs)
    
    def debug(self, component, msg, *args, **kwargs):
        """Log a debug message."""
        self.log(component, 'DEBUG', msg, *args, **kwargs)
    
    def info(self, component, msg, *args, **kwargs):
        """Log an info message."""
        self.log(component, 'INFO', msg, *args, **kwargs)
    
    def warning(self, component, msg, *args, **kwargs):
        """Log a warning message."""
        self.log(component, 'WARNING', msg, *args, **kwargs)
    
    def error(self, component, msg, *args, **kwargs):
        """Log an error message."""
        self.log(component, 'ERROR', msg, *args, **kwargs)
    
    def critical(self, component, msg, *args, **kwargs):
        """Log a critical message."""
        self.log(component, 'CRITICAL', msg, *args, **kwargs)
    
    def performance_start(self, component, operation):
        """
        Start tracking performance for an operation.
        
        Args:
            component (str): Component name
            operation (str): Operation name
        """
        key = f"{component}:{operation}"
        self.performance_tracking[key] = time.time()
    
    def performance_end(self, component, operation, log_level='DEBUG'):
        """
        End tracking performance for an operation and log if above threshold.
        
        Args:
            component (str): Component name
            operation (str): Operation name
            log_level (str): Log level to use if above threshold
        """
        key = f"{component}:{operation}"
        if key not in self.performance_tracking:
            return
        
        start_time = self.performance_tracking[key]
        end_time = time.time()
        duration = end_time - start_time
        
        # Remove from tracking
        del self.performance_tracking[key]
        
        # Log if above threshold
        if duration >= self.performance_threshold:
            self.log(component, log_level, f"Performance: {operation} took {duration:.4f} seconds")
    
    def shutdown(self):
        """Shutdown the logging system."""
        if self.async_logging:
            # Signal the thread to stop
            self.stop_event.set()
            
            # Wait for the queue to be empty
            if self.log_queue is not None:
                try:
                    self.log_queue.join(timeout=1.0)
                except:
                    pass
            
            # Wait for the thread to stop
            if self.log_thread is not None and self.log_thread.is_alive():
                self.log_thread.join(timeout=1.0)
        
        # Close all handlers
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)

# Create a singleton instance
logger = LoggingSystem()

def log_execution_time(component, level='DEBUG', threshold=None):
    """
    Decorator to log the execution time of a function.
    
    Args:
        component (str): Component name
        level (str): Log level name
        threshold (float, optional): Threshold in seconds
        
    Returns:
        function: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Start timing
            start_time = time.time()
            
            # Call the function
            result = func(*args, **kwargs)
            
            # End timing
            end_time = time.time()
            duration = end_time - start_time
            
            # Log if above threshold
            if threshold is None or duration >= threshold:
                logger.log(component, level, f"Function {func.__name__} took {duration:.4f} seconds")
            
            return result
        
        return wrapper
    
    return decorator

def disable_logging(func):
    """
    Decorator to disable logging during function execution.
    
    Args:
        func: Function to decorate
        
    Returns:
        function: Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save current levels
        saved_levels = {comp: logger.component_levels[comp] for comp in logger.COMPONENTS}
        saved_default = logger.default_level
        
        # Set all levels to NONE
        logger.set_level('NONE')
        
        try:
            # Call the function
            return func(*args, **kwargs)
        finally:
            # Restore levels
            for comp, level in saved_levels.items():
                logger.component_levels[comp] = level
                if comp in logger.loggers:
                    logger.loggers[comp].setLevel(level)
            
            logger.default_level = saved_default
            logger.loggers['root'].setLevel(saved_default)
    
    return wrapper 