#!/usr/bin/env python3
# smart_cache.py - Smart caching system with dependency tracking

import os
import sys
import pickle
import hashlib
import inspect
import time
import json
from datetime import datetime
from functools import wraps
from typing import Dict, List, Any, Tuple, Union, Optional, Callable, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('smart_cache')

class SmartCache:
    """
    Smart caching system with hash-based result caching and dependency tracking.
    
    This class provides a caching mechanism for expensive computations,
    with automatic dependency tracking and invalidation.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern to ensure only one cache exists."""
        if cls._instance is None:
            cls._instance = super(SmartCache, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, cache_dir=None, max_memory_items=1000, verbose=False):
        """
        Initialize the SmartCache.
        
        Args:
            cache_dir (str): Directory for disk cache
            max_memory_items (int): Maximum number of items to keep in memory
            verbose (bool): Whether to print verbose output
        """
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
            
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        
        # Set cache directory
        self.cache_dir = cache_dir or os.path.join(project_root, 'cache')
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set maximum number of items to keep in memory
        self.max_memory_items = max_memory_items
        
        # Set verbosity
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Initialize memory cache
        self.memory_cache = {}
        
        # Initialize access timestamps
        self.access_times = {}
        
        # Initialize dependency tracking
        self.dependencies = {}
        self.dependents = {}
        
        # Initialize statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'invalidations': 0,
            'evictions': 0
        }
        
        self._initialized = True
        
        logger.debug("SmartCache initialized")
    
    def _get_cache_key(self, func: Callable, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """
        Generate a cache key for a function call.
        
        Args:
            func (callable): Function being called
            args (tuple): Positional arguments
            kwargs (dict): Keyword arguments
            
        Returns:
            str: Cache key
        """
        # Get function name and module
        func_name = func.__name__
        module_name = func.__module__
        
        # Convert args and kwargs to a string representation
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        
        # Create a string representation of the function call
        call_str = f"{module_name}.{func_name}({args_str}, {kwargs_str})"
        
        # Generate MD5 hash as cache key
        return hashlib.md5(call_str.encode()).hexdigest()
    
    def _get_disk_cache_path(self, cache_key: str) -> str:
        """
        Get the path to a disk cache file.
        
        Args:
            cache_key (str): Cache key
            
        Returns:
            str: Path to disk cache file
        """
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _get_metadata_path(self, cache_key: str) -> str:
        """
        Get the path to a metadata file.
        
        Args:
            cache_key (str): Cache key
            
        Returns:
            str: Path to metadata file
        """
        return os.path.join(self.cache_dir, f"{cache_key}.meta.json")
    
    def _save_to_disk(self, cache_key: str, result: Any, metadata: Dict[str, Any]):
        """
        Save a result to disk.
        
        Args:
            cache_key (str): Cache key
            result (Any): Result to save
            metadata (Dict[str, Any]): Metadata for the result
        """
        # Save result
        cache_path = self._get_disk_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.error(f"Error saving result to disk: {e}")
            return
        
        # Save metadata
        metadata_path = self._get_metadata_path(cache_key)
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metadata to disk: {e}")
            # Remove result file if metadata couldn't be saved
            if os.path.exists(cache_path):
                os.remove(cache_path)
    
    def _load_from_disk(self, cache_key: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a result from disk.
        
        Args:
            cache_key (str): Cache key
            
        Returns:
            tuple: (result, metadata)
        """
        # Check if result exists
        cache_path = self._get_disk_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return None, None
        
        # Load result
        try:
            with open(cache_path, 'rb') as f:
                result = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading result from disk: {e}")
            return None, None
        
        # Load metadata
        metadata_path = self._get_metadata_path(cache_key)
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata from disk: {e}")
                metadata = {}
        else:
            metadata = {}
        
        return result, metadata
    
    def _evict_if_needed(self):
        """Evict items from memory cache if it's too large."""
        if len(self.memory_cache) <= self.max_memory_items:
            return
        
        # Sort items by access time
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        
        # Evict oldest items
        num_to_evict = len(self.memory_cache) - self.max_memory_items
        for i in range(num_to_evict):
            cache_key = sorted_items[i][0]
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
                del self.access_times[cache_key]
                self.stats['evictions'] += 1
                logger.debug(f"Evicted item {cache_key} from memory cache")
    
    def _add_dependency(self, dependent_key: str, dependency_key: str):
        """
        Add a dependency relationship between two cache keys.
        
        Args:
            dependent_key (str): Key that depends on the other
            dependency_key (str): Key that is depended upon
        """
        # Add to dependencies
        if dependent_key not in self.dependencies:
            self.dependencies[dependent_key] = set()
        self.dependencies[dependent_key].add(dependency_key)
        
        # Add to dependents
        if dependency_key not in self.dependents:
            self.dependents[dependency_key] = set()
        self.dependents[dependency_key].add(dependent_key)
        
        logger.debug(f"Added dependency: {dependent_key} depends on {dependency_key}")
    
    def _invalidate_dependents(self, cache_key: str, visited: Optional[Set[str]] = None):
        """
        Invalidate all cache items that depend on a key.
        
        Args:
            cache_key (str): Key to invalidate dependents for
            visited (Set[str], optional): Set of already visited keys
        """
        if visited is None:
            visited = set()
        
        # Skip if already visited
        if cache_key in visited:
            return
        
        # Mark as visited
        visited.add(cache_key)
        
        # Get dependents
        dependents = self.dependents.get(cache_key, set())
        
        # Invalidate each dependent
        for dependent_key in dependents:
            # Remove from memory cache
            if dependent_key in self.memory_cache:
                del self.memory_cache[dependent_key]
                logger.debug(f"Invalidated dependent {dependent_key} from memory cache")
            
            # Remove from access times
            if dependent_key in self.access_times:
                del self.access_times[dependent_key]
            
            # Remove disk cache
            cache_path = self._get_disk_cache_path(dependent_key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.debug(f"Invalidated dependent {dependent_key} from disk cache")
            
            # Remove metadata
            metadata_path = self._get_metadata_path(dependent_key)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # Increment invalidation count
            self.stats['invalidations'] += 1
            
            # Recursively invalidate dependents
            self._invalidate_dependents(dependent_key, visited)
    
    def get(self, cache_key: str) -> Tuple[Any, bool]:
        """
        Get a result from the cache.
        
        Args:
            cache_key (str): Cache key
            
        Returns:
            tuple: (result, found)
        """
        # Check memory cache
        if cache_key in self.memory_cache:
            # Update access time
            self.access_times[cache_key] = time.time()
            
            # Update stats
            self.stats['hits'] += 1
            self.stats['memory_hits'] += 1
            
            logger.debug(f"Memory cache hit for {cache_key}")
            
            return self.memory_cache[cache_key], True
        
        # Check disk cache
        result, metadata = self._load_from_disk(cache_key)
        if result is not None:
            # Add to memory cache
            self.memory_cache[cache_key] = result
            
            # Update access time
            self.access_times[cache_key] = time.time()
            
            # Evict if needed
            self._evict_if_needed()
            
            # Update stats
            self.stats['hits'] += 1
            self.stats['disk_hits'] += 1
            
            logger.debug(f"Disk cache hit for {cache_key}")
            
            return result, True
        
        # Update stats
        self.stats['misses'] += 1
        
        logger.debug(f"Cache miss for {cache_key}")
        
        return None, False
    
    def put(self, cache_key: str, result: Any, metadata: Optional[Dict[str, Any]] = None,
           dependencies: Optional[List[str]] = None):
        """
        Put a result in the cache.
        
        Args:
            cache_key (str): Cache key
            result (Any): Result to cache
            metadata (Dict[str, Any], optional): Metadata for the result
            dependencies (List[str], optional): List of cache keys this result depends on
        """
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add timestamp to metadata
        metadata['timestamp'] = datetime.now().isoformat()
        
        # Add to memory cache
        self.memory_cache[cache_key] = result
        
        # Update access time
        self.access_times[cache_key] = time.time()
        
        # Evict if needed
        self._evict_if_needed()
        
        # Save to disk
        self._save_to_disk(cache_key, result, metadata)
        
        # Add dependencies
        if dependencies:
            for dependency_key in dependencies:
                self._add_dependency(cache_key, dependency_key)
        
        logger.debug(f"Added {cache_key} to cache")
    
    def invalidate(self, cache_key: str, invalidate_dependents: bool = True):
        """
        Invalidate a cache item.
        
        Args:
            cache_key (str): Cache key to invalidate
            invalidate_dependents (bool): Whether to invalidate dependents
        """
        # Remove from memory cache
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
            logger.debug(f"Invalidated {cache_key} from memory cache")
        
        # Remove from access times
        if cache_key in self.access_times:
            del self.access_times[cache_key]
        
        # Remove disk cache
        cache_path = self._get_disk_cache_path(cache_key)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.debug(f"Invalidated {cache_key} from disk cache")
        
        # Remove metadata
        metadata_path = self._get_metadata_path(cache_key)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        # Increment invalidation count
        self.stats['invalidations'] += 1
        
        # Invalidate dependents
        if invalidate_dependents:
            self._invalidate_dependents(cache_key)
    
    def clear(self):
        """Clear the entire cache."""
        # Clear memory cache
        self.memory_cache = {}
        
        # Clear access times
        self.access_times = {}
        
        # Clear dependencies
        self.dependencies = {}
        self.dependents = {}
        
        # Clear disk cache
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Reset stats
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'invalidations': 0,
            'evictions': 0
        }
        
        logger.debug("Cache cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, int]: Cache statistics
        """
        return self.stats
    
    def get_size(self) -> Dict[str, int]:
        """
        Get cache size information.
        
        Returns:
            Dict[str, int]: Cache size information
        """
        # Count disk cache files
        disk_count = 0
        disk_size = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                disk_count += 1
                file_path = os.path.join(self.cache_dir, filename)
                disk_size += os.path.getsize(file_path)
        
        return {
            'memory_items': len(self.memory_cache),
            'disk_items': disk_count,
            'disk_size_bytes': disk_size
        }

# Create a singleton instance
cache = SmartCache()

def cached(dependencies=None, ttl=None):
    """
    Decorator for caching function results.
    
    Args:
        dependencies (List[str], optional): List of function names this function depends on
        ttl (int, optional): Time-to-live in seconds
        
    Returns:
        function: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache._get_cache_key(func, args, kwargs)
            
            # Check if result is in cache
            result, found = cache.get(cache_key)
            if found:
                # Check if result is expired
                if ttl is not None:
                    metadata_path = cache._get_metadata_path(cache_key)
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            
                            # Check timestamp
                            timestamp = datetime.fromisoformat(metadata['timestamp'])
                            age = (datetime.now() - timestamp).total_seconds()
                            
                            if age > ttl:
                                # Result is expired
                                found = False
                                logger.debug(f"Result for {cache_key} is expired")
                        except Exception as e:
                            logger.error(f"Error checking TTL: {e}")
                
                if found:
                    return result
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Generate dependency keys
            dependency_keys = []
            if dependencies:
                for dep_func_name in dependencies:
                    # Find the function in the module
                    module = inspect.getmodule(func)
                    if hasattr(module, dep_func_name):
                        dep_func = getattr(module, dep_func_name)
                        # Add all cache keys for this function
                        for dep_key in [k for k in cache.memory_cache.keys() if k.startswith(dep_func.__name__)]:
                            dependency_keys.append(dep_key)
            
            # Cache the result
            cache.put(cache_key, result, dependencies=dependency_keys)
            
            return result
        
        return wrapper
    
    return decorator 