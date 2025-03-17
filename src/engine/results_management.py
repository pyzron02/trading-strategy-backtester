#!/usr/bin/env python3
# results_management.py - In-memory results management system

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Union, Optional

class ResultsManager:
    """
    In-memory results management system for trading strategy backtests.
    
    This class handles result storage, sharing between test modules,
    and selective persistence to disk.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern to ensure only one results manager exists."""
        if cls._instance is None:
            cls._instance = super(ResultsManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, output_dir=None):
        """
        Initialize the ResultsManager.
        
        Args:
            output_dir (str): Directory for saving results to disk
        """
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
            
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        
        # Set output directory
        self.output_dir = output_dir or os.path.join(project_root, 'output')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {}
        
        # Initialize metadata
        self.metadata = {}
        
        # Initialize change tracking
        self.changes = {}
        
        # Initialize timestamp
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self._initialized = True
    
    def store_result(self, result_id: str, result_data: Any, metadata: Optional[Dict[str, Any]] = None,
                    persist: bool = False, persist_format: str = 'pickle'):
        """
        Store a result in memory.
        
        Args:
            result_id (str): Unique identifier for the result
            result_data (Any): Result data to store
            metadata (Dict[str, Any], optional): Metadata for the result
            persist (bool): Whether to persist the result to disk
            persist_format (str): Format for persistence ('pickle', 'json', 'csv')
        """
        # Store the result
        self.results[result_id] = result_data
        
        # Store metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'timestamp': datetime.now().isoformat(),
            'size': self._get_size_estimate(result_data)
        })
        
        self.metadata[result_id] = metadata
        
        # Track change
        self.changes[result_id] = True
        
        # Persist to disk if requested
        if persist:
            self.persist_result(result_id, persist_format)
    
    def get_result(self, result_id: str) -> Any:
        """
        Get a result from memory.
        
        Args:
            result_id (str): Unique identifier for the result
            
        Returns:
            Any: Result data
        """
        if result_id not in self.results:
            raise KeyError(f"Result '{result_id}' not found")
        
        return self.results[result_id]
    
    def get_metadata(self, result_id: str) -> Dict[str, Any]:
        """
        Get metadata for a result.
        
        Args:
            result_id (str): Unique identifier for the result
            
        Returns:
            Dict[str, Any]: Result metadata
        """
        if result_id not in self.metadata:
            raise KeyError(f"Metadata for result '{result_id}' not found")
        
        return self.metadata[result_id]
    
    def has_result(self, result_id: str) -> bool:
        """
        Check if a result exists.
        
        Args:
            result_id (str): Unique identifier for the result
            
        Returns:
            bool: Whether the result exists
        """
        return result_id in self.results
    
    def list_results(self) -> List[str]:
        """
        List all available results.
        
        Returns:
            List[str]: List of result IDs
        """
        return list(self.results.keys())
    
    def delete_result(self, result_id: str):
        """
        Delete a result from memory.
        
        Args:
            result_id (str): Unique identifier for the result
        """
        if result_id in self.results:
            del self.results[result_id]
        
        if result_id in self.metadata:
            del self.metadata[result_id]
        
        if result_id in self.changes:
            del self.changes[result_id]
    
    def clear_results(self):
        """Clear all results from memory."""
        self.results = {}
        self.metadata = {}
        self.changes = {}
    
    def persist_result(self, result_id: str, format: str = 'pickle'):
        """
        Persist a result to disk.
        
        Args:
            result_id (str): Unique identifier for the result
            format (str): Format for persistence ('pickle', 'json', 'csv')
        """
        if result_id not in self.results:
            raise KeyError(f"Result '{result_id}' not found")
        
        # Get result data
        result_data = self.results[result_id]
        
        # Create directory for result
        result_dir = os.path.join(self.output_dir, self.timestamp, result_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Persist based on format
        if format == 'pickle':
            filepath = os.path.join(result_dir, 'result.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(result_data, f)
        elif format == 'json':
            filepath = os.path.join(result_dir, 'result.json')
            try:
                with open(filepath, 'w') as f:
                    json.dump(result_data, f, indent=4, default=self._json_serializer)
            except TypeError:
                print(f"Warning: Could not serialize result '{result_id}' to JSON. Falling back to pickle.")
                filepath = os.path.join(result_dir, 'result.pkl')
                with open(filepath, 'wb') as f:
                    pickle.dump(result_data, f)
        elif format == 'csv':
            filepath = os.path.join(result_dir, 'result.csv')
            try:
                if isinstance(result_data, pd.DataFrame):
                    result_data.to_csv(filepath, index=False)
                else:
                    print(f"Warning: Result '{result_id}' is not a DataFrame. Falling back to pickle.")
                    filepath = os.path.join(result_dir, 'result.pkl')
                    with open(filepath, 'wb') as f:
                        pickle.dump(result_data, f)
            except Exception:
                print(f"Warning: Could not save result '{result_id}' as CSV. Falling back to pickle.")
                filepath = os.path.join(result_dir, 'result.pkl')
                with open(filepath, 'wb') as f:
                    pickle.dump(result_data, f)
        else:
            raise ValueError(f"Unknown persistence format: {format}")
        
        # Save metadata
        metadata_file = os.path.join(result_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata[result_id], f, indent=4, default=str)
        
        # Update metadata with file path
        self.metadata[result_id]['filepath'] = filepath
        self.metadata[result_id]['persisted'] = True
        
        # Reset change flag
        self.changes[result_id] = False
        
        print(f"Persisted result '{result_id}' to {filepath}")
    
    def persist_all_results(self, format: str = 'pickle'):
        """
        Persist all results to disk.
        
        Args:
            format (str): Format for persistence ('pickle', 'json', 'csv')
        """
        for result_id in self.results:
            if result_id in self.changes and self.changes[result_id]:
                self.persist_result(result_id, format)
    
    def load_result(self, result_id: str, filepath: str) -> Any:
        """
        Load a result from disk.
        
        Args:
            result_id (str): Unique identifier for the result
            filepath (str): Path to the result file
            
        Returns:
            Any: Loaded result data
        """
        # Determine format from file extension
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                result_data = pickle.load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                result_data = json.load(f)
        elif filepath.endswith('.csv'):
            result_data = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unknown file format: {filepath}")
        
        # Store the result
        self.results[result_id] = result_data
        
        # Try to load metadata
        metadata_file = os.path.join(os.path.dirname(filepath), 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata[result_id] = json.load(f)
        else:
            self.metadata[result_id] = {
                'timestamp': datetime.now().isoformat(),
                'size': self._get_size_estimate(result_data),
                'filepath': filepath,
                'persisted': True
            }
        
        # Reset change flag
        self.changes[result_id] = False
        
        return result_data
    
    def _get_size_estimate(self, obj: Any) -> str:
        """
        Get a size estimate for an object.
        
        Args:
            obj (Any): Object to estimate size for
            
        Returns:
            str: Size estimate as a string
        """
        try:
            # Try to pickle the object and get its size
            size_bytes = len(pickle.dumps(obj))
            
            # Convert to human-readable format
            if size_bytes < 1024:
                return f"{size_bytes} bytes"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
            else:
                return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
        except:
            return "Unknown"
    
    def _json_serializer(self, obj: Any) -> Any:
        """
        Custom JSON serializer for handling non-serializable objects.
        
        Args:
            obj (Any): Object to serialize
            
        Returns:
            Any: Serializable representation of the object
        """
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def get_result_summary(self, result_id: str) -> Dict[str, Any]:
        """
        Get a summary of a result.
        
        Args:
            result_id (str): Unique identifier for the result
            
        Returns:
            Dict[str, Any]: Result summary
        """
        if result_id not in self.results:
            raise KeyError(f"Result '{result_id}' not found")
        
        result_data = self.results[result_id]
        metadata = self.metadata[result_id]
        
        summary = {
            'id': result_id,
            'type': type(result_data).__name__,
            'timestamp': metadata.get('timestamp', 'Unknown'),
            'size': metadata.get('size', 'Unknown'),
            'persisted': metadata.get('persisted', False)
        }
        
        # Add type-specific information
        if isinstance(result_data, pd.DataFrame):
            summary.update({
                'shape': result_data.shape,
                'columns': result_data.columns.tolist(),
                'sample': result_data.head(5).to_dict(orient='records')
            })
        elif isinstance(result_data, dict):
            summary.update({
                'keys': list(result_data.keys()),
                'sample': {k: result_data[k] for k in list(result_data.keys())[:5]}
            })
        elif isinstance(result_data, list):
            summary.update({
                'length': len(result_data),
                'sample': result_data[:5]
            })
        
        return summary
    
    def get_all_result_summaries(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summaries for all results.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of result summaries
        """
        return {result_id: self.get_result_summary(result_id) for result_id in self.results}
    
    def merge_results(self, result_ids: List[str], merged_id: str, merge_func=None) -> Any:
        """
        Merge multiple results into a single result.
        
        Args:
            result_ids (List[str]): List of result IDs to merge
            merged_id (str): ID for the merged result
            merge_func (callable, optional): Function to merge results
            
        Returns:
            Any: Merged result data
        """
        # Check if all results exist
        for result_id in result_ids:
            if result_id not in self.results:
                raise KeyError(f"Result '{result_id}' not found")
        
        # Get result data
        result_data = [self.results[result_id] for result_id in result_ids]
        
        # Merge results
        if merge_func is not None:
            merged_data = merge_func(result_data)
        elif all(isinstance(data, pd.DataFrame) for data in result_data):
            # Default merge for DataFrames
            merged_data = pd.concat(result_data)
        elif all(isinstance(data, dict) for data in result_data):
            # Default merge for dictionaries
            merged_data = {}
            for data in result_data:
                merged_data.update(data)
        elif all(isinstance(data, list) for data in result_data):
            # Default merge for lists
            merged_data = []
            for data in result_data:
                merged_data.extend(data)
        else:
            raise ValueError("Cannot merge results of different types without a merge function")
        
        # Store merged result
        self.store_result(
            result_id=merged_id,
            result_data=merged_data,
            metadata={
                'merged_from': result_ids,
                'merge_time': datetime.now().isoformat()
            }
        )
        
        return merged_data 