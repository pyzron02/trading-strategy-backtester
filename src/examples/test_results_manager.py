#!/usr/bin/env python3
# test_results_manager.py - Test script for the ResultsManager class

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from engine.results_management import ResultsManager

def main():
    """Test the ResultsManager class."""
    print("\n" + "="*80)
    print("ResultsManager Test")
    print("="*80 + "\n")
    
    # Create ResultsManager instance
    results_manager = ResultsManager()
    
    # Test storing and retrieving results
    print("Testing storing and retrieving results:")
    
    # Create some test results
    dict_result = {
        'strategy': 'TestStrategy',
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.2,
        'total_return': 0.35,
        'win_rate': 0.65
    }
    
    df_result = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=10),
        'price': np.random.normal(100, 10, 10),
        'returns': np.random.normal(0.001, 0.01, 10)
    })
    
    list_result = [
        {'trade_id': 1, 'entry_price': 100, 'exit_price': 105, 'profit': 5},
        {'trade_id': 2, 'entry_price': 105, 'exit_price': 102, 'profit': -3},
        {'trade_id': 3, 'entry_price': 102, 'exit_price': 110, 'profit': 8}
    ]
    
    # Store results
    results_manager.store_result('dict_result', dict_result, {'description': 'Test dictionary result'})
    results_manager.store_result('df_result', df_result, {'description': 'Test DataFrame result'})
    results_manager.store_result('list_result', list_result, {'description': 'Test list result'})
    
    # List results
    result_ids = results_manager.list_results()
    print(f"Stored results: {result_ids}")
    
    # Get results
    retrieved_dict = results_manager.get_result('dict_result')
    retrieved_df = results_manager.get_result('df_result')
    retrieved_list = results_manager.get_result('list_result')
    
    print(f"Retrieved dictionary result: {retrieved_dict}")
    print(f"Retrieved DataFrame result shape: {retrieved_df.shape}")
    print(f"Retrieved list result length: {len(retrieved_list)}")
    
    # Get metadata
    dict_metadata = results_manager.get_metadata('dict_result')
    print(f"Dictionary result metadata: {dict_metadata}")
    
    # Test result summaries
    print("\nTesting result summaries:")
    
    # Get summary for a result
    dict_summary = results_manager.get_result_summary('dict_result')
    print(f"Dictionary result summary: {dict_summary}")
    
    # Get all summaries
    all_summaries = results_manager.get_all_result_summaries()
    print(f"All result summaries: {len(all_summaries)} summaries")
    
    # Test persistence
    print("\nTesting result persistence:")
    
    # Persist a result
    results_manager.persist_result('dict_result', 'json')
    results_manager.persist_result('df_result', 'csv')
    results_manager.persist_result('list_result', 'pickle')
    
    # Check if results are marked as persisted
    dict_metadata = results_manager.get_metadata('dict_result')
    df_metadata = results_manager.get_metadata('df_result')
    list_metadata = results_manager.get_metadata('list_result')
    
    print(f"Dictionary result persisted: {dict_metadata.get('persisted', False)}")
    print(f"DataFrame result persisted: {df_metadata.get('persisted', False)}")
    print(f"List result persisted: {list_metadata.get('persisted', False)}")
    
    # Test result merging
    print("\nTesting result merging:")
    
    # Create additional list result
    list_result2 = [
        {'trade_id': 4, 'entry_price': 110, 'exit_price': 115, 'profit': 5},
        {'trade_id': 5, 'entry_price': 115, 'exit_price': 112, 'profit': -3}
    ]
    
    # Store additional result
    results_manager.store_result('list_result2', list_result2, {'description': 'Additional test list result'})
    
    # Merge list results
    merged_list = results_manager.merge_results(['list_result', 'list_result2'], 'merged_list_result')
    
    print(f"Merged list result length: {len(merged_list)}")
    print(f"Merged list result: {merged_list}")
    
    # Create additional DataFrame result
    df_result2 = pd.DataFrame({
        'date': pd.date_range(start='2020-01-11', periods=5),
        'price': np.random.normal(110, 10, 5),
        'returns': np.random.normal(0.002, 0.01, 5)
    })
    
    # Store additional DataFrame result
    results_manager.store_result('df_result2', df_result2, {'description': 'Additional test DataFrame result'})
    
    # Merge DataFrame results
    merged_df = results_manager.merge_results(['df_result', 'df_result2'], 'merged_df_result')
    
    print(f"Merged DataFrame result shape: {merged_df.shape}")
    print(f"Merged DataFrame result:\n{merged_df.head()}")
    
    # Test clearing results
    print("\nTesting clearing results:")
    
    # Delete a result
    results_manager.delete_result('list_result2')
    remaining_results = results_manager.list_results()
    print(f"Results after deletion: {remaining_results}")
    
    # Clear all results
    results_manager.clear_results()
    remaining_results = results_manager.list_results()
    print(f"Results after clearing: {remaining_results}")
    
    print("\n" + "="*80)
    print("ResultsManager Test Complete")
    print("="*80 + "\n")

if __name__ == '__main__':
    main() 