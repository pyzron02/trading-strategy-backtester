#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for the Monte Carlo backtesting framework.
"""

import os
import json
import pandas as pd
import numpy as np


# Custom JSON encoder to handle pandas Series and other non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'to_json'):
            return obj.to_json()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def save_to_json(data, filepath):
    """Save data to a JSON file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=CustomJSONEncoder)
        return True
    except Exception as e:
        print(f"Error saving JSON data: {e}")
        return False 