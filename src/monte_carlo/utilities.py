#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for the Monte Carlo backtesting framework.
"""

import os
import json
import pandas as pd
import numpy as np

# Import shared JSON encoder
from engine.serialization import CustomJSONEncoder


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