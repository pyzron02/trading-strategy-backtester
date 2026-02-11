#!/usr/bin/env python3
# serialization.py - Custom JSON serialization for the backtest engine

import json
import datetime
import pandas as pd
import numpy as np


# Custom JSON encoder to handle NumPy types and other non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            # Convert keys to strings to avoid Timestamp key errors
            d = obj.to_dict()
            return {str(k): v for k, v in d.items()}
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return "NaN"
            elif np.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super(CustomJSONEncoder, self).default(obj)
