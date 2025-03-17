#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing module for the backtester.

This module contains the testing suite for the backtester.
"""

import os
import sys

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import test modules
from engine.testing.walk_forward_test import WalkForwardTest

# Define public exports
__all__ = [
    'WalkForwardTest'
] 