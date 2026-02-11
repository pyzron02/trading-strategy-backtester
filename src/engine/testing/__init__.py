#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing module for the backtester.

This module contains the testing suite for the backtester.
"""
# Import test modules using relative imports
from .walk_forward_test import WalkForwardTest

# Define public exports
__all__ = [
    'WalkForwardTest'
] 