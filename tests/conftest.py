#!/usr/bin/env python3
"""Shared pytest fixtures for the trading strategy backtester."""
import pytest


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singletons before each test for isolation."""
    yield
    # Reset after each test
    try:
        from utils.path_manager import PathManager
        PathManager._reset()
    except Exception:
        pass
    try:
        from engine.data_management import DataManager
        DataManager._reset()
    except Exception:
        pass
    try:
        from engine.parameter_management import ParameterManager
        ParameterManager._reset()
    except Exception:
        pass
    try:
        from engine.logging_system import LoggingSystem
        LoggingSystem._reset()
    except Exception:
        pass
