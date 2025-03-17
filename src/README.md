# Trading Strategy Backtester - Code Organization

This document provides an overview of the codebase organization for the trading strategy backtester system.

## Directory Structure

- `/src`: Main source code directory
  - `/engine`: Core backtesting engine components
  - `/strategies`: Trading strategy implementations
  - `/runners`: Single-purpose script runners for specific tasks
  - `/workflows`: High-level workflow orchestrators
  - `/optimizers`: Parameter optimization tools
  - `/evaluators`: Performance evaluation utilities
  - `/utils`: General utility functions
  - `/data_preprocessing`: Tools for data preparation and cleaning
  - `/examples`: Example usage and demonstration scripts
  - `/output`: Default location for output files (created at runtime)

## Main Components

### Engine Components

The `/engine` directory contains the core components of the backtesting system:

- `run_backtest.py`: Core function to run a single backtest
- `data_management.py`: Data loading and management
- `parameter_management.py`: Parameter handling and optimization
- `/testing`: Testing frameworks including walk-forward test

### Runners

The `/runners` directory contains standalone scripts for running specific tasks:

- `run_simple_backtest.py`: Run a single backtest for any strategy
  - Provides `run_strategy_backtest()` function
  - Includes specialized wrapper for MA Crossover strategy
  
- `run_walk_forward_test.py`: Run a walk-forward test for any strategy
  - Provides `run_walk_forward_test()` function

### Workflows

The `/workflows` directory contains high-level orchestrators for complete workflows:

- `unified_workflow.py`: Central entry point for all workflows
  - Provides `run_simple_workflow()`, `run_complete_workflow()` functions
  - Command-line interface with `--workflow-type` argument

## Entry Points

### For Simple Backtesting

```bash
python src/runners/run_simple_backtest.py --strategy MACrossover --tickers AAPL --start-date 2020-01-01 --end-date 2021-12-31
```

### For Walk-Forward Testing

```bash
python src/runners/run_walk_forward_test.py --strategy MACrossover --tickers AAPL
```

### For Complete Workflows

```bash
python src/workflows/unified_workflow.py --workflow-type complete --strategy MACrossover --tickers AAPL
```

## Code Organization Principles

1. **Single Responsibility**: Each module and class has a clear, focused purpose
2. **Common Utilities**: Shared functionality is extracted into utility modules
3. **Consistent Interfaces**: Components interact through well-defined interfaces
4. **Parameter Management**: All strategy parameters are handled consistently
5. **Logging and Reporting**: Standardized logging and reporting across components

## Development Guidelines

When extending the system:

1. Add new strategies to the `/strategies` directory
2. Use the existing parameter management system for strategy parameters
3. Leverage the unified workflow for complex testing scenarios
4. Follow the established patterns for output directory structure
5. Maintain backward compatibility when modifying core components 