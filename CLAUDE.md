# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running workflows

- Basic backtest: `python src/utils/run_simple_workflow.py --strategy [StrategyName] --tickers [Tickers]`
- CLI interface: `python src/workflows/cli.py --workflow [type] --strategy [StrategyName] --tickers [Tickers]`
- Using config file: `python src/workflows/cli.py --config [config_file.json]`
- Unified workflow: `python src/workflows/unified_workflow.py [config_file.json]`
- Monte Carlo: `python src/workflows/cli.py --workflow monte_carlo --strategy [StrategyName] --tickers [Tickers]`

Workflow types: `simple`, `optimization`, `monte_carlo`, `walkforward`, `complete`

### Testing commands

- Test all workflows: `bash tests/test_all_workflows.sh`
- Test strategies: `python src/evaluators/test_strategies.py`
- Run strategy tests: `python src/engine/testing/strategy_tester.py --strategy [StrategyName] --tickers [Tickers]`

### Error reporting

- Consolidated error report: `python src/utils/generate_error_report.py --consolidated --days [N]`
- Workflow-specific: `python src/utils/generate_error_report.py --workflow-dir [path]`

## Architecture

### Execution flow

```
CLI (src/workflows/cli.py) or Web (frontend/app.py)
  → unified_workflow.py (routes to appropriate workflow)
    → simple/optimization/monte_carlo/walkforward/complete workflow
      → run_backtest.py (core backtrader engine execution)
        → Strategy class (from registry) generates signals
        → Performance evaluation → Results saved to output/
```

### Strategy system

Strategies inherit from `backtrader.Strategy` and are registered in `src/strategies/registry.py` via `register_strategy(name, class, version)`. Registered names: **SimpleStock**, **MACrossover**, **AuctionMarket**, **MultiPosition**, **PairsTrading**.

To add a new strategy:
1. Create a new file in `src/strategies/`
2. Subclass `backtrader.Strategy` with `params` tuple for parameters
3. Implement `__init__()` (indicators), `next()` (trading logic)
4. Register in `registry.py` with `register_strategy('Name', ClassName)`

### Workflow layer (`src/workflows/`)

- **simple_workflow.py** — Single-parameter-set backtest
- **optimization_workflow.py** — Parameter grid search using Optuna
- **monte_carlo_workflow.py** — Permutation-based statistical validation (returns, block, stationary bootstrap)
- **walkforward_workflow.py** — Sliding window in-sample optimization + out-of-sample testing
- **complete_workflow.py** — Chains optimization → backtest → Monte Carlo → walk-forward
- **unified_workflow.py** — Orchestrator that routes JSON configs to the appropriate workflow

### Core engine (`src/engine/`)

- **run_backtest.py** — Main backtest executor. Loads data via `ValidatingCSVData`, instantiates strategy, runs backtrader, calculates metrics, saves results.
- **data_management.py** — Singleton data loader with caching
- **parameter_management.py** — Singleton for extracting/validating strategy parameters
- **evaluate_performance.py** — Metrics calculation (Sharpe, drawdown, win rate, etc.)
- **results_management.py** — Output file organization
- **logging_system.py** — Async component-based logging (engine, strategies, data, testing, results, cache, parallel)

### Key singletons

Several core classes use the singleton pattern and are accessed globally:
- `PathManager` (`src/utils/path_manager.py`) — Centralized path resolution, auto-detects project root
- `ParameterManager` (`src/engine/parameter_management.py`) — Parameter extraction and validation
- `LoggingSystem` (`src/engine/logging_system.py`) — Async logging with component namespaces
- `DataManager` (`src/engine/data_management.py`) — Data loading with caching

### Configuration

Workflow configs are JSON files in `input/workflow_configs/`. Structure:
```json
{
  "workflow_type": "simple|optimization|monte_carlo|walkforward|complete",
  "common_params": {
    "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD",
    "tickers": ["AAPL"], "initial_capital": 100000.0, "commission": 0.001
  },
  "strategies": {
    "StrategyName": {
      "parameters": {},
      "parameter_grid": {},
      "monte_carlo": {"n_simulations": 100},
      "walkforward": {"window_size": 252}
    }
  }
}
```

Parameter resolution order (highest to lowest priority): CLI args → inline config `parameters` → `param_file` reference → strategy class defaults.

Parameter files live in `input/parameters/`, parameter grids in `input/parameter_grids/`.

### Data

Market data is loaded from `input/stock_data.csv` (multi-ticker CSV with Date, Open, High, Low, Close, Volume columns per ticker). Data can also be fetched via yfinance. The engine uses `ValidatingCSVData` to filter invalid prices.

### Output

Results are saved to `output/[Strategy]_[workflow]_[timestamp]_[hash]/` containing: `equity_curve.csv`, `trades.json`, `metrics.json`, log files, and visualizations (Plotly HTML + static PNG).

### Frontend

Flask web app in `frontend/`. Routes: `/` (config form), `/run-backtest` (execute), `/results` (view). Generates a temporary JSON config from form input and passes it to `unified_workflow`.

## Code Style Guidelines

- **Imports**: Group standard library first, then third-party, then local imports
- **Docstrings**: Use Google style docstrings with Args/Returns sections
- **Error handling**: Use try/except with specific exception types and meaningful error messages
- **Type hints**: Optional but encouraged for function parameters and return values
- **Naming**: snake_case for functions/variables, PascalCase for classes, strategy names must match their registry key
- **Logging**: Use the `logging_system` module, not raw `print()` or `logging` directly
- **Parameters**: Strategy parameters should be passed as dictionaries
- **Testing**: Use the `StrategyTester` class for strategy validation
- **Formatting**: 4-space indentation, 80-character line length limit