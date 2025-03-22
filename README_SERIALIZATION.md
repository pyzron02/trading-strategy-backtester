# Serialization Improvements in the Trading Strategy Backtester

## Overview
This branch (`fix-serialization-issues`) addresses several key serialization challenges in the backtesting framework, ensuring robust storage and retrieval of backtest results.

## Implemented Changes

### 1. Custom JSON Encoder
- Added a `CustomJSONEncoder` class to handle various data types:
  - NumPy integers and floating point numbers
  - NumPy arrays
  - Pandas Timestamps
  - Datetime objects
  - NaN/None values

### 2. Improved Results Storage
- Implemented dual storage formats:
  - JSON format for human readability and portability
  - Pickle format for full object preservation
- Created serializable dictionaries instead of trying to directly pickle complex objects

### 3. Robust Error Handling
- Added proper exception handling for serialization errors
- Implemented fallback mechanisms when certain objects can't be serialized

### 4. Strategy Improvements
- Enhanced MA Crossover strategy with updated parameters
- Added SafeSMA indicator for improved stability
- Added NumPy dependency to strategy files for better calculations

## Usage
The serialization improvements should be transparent to users. Backtest results are now saved in both formats:

```python
# JSON results
backtest_results.json

# Pickle results (for programmatic access)
backtest_results.pkl
```

## Testing
The improvements have been tested with multiple strategies:
- MACrossover
- SimpleStock
- AuctionMarket

All tests show successful serialization and retrieval of results. 