#!/usr/bin/env python3
import os
import sys
import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the backtest function
from src.engine.run_backtest import run_backtest

def main():
    """
    Run a custom optimization for the PairsTrading strategy.
    """
    print("Starting custom optimization for PairsTrading strategy")
    
    # Strategy parameters to try
    parameter_combinations = [
        {"lookback_period": 30, "entry_threshold": 1.5, "exit_threshold": 0.3, "rebalance_freq": 10, "position_size": 100, "stop_loss": 0.03},
        {"lookback_period": 30, "entry_threshold": 2.0, "exit_threshold": 0.5, "rebalance_freq": 20, "position_size": 100, "stop_loss": 0.05},
        {"lookback_period": 45, "entry_threshold": 2.0, "exit_threshold": 0.5, "rebalance_freq": 20, "position_size": 100, "stop_loss": 0.05},
        {"lookback_period": 60, "entry_threshold": 2.0, "exit_threshold": 0.5, "rebalance_freq": 20, "position_size": 100, "stop_loss": 0.05},
        {"lookback_period": 75, "entry_threshold": 2.0, "exit_threshold": 0.5, "rebalance_freq": 20, "position_size": 100, "stop_loss": 0.05},
        {"lookback_period": 90, "entry_threshold": 2.0, "exit_threshold": 0.5, "rebalance_freq": 20, "position_size": 100, "stop_loss": 0.05},
        {"lookback_period": 60, "entry_threshold": 1.5, "exit_threshold": 0.5, "rebalance_freq": 20, "position_size": 100, "stop_loss": 0.05},
        {"lookback_period": 60, "entry_threshold": 2.5, "exit_threshold": 0.5, "rebalance_freq": 20, "position_size": 100, "stop_loss": 0.05},
        {"lookback_period": 60, "entry_threshold": 3.0, "exit_threshold": 0.5, "rebalance_freq": 20, "position_size": 100, "stop_loss": 0.05},
        {"lookback_period": 60, "entry_threshold": 2.0, "exit_threshold": 0.3, "rebalance_freq": 20, "position_size": 100, "stop_loss": 0.05},
        {"lookback_period": 60, "entry_threshold": 2.0, "exit_threshold": 0.7, "rebalance_freq": 20, "position_size": 100, "stop_loss": 0.05},
        {"lookback_period": 60, "entry_threshold": 2.0, "exit_threshold": 1.0, "rebalance_freq": 20, "position_size": 100, "stop_loss": 0.05},
        {"lookback_period": 60, "entry_threshold": 2.0, "exit_threshold": 0.5, "rebalance_freq": 10, "position_size": 100, "stop_loss": 0.05},
        {"lookback_period": 60, "entry_threshold": 2.0, "exit_threshold": 0.5, "rebalance_freq": 30, "position_size": 100, "stop_loss": 0.05},
        {"lookback_period": 60, "entry_threshold": 2.0, "exit_threshold": 0.5, "rebalance_freq": 20, "position_size": 100, "stop_loss": 0.03},
        {"lookback_period": 60, "entry_threshold": 2.0, "exit_threshold": 0.5, "rebalance_freq": 20, "position_size": 100, "stop_loss": 0.07}
    ]
    
    # Common parameters
    common_params = {
        "strategy_name": "PairsTrading",
        "tickers": ["AAPL", "MSFT"],
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "initial_capital": 100000.0,
        "commission": 0.001,
        "plot": False,
        "verbose": False
    }
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"PairsTrading_custom_optimization_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run backtests for each parameter combination
    results = []
    
    for i, params in enumerate(tqdm(parameter_combinations)):
        # Create a subdirectory for this run
        run_output_dir = os.path.join(output_dir, f"run_{i:02d}")
        os.makedirs(run_output_dir, exist_ok=True)
        
        # Run backtest
        print(f"\nRunning backtest {i+1}/{len(parameter_combinations)} with parameters: {params}")
        backtest_result = run_backtest(
            **common_params,
            parameters=params,
            output_dir=run_output_dir
        )
        
        if backtest_result:
            # Extract key metrics
            metrics = backtest_result.get("metrics", {})
            
            result = {
                "run_id": i,
                "parameters": params,
                "initial_value": metrics.get("initial_value", 0),
                "final_value": metrics.get("final_value", 0),
                "total_return": metrics.get("total_return", 0),
                "total_return_pct": metrics.get("total_return_pct", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
                "win_rate": metrics.get("win_rate", 0),
                "win_rate_pct": metrics.get("win_rate_pct", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "avg_trade_pnl": metrics.get("avg_trade_pnl", 0),
                "total_trades": metrics.get("total_trades", 0)
            }
            
            results.append(result)
            print(f"Run {i} completed with sharpe_ratio: {result['sharpe_ratio']:.4f}, total_return: {result['total_return_pct']:.2f}%")
        else:
            print(f"Run {i} failed")
    
    # Convert results to a DataFrame for analysis
    if results:
        results_df = pd.DataFrame(results)
        
        # Sort by Sharpe ratio (descending)
        sorted_df = results_df.sort_values(by=["sharpe_ratio"], ascending=False)
        
        # Save the results to CSV
        csv_path = os.path.join(output_dir, "optimization_results.csv")
        sorted_df.to_csv(csv_path, index=False)
        
        # Save the best parameters
        best_params = sorted_df.iloc[0]["parameters"]
        best_params_path = os.path.join(output_dir, "best_parameters.json")
        with open(best_params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        
        # Print the top 5 parameter sets
        print("\nTop 5 Parameter Sets (by Sharpe Ratio):")
        for i, row in sorted_df.head(5).iterrows():
            params = row["parameters"]
            print(f"Rank {i+1}: Sharpe={row['sharpe_ratio']:.4f}, Return={row['total_return_pct']:.2f}%, "
                  f"MaxDD={row['max_drawdown_pct']:.2f}%, Win Rate={row['win_rate_pct']:.2f}%")
            print(f"  Parameters: lookback={params['lookback_period']}, entry={params['entry_threshold']}, "
                  f"exit={params['exit_threshold']}, rebalance={params['rebalance_freq']}, "
                  f"stop_loss={params['stop_loss']}")
        
        print(f"\nResults saved to {csv_path}")
        print(f"Best parameters saved to {best_params_path}")
    else:
        print("No successful runs")

if __name__ == "__main__":
    main()