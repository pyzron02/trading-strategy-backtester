# Change to the project root directory first
cd ~/trading-strategy-backtester

# Loop through config files
for config in input/workflow_configs/*.json; do
    echo "Running workflow for $config"
    python3 src/workflows/cli.py --config "$config"
done