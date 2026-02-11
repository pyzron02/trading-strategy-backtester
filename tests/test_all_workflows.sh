#!/bin/bash
# Test all workflow configurations

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Project root is one level up
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Loop through config files
for config in input/workflow_configs/*.json; do
    echo "Running workflow for $config"
    python3 src/workflows/cli.py --config "$config"
done
