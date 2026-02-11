#!/bin/bash

# Change to the project root directory
cd ~/trading-strategy-backtester

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
total=0
passed=0
failed=0

echo "=================================="
echo "Testing All Workflow Configurations"
echo "=================================="
echo ""

# Loop through config files
for config in input/workflow_configs/*.json; do
    total=$((total + 1))
    config_name=$(basename "$config")
    
    echo -n "Testing $config_name... "
    
    # Run the workflow with timeout and capture output
    if timeout 300 python3 src/workflows/cli.py --config "$config" > /tmp/workflow_test.log 2>&1; then
        # Check if the log contains success indicators
        if grep -q "Workflow completed successfully" /tmp/workflow_test.log || grep -q "Results saved to" /tmp/workflow_test.log; then
            echo -e "${GREEN}✓ PASSED${NC}"
            passed=$((passed + 1))
        else
            echo -e "${YELLOW}⚠ COMPLETED BUT NO SUCCESS MESSAGE${NC}"
            echo "  Check /tmp/workflow_test.log for details"
            passed=$((passed + 1))
        fi
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo -e "${RED}✗ FAILED (TIMEOUT)${NC}"
        else
            echo -e "${RED}✗ FAILED (EXIT CODE: $exit_code)${NC}"
        fi
        failed=$((failed + 1))
        
        # Show last few lines of error
        echo "  Error details:"
        tail -10 /tmp/workflow_test.log | sed 's/^/    /'
    fi
    
    echo ""
done

# Summary
echo "=================================="
echo "Test Summary"
echo "=================================="
echo "Total: $total"
echo -e "Passed: ${GREEN}$passed${NC}"
echo -e "Failed: ${RED}$failed${NC}"

if [ $failed -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed!${NC}"
    exit 1
fi