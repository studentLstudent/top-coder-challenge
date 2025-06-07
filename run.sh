#!/usr/bin/env bash
set -euo pipefail

days=$1
miles=$2
receipts=$3

# Test the simplest formula: just add everything with standard rates
# Per diem: $86/day (from analysis)
# Mileage: $0.55/mile (standard rate mentioned in interviews)
# Receipts: Added directly

total=$(echo "86 * $days + 0.55 * $miles + $receipts" | bc -l)

# Output with 2 decimal places
printf "%.2f\n" "$total"