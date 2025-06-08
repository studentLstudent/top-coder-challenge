#!/usr/bin/env bash
set -euo pipefail

# Read input
days=$1
miles=$2
receipts=$3

# Get the cents portion
cents=$(echo "scale=0; ($receipts * 100) % 100 / 1" | bc)

# Calculate base reimbursement
if [ "$cents" -eq 49 ]; then
    # Special formula for .49 receipts
    base=$(echo "scale=4; 81.25 * $days + 0.37 * $miles - 0.024 * $receipts + 20.63" | bc)
elif [ "$cents" -eq 99 ]; then
    # Special formula for .99 receipts
    base=$(echo "scale=4; 52 * $days + 0.47 * $miles - 0.15 * $receipts + 189" | bc)
else
    # Normal piecewise formula
    if [ $(echo "$receipts <= 828.10" | bc -l) -eq 1 ]; then
        # Low receipt regime
        base=$(echo "scale=4; 56 * $days + 0.52 * $miles + 0.51 * $receipts + 32.49" | bc)
    else
        # High receipt regime
        base=$(echo "scale=4; 43 * $days + 0.39 * $miles + 0.08 * $receipts + 907.66" | bc)
    fi
fi

# Apply 5-day bonus (10%)
if [ "$days" -eq 5 ]; then
    base=$(echo "scale=4; $base * 1.1" | bc)
fi

# Apply high-miles bonuses for certain days
miles_per_day=$(echo "scale=2; $miles / $days" | bc)
if [ "$days" -ge 5 ] && [ "$days" -le 8 ]; then
    if [ $(echo "$miles_per_day > 140" | bc -l) -eq 1 ]; then
        # High efficiency bonus
        base=$(echo "scale=4; $base * 1.05" | bc)
    fi
fi

# Apply daily caps
case $days in
    1) cap=1475.40 ;;
    2) cap=1549.54 ;;
    3) cap=1587.80 ;;
    4) cap=1699.56 ;;
    5) cap=1810.37 ;;
    6) cap=1972.88 ;;
    7) 
        # Special handling for 7-day trips
        if [ $(echo "$base >= 1500" | bc -l) -eq 1 ]; then
            cap=99999  # No cap for high-value 7-day trips
        else
            cap=1459.63
        fi
        ;;
    8) cap=1897.19 ;;
    9) cap=1945.95 ;;
    10) cap=2013.21 ;;
    11) cap=2159.33 ;;
    12) cap=2162.13 ;;
    13) cap=2214.64 ;;
    14) cap=2337.73 ;;
    *) cap=99999 ;;  # No cap for other days
esac

# Apply cap
if [ $(echo "$base > $cap" | bc -l) -eq 1 ]; then
    reimbursement=$cap
else
    reimbursement=$base
fi

# Output with proper formatting
printf "%.2f\n" "$reimbursement"