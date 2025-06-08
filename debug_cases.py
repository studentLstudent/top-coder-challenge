#!/usr/bin/env python3
"""
Debug specific cases to understand errors
"""

import subprocess
import json

# Load all cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

# Test cases from eval output
test_indices = [149, 513, 229, 335, 603]

for idx in test_indices:
    if idx < len(cases):
        case = cases[idx]
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled'] 
        r = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        # Run our script
        result = subprocess.run(['./run.sh', str(d), str(m), str(r)], 
                               capture_output=True, text=True)
        predicted = float(result.stdout.strip())
        
        print(f"\nCase {idx}: {d} days, {m} miles, ${r} receipts")
        print(f"  Expected: ${expected:.2f}")
        print(f"  Predicted: ${predicted:.2f}")
        print(f"  Error: ${abs(expected - predicted):.2f}")
        
        # Analyze
        cents = round((r % 1) * 100)
        print(f"  Cents: {cents}")
        
        # Calculate what base would be
        if cents == 49:
            base = 81.25 * d + 0.37 * m - 0.024 * r + 20.63
        elif cents == 99:
            base = 52 * d + 0.47 * m - 0.15 * r + 189
        else:
            if r <= 828.10:
                base = 56 * d + 0.52 * m + 0.51 * r + 32.49
            else:
                base = 43 * d + 0.39 * m + 0.08 * r + 907.66
        
        print(f"  Base calculation: ${base:.2f}")
        
        # Check if 5-day bonus applied
        if d == 5:
            base *= 1.1
            print(f"  After 5-day bonus: ${base:.2f}")
        
        # Check high miles bonus
        if d >= 5 and d <= 8 and m/d > 140:
            base *= 1.05
            print(f"  After high-miles bonus: ${base:.2f}")