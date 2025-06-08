#!/usr/bin/env python3
"""
Complete model with all discovered patterns:
1. Piecewise formula based on receipt threshold
2. Special formula for .49 receipts
3. Daily caps
4. Other quirks for .99 receipts
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict

# Load data
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

data = []
for case in cases:
    d = int(case['input']['trip_duration_days'])
    m = float(case['input']['miles_traveled'])
    r = float(case['input']['total_receipts_amount'])
    output = float(case['expected_output'])
    
    cents = round((r % 1) * 100)
    
    data.append({
        'days': d, 
        'miles': m, 
        'receipts': r, 
        'expected': output,
        'cents': cents,
        'ends_49': cents == 49,
        'ends_99': cents == 99
    })

df = pd.DataFrame(data)

def calculate_reimbursement(days, miles, receipts):
    """Calculate reimbursement with all discovered rules"""
    
    cents = round((receipts % 1) * 100)
    
    # Special formula for .49 receipts
    if cents == 49:
        return 81.25 * days + 0.37 * miles - 0.024 * receipts + 20.63
    
    # Normal piecewise formula
    BREAKPOINT = 828.10
    
    if receipts <= BREAKPOINT:
        base = 56 * days + 0.52 * miles + 0.51 * receipts + 32.49
    else:
        base = 43 * days + 0.39 * miles + 0.08 * receipts + 907.66
    
    # For .99 receipts, need to find the pattern
    if cents == 99:
        # This needs more analysis - for now return base
        return base
    
    # Apply daily caps (need to determine these properly)
    # For now, return base
    return base

# Test the model
df['predicted'] = df.apply(lambda row: calculate_reimbursement(row['days'], row['miles'], row['receipts']), axis=1)
df['error'] = df['expected'] - df['predicted']
df['abs_error'] = abs(df['error'])

print("=== Model Performance ===")
print(f"Total absolute error: ${df['abs_error'].sum():.2f}")
print(f"Average error: ${df['abs_error'].mean():.2f}")
print(f"Max error: ${df['abs_error'].max():.2f}")

# Show worst predictions
print("\n=== Top 10 Worst Predictions ===")
for _, row in df.nlargest(10, 'abs_error').iterrows():
    print(f"Days: {row['days']}, Miles: {row['miles']:.0f}, Receipts: ${row['receipts']:.2f}")
    print(f"  Expected: ${row['expected']:.2f}, Predicted: ${row['predicted']:.2f}, Error: ${row['error']:.2f}")
    print(f"  Ends in: .{row['cents']:02d}")

# Analyze remaining patterns
print("\n=== Analyzing Remaining Patterns ===")

# Look at .99 cases more carefully
df_99 = df[df['ends_99']]
print(f"\n.99 Receipt Cases ({len(df_99)} total):")

# Check if .99 cases have caps
for _, row in df_99.iterrows():
    if row['abs_error'] > 100:
        print(f"  Days: {row['days']}, Receipts: ${row['receipts']:.2f}, Expected: ${row['expected']:.2f}, Predicted: ${row['predicted']:.2f}")

# Now let's find the actual caps by looking at all cases
print("\n=== Finding True Daily Caps ===")

# Group by days and find cases where expected < predicted
for days in range(1, 15):
    day_df = df[(df['days'] == days) & (df['expected'] < df['predicted'] - 10)]
    if len(day_df) > 0:
        # Find the maximum expected value for capped cases
        max_exp = day_df['expected'].max()
        print(f"Day {days}: Max capped value = ${max_exp:.2f} ({len(day_df)} capped cases)")
        
        # Look for patterns in receipt amounts
        receipt_groups = day_df.groupby(pd.cut(day_df['receipts'], bins=10))['expected'].agg(['mean', 'count'])
        if len(receipt_groups[receipt_groups['count'] > 0]) > 1:
            print(f"  Multiple cap levels detected:")