#!/usr/bin/env python3
"""
Iterative refinement approach - shrink the diff to zero
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
        'cents': cents
    })

df = pd.DataFrame(data)

def calculate_reimbursement_v1(days, miles, receipts):
    """Version 1: Base formulas only"""
    cents = round((receipts % 1) * 100)
    
    # Special formulas for .49 and .99
    if cents == 49:
        return 81.25 * days + 0.37 * miles - 0.024 * receipts + 20.63
    
    if cents == 99:
        return 52 * days + 0.47 * miles - 0.15 * receipts + 189
    
    # Normal piecewise formula
    BREAKPOINT = 828.10
    
    if receipts <= BREAKPOINT:
        return 56 * days + 0.52 * miles + 0.51 * receipts + 32.49
    else:
        return 43 * days + 0.39 * miles + 0.08 * receipts + 907.66

# Test v1
df['pred_v1'] = df.apply(lambda row: calculate_reimbursement_v1(row['days'], row['miles'], row['receipts']), axis=1)
df['error_v1'] = df['expected'] - df['pred_v1']
df['abs_error_v1'] = abs(df['error_v1'])

print("=== Version 1: Base Formulas ===")
print(f"Total error: ${df['abs_error_v1'].sum():.2f}")
print(f"Perfect predictions: {sum(df['abs_error_v1'] < 0.01)}/1000")

# Analyze remaining errors
print("\n=== Analyzing Remaining Patterns ===")

# Look for caps
df['is_capped'] = df['error_v1'] < -10  # Predicted too high

# Find caps by day
daily_caps = {}
for days in range(1, 15):
    day_df = df[(df['days'] == days) & df['is_capped']]
    if len(day_df) > 0:
        # The cap is the maximum expected value among capped cases
        cap = day_df['expected'].max()
        daily_caps[days] = cap
        print(f"Day {days}: Cap = ${cap:.2f} ({len(day_df)} capped cases)")

print(f"\nFound caps for {len(daily_caps)} different day values")

def calculate_reimbursement_v2(days, miles, receipts):
    """Version 2: With daily caps"""
    base = calculate_reimbursement_v1(days, miles, receipts)
    
    # Apply cap if exists
    if days in daily_caps:
        return min(base, daily_caps[days])
    
    return base

# Test v2
df['pred_v2'] = df.apply(lambda row: calculate_reimbursement_v2(row['days'], row['miles'], row['receipts']), axis=1)
df['error_v2'] = df['expected'] - df['pred_v2']
df['abs_error_v2'] = abs(df['error_v2'])

print("\n=== Version 2: With Daily Caps ===")
print(f"Total error: ${df['abs_error_v2'].sum():.2f}")
print(f"Perfect predictions: {sum(df['abs_error_v2'] < 0.01)}/1000")

# Show remaining errors
print("\n=== Top 10 Remaining Errors ===")
for _, row in df.nlargest(10, 'abs_error_v2').iterrows():
    print(f"Days: {row['days']}, Miles: {row['miles']:.0f}, Receipts: ${row['receipts']:.2f}")
    print(f"  Expected: ${row['expected']:.2f}, Predicted: ${row['pred_v2']:.2f}, Error: ${row['error_v2']:.2f}")

# Look for more patterns
print("\n=== Looking for More Patterns ===")

# Check if errors correlate with specific features
high_error_df = df[df['abs_error_v2'] > 50]

# 5-day trips
five_day_errors = df[df['days'] == 5]['error_v2']
if abs(five_day_errors.mean()) > 5:
    print(f"\n5-day trips have systematic error: mean = ${five_day_errors.mean():.2f}")
    
# High efficiency trips
df['miles_per_day'] = df['miles'] / df['days']
high_eff = df[df['miles_per_day'] > 200]
if len(high_eff) > 0 and abs(high_eff['error_v2'].mean()) > 5:
    print(f"\nHigh efficiency trips (>200 mi/day) have error: mean = ${high_eff['error_v2'].mean():.2f}")

# Export current best model parameters
model_params = {
    'breakpoint': 828.10,
    'low_regime': {'per_diem': 56, 'mileage': 0.52, 'receipt_rate': 0.51, 'intercept': 32.49},
    'high_regime': {'per_diem': 43, 'mileage': 0.39, 'receipt_rate': 0.08, 'intercept': 907.66},
    'special_49': {'per_diem': 81.25, 'mileage': 0.37, 'receipt_rate': -0.024, 'intercept': 20.63},
    'special_99': {'per_diem': 52, 'mileage': 0.47, 'receipt_rate': -0.15, 'intercept': 189},
    'daily_caps': daily_caps,
    'total_error': df['abs_error_v2'].sum(),
    'perfect_predictions': sum(df['abs_error_v2'] < 0.01)
}

with open('model_params_v2.json', 'w') as f:
    json.dump(model_params, f, indent=2)

print("\nModel parameters saved to model_params_v2.json")