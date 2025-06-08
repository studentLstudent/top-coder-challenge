#!/usr/bin/env python3
"""
Find what makes certain cases special - they have very low reimbursements
"""

import json
import pandas as pd
import numpy as np

# Load data
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

data = []
for case in cases:
    d = int(case['input']['trip_duration_days'])
    m = float(case['input']['miles_traveled'])
    r = float(case['input']['total_receipts_amount'])
    output = float(case['expected_output'])
    
    # Calculate some ratios
    miles_per_day = m / d if d > 0 else 0
    receipts_per_day = r / d if d > 0 else 0
    output_per_day = output / d if d > 0 else 0
    
    data.append({
        'days': d, 
        'miles': m, 
        'receipts': r, 
        'expected': output,
        'miles_per_day': miles_per_day,
        'receipts_per_day': receipts_per_day,
        'output_per_day': output_per_day,
        'cents': int(round((r % 1) * 100))
    })

df = pd.DataFrame(data)

# Apply basic piecewise formula
BREAKPOINT = 828.10

def base_formula(row):
    d, m, r = row['days'], row['miles'], row['receipts']
    if r <= BREAKPOINT:
        return 56*d + 0.52*m + 0.51*r + 32.49
    else:
        return 43*d + 0.39*m + 0.08*r + 907.66

df['base_calc'] = df.apply(base_formula, axis=1)
df['ratio'] = df['expected'] / df['base_calc']

# Find extremely low reimbursement cases (ratio < 0.5)
extreme_low = df[df['ratio'] < 0.5].sort_values('ratio')

print(f"Found {len(extreme_low)} cases with reimbursement < 50% of base formula\n")

print("=== Extreme Low Reimbursement Cases ===")
for _, row in extreme_low.head(20).iterrows():
    print(f"Days: {row['days']}, Miles: {row['miles']:.0f}, Receipts: ${row['receipts']:.2f}")
    print(f"  Expected: ${row['expected']:.2f}, Base: ${row['base_calc']:.2f}, Ratio: {row['ratio']:.2f}")
    print(f"  Miles/day: {row['miles_per_day']:.1f}, Receipts/day: ${row['receipts_per_day']:.2f}")
    print(f"  Cents: {row['cents']}")
    print()

# Look for patterns
print("\n=== Pattern Analysis ===")

# Check if they all have .49 receipts
cents_49 = extreme_low[extreme_low['cents'] == 49]
print(f"Cases ending in .49: {len(cents_49)} out of {len(extreme_low)}")

# Check miles patterns
print(f"\nMiles statistics:")
print(f"  Mean: {extreme_low['miles'].mean():.1f}")
print(f"  Median: {extreme_low['miles'].median():.1f}")
print(f"  Min: {extreme_low['miles'].min():.1f}")
print(f"  Max: {extreme_low['miles'].max():.1f}")

# Check for high receipts
high_receipt_extreme = extreme_low[extreme_low['receipts'] > 1000]
print(f"\nHigh receipt (>$1000) extreme cases: {len(high_receipt_extreme)}")

# Look at the actual expected values
print("\n=== Expected Values Analysis ===")
for _, row in extreme_low.head(10).iterrows():
    # Try to find a pattern in the expected value
    exp = row['expected']
    
    # Check if it's related to miles * some rate
    if row['miles'] > 0:
        mile_rate = exp / row['miles']
        print(f"Expected ${exp:.2f} / {row['miles']:.0f} miles = ${mile_rate:.2f}/mile")
    
    # Check if it's a fixed value based on days
    print(f"  Days: {row['days']}, Expected/day: ${row['output_per_day']:.2f}")
    
# Group by days to see if there's a pattern
print("\n=== Grouped by Days ===")
for days in sorted(extreme_low['days'].unique()):
    day_cases = extreme_low[extreme_low['days'] == days]
    print(f"\nDay {days}: {len(day_cases)} cases")
    for _, row in day_cases.iterrows():
        print(f"  Expected: ${row['expected']:.2f}, Miles: {row['miles']:.0f}, Receipts: ${row['receipts']:.2f}")