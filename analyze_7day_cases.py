#!/usr/bin/env python3
"""
Analyze 7-day cases - the cap seems too low
"""

import json
import pandas as pd

# Load data
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

data = []
for case in cases:
    d = int(case['input']['trip_duration_days'])
    m = float(case['input']['miles_traveled'])
    r = float(case['input']['total_receipts_amount'])
    output = float(case['expected_output'])
    
    if d == 7:  # Only 7-day trips
        data.append({
            'days': d, 
            'miles': m, 
            'receipts': r, 
            'expected': output
        })

df = pd.DataFrame(data)

# Sort by expected output
df_sorted = df.sort_values('expected')

print(f"Total 7-day trips: {len(df)}")
print(f"Expected output range: ${df['expected'].min():.2f} - ${df['expected'].max():.2f}")

# Look for a gap in expected values
print("\n=== Expected Values Distribution ===")
prev_val = None
for _, row in df_sorted.iterrows():
    if prev_val is not None and row['expected'] - prev_val > 100:
        print(f"GAP: ${prev_val:.2f} -> ${row['expected']:.2f} (diff: ${row['expected'] - prev_val:.2f})")
    prev_val = row['expected']

# Show all cases
print("\n=== All 7-Day Cases ===")
for _, row in df_sorted.iterrows():
    print(f"Expected: ${row['expected']:7.2f}, Miles: {row['miles']:4.0f}, Receipts: ${row['receipts']:7.2f}")

# Check if there are two different groups
low_group = df[df['expected'] < 1500]
high_group = df[df['expected'] > 1500]

print(f"\nLow group (<$1500): {len(low_group)} cases")
print(f"High group (>$1500): {len(high_group)} cases")

if len(low_group) > 0:
    print(f"\nLow group max: ${low_group['expected'].max():.2f}")
if len(high_group) > 0:
    print(f"High group min: ${high_group['expected'].min():.2f}")

# Check for patterns
print("\n=== Looking for Pattern ===")
# Maybe high miles get uncapped?
if len(low_group) > 0:
    print(f"\nLow group miles: min={low_group['miles'].min()}, max={low_group['miles'].max()}, avg={low_group['miles'].mean():.1f}")
if len(high_group) > 0:
    print(f"High group miles: min={high_group['miles'].min()}, max={high_group['miles'].max()}, avg={high_group['miles'].mean():.1f}")

# Check receipts
print("\nReceipts statistics:")
print(f"Low group: ${low_group['receipts'].mean():.2f} avg receipts" if len(low_group) > 0 else "No low group")
print(f"High group: ${high_group['receipts'].mean():.2f} avg receipts" if len(high_group) > 0 else "No high group")