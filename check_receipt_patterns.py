#!/usr/bin/env python3
"""
Check for receipt patterns like .49 and .99
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
    
    # Check decimal part
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

print(f"Total cases: {len(df)}")
print(f"Cases ending in .49: {df['ends_49'].sum()}")
print(f"Cases ending in .99: {df['ends_99'].sum()}")

# Look at some examples
print("\nExamples of .49 receipts:")
for _, row in df[df['ends_49']].head(10).iterrows():
    print(f"  Receipts: ${row['receipts']:.2f}, Expected: ${row['expected']:.2f}")

print("\nExamples of .99 receipts:")
for _, row in df[df['ends_99']].head(10).iterrows():
    print(f"  Receipts: ${row['receipts']:.2f}, Expected: ${row['expected']:.2f}")

# Check distribution of cents values
cents_dist = df['cents'].value_counts().sort_index()
print("\nMost common cents values:")
print(cents_dist.head(20))