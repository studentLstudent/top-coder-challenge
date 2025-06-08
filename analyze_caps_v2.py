#!/usr/bin/env python3
"""
Analyze caps more carefully - it seems the caps depend on BOTH days and receipts
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
    data.append({'days': d, 'miles': m, 'receipts': r, 'expected': output})

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
df['is_capped'] = df['expected'] < df['base_calc']

# Look at capped cases
capped = df[df['is_capped']]
print(f"Found {len(capped)} capped cases out of {len(df)}")

# Analyze cap patterns by day and receipt level
print("\n=== Cap Analysis by Days and Receipt Levels ===")
for days in range(1, 15):
    day_df = capped[capped['days'] == days]
    if len(day_df) > 0:
        print(f"\nDay {days}: {len(day_df)} capped cases")
        
        # Sort by receipts and look at expected outputs
        day_sorted = day_df.sort_values('receipts')
        
        # Group by similar expected values (within $5)
        caps_for_day = []
        current_cap = None
        cap_threshold = 5.0
        
        for _, row in day_sorted.iterrows():
            exp = row['expected']
            if current_cap is None or abs(exp - current_cap['value']) > cap_threshold:
                if current_cap:
                    caps_for_day.append(current_cap)
                current_cap = {
                    'value': exp,
                    'min_receipt': row['receipts'],
                    'max_receipt': row['receipts'],
                    'count': 1
                }
            else:
                current_cap['max_receipt'] = row['receipts']
                current_cap['count'] += 1
                # Update to average
                current_cap['value'] = (current_cap['value'] * (current_cap['count'] - 1) + exp) / current_cap['count']
        
        if current_cap:
            caps_for_day.append(current_cap)
        
        # Print cap levels
        for cap in caps_for_day:
            print(f"  Cap ~${cap['value']:.2f} for receipts ${cap['min_receipt']:.2f}-${cap['max_receipt']:.2f} ({cap['count']} cases)")

# Look at specific problematic cases
print("\n=== Analyzing Specific Problem Cases ===")
problems = [
    {'days': 8, 'miles': 795, 'receipts': 1645.99, 'expected': 644.69},
    {'days': 1, 'miles': 1082, 'receipts': 1809.49, 'expected': 446.94},
    {'days': 4, 'miles': 69, 'receipts': 2321.49, 'expected': 322.00}
]

for p in problems:
    row = pd.Series(p)
    base = base_formula(row)
    print(f"\nDays: {p['days']}, Miles: {p['miles']}, Receipts: ${p['receipts']:.2f}")
    print(f"  Base formula: ${base:.2f}")
    print(f"  Expected: ${p['expected']:.2f}")
    print(f"  Difference: ${p['expected'] - base:.2f}")
    
    # Look for similar cases
    similar = df[(df['days'] == p['days']) & (abs(df['receipts'] - p['receipts']) < 200)]
    print(f"  Similar cases ({len(similar)}):")
    for _, s in similar.iterrows():
        print(f"    Receipts: ${s['receipts']:.2f}, Expected: ${s['expected']:.2f}")