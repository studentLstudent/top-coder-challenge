#!/usr/bin/env python3
"""
Final push - analyze all remaining errors to find the last patterns
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
    
    cents = round((r % 1) * 100)
    
    data.append({
        'days': d, 
        'miles': m, 
        'receipts': r, 
        'expected': output,
        'cents': cents,
        'miles_per_day': m / d
    })

df = pd.DataFrame(data)

def calculate_base(days, miles, receipts):
    """Calculate base reimbursement without caps"""
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

# Define caps based on our analysis
caps = {
    1: 1475.40,
    2: 1549.54,
    3: 1587.80,
    4: 1699.56,
    5: 1810.37,
    6: 1972.88,
    7: 1459.63,  # Lower cap for 7-day trips
    8: 1897.19,
    9: 1945.95,
    10: 2013.21,
    11: 2159.33,
    12: 2162.13,
    13: 2214.64,
    14: 2337.73
}

# Apply the model with caps
def apply_model(days, miles, receipts):
    base = calculate_base(days, miles, receipts)
    
    # Special handling for 7-day trips
    if days == 7 and base >= 1500:
        # No cap for high-value 7-day trips
        return base
    
    # Apply cap if exists
    if days in caps:
        return min(base, caps[days])
    
    return base

# Test model
df['predicted'] = df.apply(lambda row: apply_model(row['days'], row['miles'], row['receipts']), axis=1)
df['error'] = df['expected'] - df['predicted']
df['abs_error'] = abs(df['error'])

print(f"Total error: ${df['abs_error'].sum():.2f}")
print(f"Perfect predictions: {sum(df['abs_error'] < 0.01)}/1000")

# Analyze errors by characteristics
print("\n=== Error Analysis ===")

# Group errors by size
large_errors = df[df['abs_error'] > 200]
medium_errors = df[(df['abs_error'] > 50) & (df['abs_error'] <= 200)]

print(f"\nLarge errors (>$200): {len(large_errors)} cases")
print(f"Medium errors ($50-$200): {len(medium_errors)} cases")

# Look for patterns in large errors
print("\n=== Large Error Patterns ===")

# Check if they're mostly positive or negative
pos_errors = large_errors[large_errors['error'] > 0]
neg_errors = large_errors[large_errors['error'] < 0]

print(f"Positive errors (under-predicted): {len(pos_errors)}")
print(f"Negative errors (over-predicted): {len(neg_errors)}")

# Check for high miles per day
high_mpd = large_errors[large_errors['miles_per_day'] > 140]
print(f"\nHigh miles/day (>140): {len(high_mpd)} cases")
if len(high_mpd) > 0:
    print("Examples:")
    for _, row in high_mpd.head(5).iterrows():
        print(f"  Days: {row['days']}, Miles: {row['miles']:.0f} ({row['miles_per_day']:.0f}/day), "
              f"Expected: ${row['expected']:.2f}, Predicted: ${row['predicted']:.2f}, Error: ${row['error']:.2f}")

# Check specific day patterns
print("\n=== Errors by Day ===")
for days in range(1, 15):
    day_errors = large_errors[large_errors['days'] == days]
    if len(day_errors) > 0:
        print(f"Day {days}: {len(day_errors)} large errors")
        # Check if mostly under or over
        pos = len(day_errors[day_errors['error'] > 0])
        neg = len(day_errors[day_errors['error'] < 0])
        if pos > neg:
            print(f"  Mostly under-predicted ({pos} vs {neg})")
        elif neg > pos:
            print(f"  Mostly over-predicted ({neg} vs {pos})")

# Look for miles thresholds
print("\n=== Looking for Miles-Based Bonuses ===")
# Check if high-miles trips get bonuses
for days in [5, 6, 7, 8, 9]:
    day_df = df[df['days'] == days]
    if len(day_df) > 10:
        # Sort by miles and look for jumps in error
        sorted_df = day_df.sort_values('miles')
        
        # Look for systematic positive errors at high miles
        high_miles = sorted_df[sorted_df['miles'] > sorted_df['miles'].quantile(0.7)]
        if len(high_miles) > 0:
            avg_error = high_miles['error'].mean()
            if abs(avg_error) > 50:
                print(f"\nDay {days}: High miles (>{high_miles['miles'].min():.0f}) have avg error ${avg_error:.2f}")

# Check 5-day trips specifically
print("\n=== 5-Day Trip Analysis ===")
five_day = df[df['days'] == 5]
print(f"Total 5-day trips: {len(five_day)}")
print(f"Average error: ${five_day['error'].mean():.2f}")
print(f"Trips with error > $50: {len(five_day[five_day['abs_error'] > 50])}")

# Look for a bonus pattern
five_day_sorted = five_day.sort_values('expected')
# Check if there's a percentage bonus
five_day['bonus_pct'] = (five_day['expected'] - five_day['predicted']) / five_day['predicted'] * 100
avg_bonus_pct = five_day[five_day['bonus_pct'] > 0]['bonus_pct'].mean()
print(f"Average bonus percentage for under-predicted: {avg_bonus_pct:.1f}%")