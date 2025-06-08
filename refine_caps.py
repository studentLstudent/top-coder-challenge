#!/usr/bin/env python3
"""
Refine the cap logic - some caps only apply to certain receipt ranges
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
        'cents': cents
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

# Calculate base for all
df['base'] = df.apply(lambda row: calculate_base(row['days'], row['miles'], row['receipts']), axis=1)
df['is_capped'] = df['expected'] < df['base'] - 5

# Analyze caps by day and receipt level
print("=== Analyzing Caps by Day and Receipt Level ===")

cap_rules = {}

for days in range(1, 15):
    day_df = df[df['days'] == days]
    capped_df = day_df[day_df['is_capped']]
    
    if len(capped_df) == 0:
        continue
        
    print(f"\nDay {days}: {len(capped_df)} capped cases out of {len(day_df)}")
    
    # Look for receipt thresholds
    capped_receipts = capped_df['receipts'].values
    uncapped_df = day_df[~day_df['is_capped']]
    
    if len(uncapped_df) > 0:
        uncapped_receipts = uncapped_df['receipts'].values
        
        # Find the boundary
        max_capped_receipt = capped_receipts.max() if len(capped_receipts) > 0 else 0
        min_uncapped_receipt = uncapped_receipts.min() if len(uncapped_receipts) > 0 else 99999
        
        if min_uncapped_receipt < max_capped_receipt:
            # Overlapping - look for expected value threshold
            max_capped_expected = capped_df['expected'].max()
            min_uncapped_expected = uncapped_df['expected'].min()
            
            print(f"  Complex cap structure:")
            print(f"    Capped cases: receipts ${capped_receipts.min():.2f}-${capped_receipts.max():.2f}")
            print(f"    Capped expected: ${capped_df['expected'].min():.2f}-${capped_df['expected'].max():.2f}")
            print(f"    Uncapped cases: receipts ${uncapped_receipts.min():.2f}-${uncapped_receipts.max():.2f}")
            print(f"    Uncapped expected: ${uncapped_df['expected'].min():.2f}-${uncapped_df['expected'].max():.2f}")
            
            # For 7-day trips specifically
            if days == 7:
                # Check if there's a clear expected value gap
                if min_uncapped_expected - max_capped_expected > 20:
                    print(f"    GAP in expected values: ${max_capped_expected:.2f} -> ${min_uncapped_expected:.2f}")
                    cap_rules[days] = {
                        'type': 'expected_threshold',
                        'cap': max_capped_expected,
                        'threshold': (max_capped_expected + min_uncapped_expected) / 2
                    }
            else:
                cap_rules[days] = {
                    'type': 'simple',
                    'cap': capped_df['expected'].max()
                }
        else:
            # Clear receipt threshold
            print(f"  Receipt threshold at ~${(max_capped_receipt + min_uncapped_receipt) / 2:.2f}")
            print(f"  Cap value: ${capped_df['expected'].max():.2f}")
            cap_rules[days] = {
                'type': 'receipt_threshold',
                'receipt_threshold': (max_capped_receipt + min_uncapped_receipt) / 2,
                'cap': capped_df['expected'].max()
            }
    else:
        # All cases are capped
        cap_rules[days] = {
            'type': 'simple',
            'cap': capped_df['expected'].max()
        }
        print(f"  All cases capped at ${capped_df['expected'].max():.2f}")

# Test the refined model
def calculate_with_refined_caps(days, miles, receipts):
    base = calculate_base(days, miles, receipts)
    
    if days not in cap_rules:
        return base
        
    rule = cap_rules[days]
    
    if rule['type'] == 'simple':
        return min(base, rule['cap'])
    elif rule['type'] == 'receipt_threshold':
        if receipts < rule['receipt_threshold']:
            return min(base, rule['cap'])
        else:
            return base
    elif rule['type'] == 'expected_threshold':
        # For 7-day trips, cap only if base calculation is below threshold
        if base < rule['threshold']:
            return min(base, rule['cap'])
        else:
            return base
    
    return base

# Test refined model
df['refined'] = df.apply(lambda row: calculate_with_refined_caps(row['days'], row['miles'], row['receipts']), axis=1)
df['error_refined'] = df['expected'] - df['refined']
df['abs_error_refined'] = abs(df['error_refined'])

print("\n=== Refined Model Performance ===")
print(f"Total error: ${df['abs_error_refined'].sum():.2f}")
print(f"Perfect predictions: {sum(df['abs_error_refined'] < 0.01)}/1000")

# Show remaining large errors
print("\n=== Top 10 Remaining Errors ===")
for _, row in df.nlargest(10, 'abs_error_refined').iterrows():
    print(f"Days: {row['days']}, Miles: {row['miles']:.0f}, Receipts: ${row['receipts']:.2f}")
    print(f"  Expected: ${row['expected']:.2f}, Predicted: ${row['refined']:.2f}, Error: ${row['error_refined']:.2f}")

# Save refined rules
with open('refined_cap_rules.json', 'w') as f:
    # Convert numpy types to native Python types
    clean_rules = {}
    for k, v in cap_rules.items():
        clean_rules[str(k)] = {k2: float(v2) if isinstance(v2, (np.integer, np.floating)) else v2 
                               for k2, v2 in v.items()}
    json.dump(clean_rules, f, indent=2)