#!/usr/bin/env python3
"""
Find which cases we're getting exactly right
"""

import subprocess
import json

# Load all cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

perfect_matches = []
close_matches = []

print("Testing all cases...")
for idx, case in enumerate(cases):
    if idx % 100 == 0:
        print(f"Progress: {idx}/1000")
    
    d = case['input']['trip_duration_days']
    m = case['input']['miles_traveled'] 
    r = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    # Run our script
    result = subprocess.run(['./run.sh', str(d), str(m), str(r)], 
                           capture_output=True, text=True)
    predicted = float(result.stdout.strip())
    
    error = abs(expected - predicted)
    
    if error < 0.01:
        perfect_matches.append((idx, d, m, r, expected, predicted))
    elif error < 1.0:
        close_matches.append((idx, d, m, r, expected, predicted, error))

print(f"\n\nFound {len(perfect_matches)} perfect matches:")
for match in perfect_matches:
    idx, d, m, r, exp, pred = match
    print(f"  Case {idx}: {d} days, {m} miles, ${r:.2f} receipts => ${exp:.2f}")

print(f"\nFound {len(close_matches)} close matches (< $1 error):")
for match in close_matches[:10]:  # Show first 10
    idx, d, m, r, exp, pred, err = match
    print(f"  Case {idx}: {d} days, {m} miles, ${r:.2f} receipts => ${exp:.2f} (predicted: ${pred:.2f}, error: ${err:.2f})")