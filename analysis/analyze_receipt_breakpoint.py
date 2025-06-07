#!/usr/bin/env python3
import json
import numpy as np

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== ANALYZING THE $828.10 RECEIPT BREAKPOINT ===")

# Separate cases by receipt amount
low_receipt_cases = []
high_receipt_cases = []

for case in cases:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    if receipts <= 828.10:
        low_receipt_cases.append((days, miles, receipts, output))
    else:
        high_receipt_cases.append((days, miles, receipts, output))

print(f"Cases with receipts <= $828.10: {len(low_receipt_cases)}")
print(f"Cases with receipts > $828.10: {len(high_receipt_cases)}")

# Analyze patterns in each group
print("\n=== LOW RECEIPT CASES (≤ $828.10) ===")
print("Testing formula: output = 86*days + 0.55*miles + receipts")

errors = []
for d, m, r, o in low_receipt_cases[:20]:
    predicted = 86 * d + 0.55 * m + r
    error = o - predicted
    errors.append(abs(error))
    print(f"  D={d}, M={m:.0f}, R=${r:.2f} -> Expected=${o:.2f}, Predicted=${predicted:.2f}, Error=${error:.2f}")

print(f"\nAverage absolute error: ${np.mean(errors):.2f}")

print("\n=== HIGH RECEIPT CASES (> $828.10) ===")
print("Looking for different formula...")

# Test if high receipts might have a cap or different treatment
for d, m, r, o in high_receipt_cases[:10]:
    base = 86 * d + 0.55 * m
    receipt_contribution = o - base
    ratio = receipt_contribution / r
    print(f"  D={d}, M={m:.0f}, R=${r:.2f} -> Output=${o:.2f}")
    print(f"    Base (86*d + 0.55*m) = ${base:.2f}")
    print(f"    Receipt contribution = ${receipt_contribution:.2f} ({ratio:.3f} of receipts)")

# Test specific hypothesis: receipts > 828.10 are capped
print("\n=== TESTING RECEIPT CAP HYPOTHESIS ===")
receipt_cap = 828.10

total_error = 0
exact_matches = 0

for case in cases:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    # Apply cap
    capped_receipts = min(receipts, receipt_cap)
    predicted = 86 * days + 0.55 * miles + capped_receipts
    
    error = abs(output - predicted)
    total_error += error
    
    if error < 0.01:
        exact_matches += 1

print(f"With receipt cap at ${receipt_cap}:")
print(f"  Average error: ${total_error / len(cases):.2f}")
print(f"  Exact matches: {exact_matches}/{len(cases)}")

# Look for other patterns near the breakpoint
print("\n=== CASES NEAR THE BREAKPOINT ===")
near_breakpoint = []
for case in cases:
    receipts = case['input']['total_receipts_amount']
    if 800 <= receipts <= 850:
        near_breakpoint.append(case)

near_breakpoint.sort(key=lambda x: x['input']['total_receipts_amount'])

for case in near_breakpoint[:10]:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    predicted_no_cap = 86 * days + 0.55 * miles + receipts
    predicted_with_cap = 86 * days + 0.55 * miles + min(receipts, 828.10)
    
    print(f"  R=${receipts:.2f}: Output=${output:.2f}")
    print(f"    No cap: ${predicted_no_cap:.2f} (error=${abs(output-predicted_no_cap):.2f})")
    print(f"    With cap: ${predicted_with_cap:.2f} (error=${abs(output-predicted_with_cap):.2f})")