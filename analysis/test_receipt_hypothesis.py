#!/usr/bin/env python3
import json

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== TESTING RECEIPT HYPOTHESIS ===")
print("What if receipts are NOT directly added but instead affect the rate?\n")

# Look at cases with very high receipts
high_receipt_cases = []
for case in cases:
    if case['input']['total_receipts_amount'] > 2000:
        high_receipt_cases.append(case)

print(f"Found {len(high_receipt_cases)} cases with receipts > $2000")
print("\nFirst 10 high-receipt cases:")
for case in high_receipt_cases[:10]:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    # Calculate base (without receipts)
    base_86 = 86 * days + 0.55 * miles
    difference = output - base_86
    ratio = difference / receipts if receipts > 0 else 0
    
    print(f"  Days={days}, Miles={miles:.0f}, Receipts=${receipts:.2f}")
    print(f"    Output=${output:.2f}, Base (86*d + 0.55*m)=${base_86:.2f}")
    print(f"    Difference=${difference:.2f}, Ratio={ratio:.3f}")

# Now look at low receipt cases
print("\n\nLow receipt cases (< $50):")
low_receipt_cases = []
for case in cases:
    if case['input']['total_receipts_amount'] < 50:
        low_receipt_cases.append(case)

for case in low_receipt_cases[:10]:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    # Calculate base
    base_86 = 86 * days + 0.55 * miles
    difference = output - base_86
    
    print(f"  Days={days}, Miles={miles:.0f}, Receipts=${receipts:.2f}")
    print(f"    Output=${output:.2f}, Base=${base_86:.2f}, Diff=${difference:.2f}")

# Test if receipts might be ignored entirely in the formula
print("\n\n=== TESTING IF RECEIPTS ARE IGNORED ===")
total_error = 0
count = 0

for case in cases:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    # Simple formula without receipts
    predicted = 86 * days + 0.55 * miles
    error = abs(predicted - output)
    total_error += error
    count += 1
    
    if count <= 5:
        print(f"  Predicted=${predicted:.2f}, Actual=${output:.2f}, Error=${error:.2f}")

avg_error = total_error / count
print(f"\nAverage error if receipts ignored: ${avg_error:.2f}")