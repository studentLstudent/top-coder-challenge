#!/usr/bin/env python3
import json
import numpy as np

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

# Let's look at some specific test cases from the errors
print("=== ANALYZING ERROR CASES ===")
error_cases = [
    {"case": 152, "days": 4, "miles": 69, "receipts": 2321.49, "expected": 322.00},
    {"case": 242, "days": 14, "miles": 1056, "receipts": 2489.69, "expected": 1894.16},
]

for ec in error_cases:
    print(f"\nCase {ec['case']}: {ec['days']} days, {ec['miles']} miles, ${ec['receipts']} receipts")
    print(f"  Expected: ${ec['expected']}")
    
    # Test different hypotheses
    # Hypothesis 1: receipts are NOT added directly
    base = 85 * ec['days'] + 0.58 * ec['miles']
    print(f"  If no receipts counted: ${base:.2f}")
    
    # Hypothesis 2: receipts have a cap
    base_with_cap = 85 * ec['days'] + 0.58 * ec['miles'] + min(ec['receipts'], 100)
    print(f"  If receipts capped at $100: ${base_with_cap:.2f}")

# Let's look for cases where receipts seem to be ignored
print("\n=== LOOKING FOR RECEIPT PATTERNS ===")
for i, case in enumerate(cases[:50]):
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    # Calculate what output would be with NO receipt reimbursement
    no_receipt = 85 * days + 0.58 * miles
    
    # If output is less than no_receipt + receipts, receipts are being penalized
    if output < no_receipt:
        print(f"Case {i}: Output ${output:.2f} < base ${no_receipt:.2f} (receipts ${receipts:.2f})")

# Analyze the simplest cases
print("\n=== ANALYZING SIMPLE CASES ===")
simple_cases = []
for case in cases:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    # Look for 1-day trips with low miles and low receipts
    if days == 1 and miles < 100 and receipts < 50:
        simple_cases.append((days, miles, receipts, output))
        
simple_cases.sort(key=lambda x: x[3])  # Sort by output
print("Simple 1-day trips:")
for d, m, r, o in simple_cases[:10]:
    # Try to reverse engineer the formula
    base_component = o - r  # Assuming receipts are added
    per_mile = (base_component - 85) / m if m > 0 else 0
    print(f"  Miles={m:.1f}, Receipts=${r:.2f}, Output=${o:.2f}")
    print(f"    If per_diem=85, mileage rate=${per_mile:.3f}/mi")
    
# Check if per diem might be 86 instead of 85
print("\n=== TESTING PER DIEM = 86 ===")
for d, m, r, o in simple_cases[:5]:
    base_component = o - r
    per_mile = (base_component - 86) / m if m > 0 else 0
    print(f"  Miles={m:.1f}, Output=${o:.2f}, Implied rate with PD=86: ${per_mile:.3f}/mi")