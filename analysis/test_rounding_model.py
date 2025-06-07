#!/usr/bin/env python3
import json
import math

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== TESTING ROUNDING MODEL PER THE BIG PLAN PART 2 ===")
print("Hypothesis: Legacy system uses integer math (cents) with INT(x + 0.5) rounding\n")

# Test a simple formula with different rounding approaches
def test_rounding_methods():
    errors_float = []
    errors_int_cents = []
    errors_int_early_round = []
    
    # Test parameters
    per_diem = 86
    mileage_rate = 0.55
    
    for case in cases[:100]:  # Test first 100 cases
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        # Method 1: Float arithmetic with final rounding
        result_float = per_diem * days + mileage_rate * miles + receipts
        result_float = round(result_float, 2)
        errors_float.append(abs(result_float - expected))
        
        # Method 2: Integer cents with late rounding (COBOL style)
        per_diem_cents = per_diem * 100
        mileage_cents = int(miles * 55)  # 0.55 * 100 = 55 cents per mile
        receipts_cents = int(receipts * 100)
        
        total_cents = per_diem_cents * days + mileage_cents + receipts_cents
        # COBOL rounding: INT(x + 0.5) at cent level
        result_int_cents = total_cents / 100.0
        errors_int_cents.append(abs(result_int_cents - expected))
        
        # Method 3: Early rounding (wrong approach)
        per_diem_rounded = round(per_diem * days, 2)
        mileage_rounded = round(mileage_rate * miles, 2)
        receipts_rounded = round(receipts, 2)
        result_early = per_diem_rounded + mileage_rounded + receipts_rounded
        errors_int_early_round.append(abs(result_early - expected))
    
    print("Average errors for different rounding methods:")
    print(f"  Float with final round: ${sum(errors_float)/len(errors_float):.4f}")
    print(f"  Integer cents (COBOL):  ${sum(errors_int_cents)/len(errors_int_cents):.4f}")
    print(f"  Early rounding:         ${sum(errors_int_early_round)/len(errors_int_early_round):.4f}")
    
    # Check if there's a consistent offset
    print("\nChecking for systematic bias in first 10 cases:")
    for i in range(10):
        days = cases[i]['input']['trip_duration_days']
        miles = cases[i]['input']['miles_traveled']
        receipts = cases[i]['input']['total_receipts_amount']
        expected = cases[i]['expected_output']
        
        # Integer cents calculation
        total_cents = 8600 * days + int(miles * 55) + int(receipts * 100)
        result = total_cents / 100.0
        
        diff = result - expected
        print(f"  Case {i}: Expected=${expected:.2f}, IntCents=${result:.2f}, Diff=${diff:.2f}")

test_rounding_methods()

print("\n\n=== TESTING SPECIFIC ROUNDING FUNCTIONS ===")

def cobol_round(x):
    """COBOL-style rounding: INT(x + 0.5)"""
    return math.floor(x + 0.5)

def bankers_round(x):
    """Python's round() uses banker's rounding"""
    return round(x)

# Test on specific values that differ between rounding methods
test_values = [2.5, 3.5, 4.5, 5.5, 10.25, 10.75, 10.50]
print("Comparing rounding methods on edge cases:")
print("Value   COBOL  Banker's  Difference")
for val in test_values:
    cobol = cobol_round(val)
    banker = bankers_round(val)
    print(f"{val:5.2f}   {cobol:5.0f}    {banker:5.0f}     {cobol-banker:2.0f}")

# Now test if using COBOL rounding improves accuracy
print("\n\n=== TESTING WITH COBOL ROUNDING ===")

exact_matches = 0
close_matches = 0

for i, case in enumerate(cases[:50]):
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    # Test with integer cents and COBOL rounding
    # Assuming simple formula for now
    total_cents = 8600 * days + cobol_round(miles * 55) + cobol_round(receipts * 100)
    result = total_cents / 100.0
    
    error = abs(result - expected)
    if error < 0.01:
        exact_matches += 1
    elif error < 1.00:
        close_matches += 1
    
    if i < 5:
        print(f"Case {i}: Expected=${expected:.2f}, Calculated=${result:.2f}, Error=${error:.2f}")

print(f"\nExact matches: {exact_matches}/50")
print(f"Close matches: {close_matches}/50")