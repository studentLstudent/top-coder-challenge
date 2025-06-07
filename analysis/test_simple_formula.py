#!/usr/bin/env python3
import json

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== TESTING SIMPLE FORMULA HYPOTHESIS ===")
print("What if the formula is just: per_diem * days + mileage_rate * miles?")
print("And receipts affect it in a more complex way?\n")

# Test different per diem and mileage rates
best_pd = None
best_mr = None
best_error = float('inf')

for pd in range(80, 105):
    for mr in [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60]:
        total_error = 0
        
        for case in cases:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            output = case['expected_output']
            
            # Simple formula without receipts
            predicted = pd * days + mr * miles
            error = abs(output - predicted)
            total_error += error
        
        avg_error = total_error / len(cases)
        if avg_error < best_error:
            best_error = avg_error
            best_pd = pd
            best_mr = mr

print(f"Best parameters (ignoring receipts):")
print(f"  Per diem: ${best_pd}")
print(f"  Mileage rate: ${best_mr}")
print(f"  Average error: ${best_error:.2f}")

# Now analyze the residuals
print("\n=== ANALYZING RESIDUALS ===")
residuals = []

for case in cases:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    base = best_pd * days + best_mr * miles
    residual = output - base
    residuals.append((receipts, residual, residual/receipts if receipts > 0 else 0))

# Sort by receipts
residuals.sort(key=lambda x: x[0])

# Look at pattern
print("\nResidual pattern by receipt amount:")
for i in range(0, len(residuals), 100):
    r, res, ratio = residuals[i]
    print(f"  Receipts=${r:.2f}, Residual=${res:.2f}, Ratio={ratio:.3f}")

# Check if residuals follow a simple pattern
print("\n=== TESTING RECEIPT FORMULA ===")
# Maybe receipts contribute partially based on amount
print("Testing: receipt_contribution = k * receipts where k varies by amount")

# Group by receipt ranges
ranges = [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 3000)]
for low, high in ranges:
    in_range = [(r, res) for r, res, _ in residuals if low <= r < high]
    if in_range:
        avg_residual = sum(res for _, res in in_range) / len(in_range)
        avg_receipt = sum(r for r, _ in in_range) / len(in_range)
        implied_k = avg_residual / avg_receipt if avg_receipt > 0 else 0
        print(f"  ${low}-${high}: implied k = {implied_k:.3f}")

# Test THE EXACT FORMULA from interviews
print("\n=== TESTING INTERVIEW CLUES ===")
# Lisa mentioned: "$100 a day seems to be the base"
# Also mentioned: "like 58 cents per mile"
print("Testing: 100*days + 0.58*miles + receipt_adjustment")

test_errors = []
for case in cases[:20]:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    base = 100 * days + 0.58 * miles
    diff = output - base
    
    print(f"  D={days}, M={miles:.0f}, R=${receipts:.2f}")
    print(f"    Output=${output:.2f}, Base=${base:.2f}, Diff=${diff:.2f}")
    
    test_errors.append(abs(diff))