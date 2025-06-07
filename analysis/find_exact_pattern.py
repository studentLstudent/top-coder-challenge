#!/usr/bin/env python3
import json

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== SEARCHING FOR EXACT PATTERN ===")
print("Since there's a perfect analytical solution, let's find it!\n")

# First, let's check if the formula might be:
# output = base_amount + receipt_adjustment
# where base_amount = per_diem * days + mileage_rate * miles

# Test with exact values
print("Testing hypothesis: output = per_diem * days + mileage_rate * miles + receipt_function")

# Let's focus on cases with very specific receipt amounts
print("\nLooking at cases with receipts = 0 or very low:")
zero_receipt_cases = []
for case in cases:
    if case['input']['total_receipts_amount'] < 10:
        zero_receipt_cases.append(case)

if not zero_receipt_cases:
    print("No cases with receipts < 10")
else:
    for case in zero_receipt_cases[:5]:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        output = case['expected_output']
        
        print(f"  D={days}, M={miles}, R=${receipts:.2f}, Output=${output:.2f}")
        
        # If receipts are near zero, output should be close to base
        # Try different combinations
        for pd in [85, 86, 87]:
            for mr in [0.55, 0.56, 0.57, 0.58]:
                base = pd * days + mr * miles
                diff = output - base - receipts
                if abs(diff) < 1:
                    print(f"    CLOSE: {pd}*{days} + {mr}*{miles} + {receipts} = {base + receipts:.2f} (diff={diff:.2f})")

# Let's check the decision tree breakpoint more carefully
print("\n\nAnalyzing the $828.10 breakpoint from decision tree:")
print("This is suspiciously specific! Let's see what's special about it...")

# Convert to cents to check for integer patterns
breakpoint_cents = int(828.10 * 100)
print(f"$828.10 = {breakpoint_cents} cents")
print(f"Factors of {breakpoint_cents}: ", end="")
factors = []
for i in range(1, min(1000, breakpoint_cents)):
    if breakpoint_cents % i == 0:
        factors.append(i)
print(factors[:20], "...")

# Maybe it's related to a threshold or cap?
print("\n\nTesting if $828.10 is related to per diem or mileage calculations:")
for days in range(1, 15):
    for pd in [85, 86, 87]:
        if pd * days == 828.10:
            print(f"  {pd} * {days} days = ${pd * days}")
            
for miles in range(1000, 2000):
    for mr in [0.55, 0.56, 0.57, 0.58]:
        if abs(mr * miles - 828.10) < 0.01:
            print(f"  {mr} * {miles} miles = ${mr * miles:.2f}")

# Let's check if outputs are always whole cents
print("\n\nChecking if outputs follow specific decimal patterns:")
decimal_parts = {}
for case in cases:
    output = case['expected_output']
    decimal = round((output - int(output)) * 100)  # Convert to cents
    if decimal not in decimal_parts:
        decimal_parts[decimal] = 0
    decimal_parts[decimal] += 1

print("Most common decimal parts (in cents):")
sorted_decimals = sorted(decimal_parts.items(), key=lambda x: -x[1])
for cents, count in sorted_decimals[:20]:
    print(f"  .{cents:02d}: {count} times")

# Final test: maybe the formula is simpler than we think
print("\n\nTesting very simple formulas with common financial values:")
test_formulas = [
    (86, 0.55, "86*days + 0.55*miles + receipts"),
    (86, 0.56, "86*days + 0.56*miles + receipts"),
    (86, 0.57, "86*days + 0.57*miles + receipts"),
    (86, 0.58, "86*days + 0.58*miles + receipts"),
    (87, 0.55, "87*days + 0.55*miles + receipts"),
    (87, 0.56, "87*days + 0.56*miles + receipts"),
    (87, 0.57, "87*days + 0.57*miles + receipts"),
    (87, 0.58, "87*days + 0.58*miles + receipts"),
]

for pd, mr, formula in test_formulas:
    exact_matches = 0
    total_error = 0
    
    for case in cases:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        output = case['expected_output']
        
        predicted = pd * days + mr * miles + receipts
        error = abs(output - predicted)
        
        if error < 0.01:
            exact_matches += 1
        total_error += error
    
    if exact_matches > 10:  # If we get some exact matches
        print(f"\n{formula}:")
        print(f"  Exact matches: {exact_matches}/1000")
        print(f"  Average error: ${total_error/1000:.2f}")