#!/usr/bin/env python3
import json

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== REVERSE ENGINEERING THE EXACT FORMULA ===")

# Look at Case 152 specifically: 4 days, 69 miles, $2321.49 receipts -> $322.00
print("Case 152: 4 days, 69 miles, $2321.49 receipts -> Expected $322.00")

# If we ignore receipts entirely:
for pd in range(80, 90):
    for mr_cents in range(50, 60):
        mr = mr_cents / 100
        base = pd * 4 + mr * 69
        if abs(base - 322.00) < 1:
            print(f"  Found: {pd}*4 + {mr:.2f}*69 = ${base:.2f}")

# Let's check more cases to see if receipts are completely ignored
print("\n\nTesting if receipts are ignored for high-receipt cases:")
high_receipt_cases = [c for c in cases if c['input']['total_receipts_amount'] > 2000]

matches = 0
for case in high_receipt_cases[:20]:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    # Test formula without receipts
    predicted = 86 * days + 0.55 * miles
    error = abs(output - predicted)
    
    if error < 1:
        matches += 1
        print(f"  MATCH: D={days}, M={miles}, R=${receipts:.2f}")
        print(f"    Expected=${output:.2f}, Predicted=${predicted:.2f}")

print(f"\nMatches for high receipts: {matches}/{min(20, len(high_receipt_cases))}")

# But we know low receipts DO affect output (from earlier analysis)
# So maybe there's a threshold?

print("\n\nTesting receipt threshold hypothesis:")
print("Maybe receipts are added up to a certain amount?")

# Test different thresholds
for threshold in [50, 100, 150, 200, 250, 300]:
    exact_matches = 0
    
    for case in cases:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        output = case['expected_output']
        
        # Cap receipts at threshold
        capped_receipts = min(receipts, threshold)
        predicted = 86 * days + 0.55 * miles + capped_receipts
        
        if abs(output - predicted) < 0.01:
            exact_matches += 1
    
    if exact_matches > 0:
        print(f"  Threshold ${threshold}: {exact_matches} exact matches")

# Maybe receipts are subtracted?
print("\n\nTesting if high receipts cause PENALTIES:")
for case in high_receipt_cases[:5]:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    base = 86 * days + 0.55 * miles
    diff = output - base
    
    print(f"  D={days}, M={miles}, R=${receipts:.2f}")
    print(f"    Output=${output:.2f}, Base=${base:.2f}")
    print(f"    Difference=${diff:.2f} (receipts would add ${receipts:.2f})")

# Check for a specific pattern
print("\n\nLooking for receipt formula pattern:")
receipt_effects = []

for case in cases:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    base = 86 * days + 0.55 * miles
    effect = output - base
    
    receipt_effects.append((receipts, effect))

# Sort by receipts
receipt_effects.sort(key=lambda x: x[0])

# Sample every 50 cases
print("\nReceipt vs Effect on Output:")
for i in range(0, len(receipt_effects), 50):
    r, e = receipt_effects[i]
    print(f"  Receipts=${r:.2f} -> Effect=${e:.2f}")

# Check if there's a simple transformation
print("\n\nTesting logarithmic or other transformations:")
import math

for case in cases[:10]:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    base = 86 * days + 0.55 * miles
    
    # Test different transformations
    if receipts > 0:
        log_receipts = math.log(receipts + 1)
        sqrt_receipts = math.sqrt(receipts)
        
        print(f"\nCase: D={days}, M={miles}, R=${receipts:.2f}, Output=${output:.2f}")
        print(f"  Base=${base:.2f}, Need=${output - base:.2f}")
        print(f"  log(R+1)={log_receipts:.2f}")
        print(f"  sqrt(R)={sqrt_receipts:.2f}")