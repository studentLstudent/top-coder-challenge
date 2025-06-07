#!/usr/bin/env python3
import json

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== ANALYZING HIGH ERROR CASES ===")

# High error cases from eval.sh
error_cases = [
    (684, 8, 795, 1645.99, 644.69),
    (367, 11, 740, 1171.99, 902.09),
    (520, 14, 481, 939.99, 877.17),
    (548, 8, 482, 1411.49, 631.81),
    (996, 1, 1082, 1809.49, 446.94)
]

print("\nNotice: All high-error cases have HIGH receipts but LOW expected output!")
print("This suggests receipts might not be added at all!\n")

for case_num, days, miles, receipts, expected in error_cases:
    base = 100 * days + 0.58 * miles
    diff = expected - base
    
    print(f"Case {case_num}: {days} days, {miles} miles, ${receipts:.2f} receipts")
    print(f"  Expected: ${expected:.2f}")
    print(f"  Base (100*d + 0.58*m): ${base:.2f}")
    print(f"  Difference: ${diff:.2f}")
    print(f"  Receipt contribution: ${diff:.2f} (NOT ${receipts:.2f}!)")
    print()

# Test hypothesis: Receipts are NOT added, but affect the rate
print("=== TESTING NEW HYPOTHESIS ===")
print("What if output = per_diem * days + mileage_rate * miles?")
print("And receipts DON'T contribute directly?\n")

# Find cases with same days/miles but different receipts
duplicates = {}
for i, case in enumerate(cases):
    key = (case['input']['trip_duration_days'], case['input']['miles_traveled'])
    if key not in duplicates:
        duplicates[key] = []
    duplicates[key].append((i, case))

print("Looking for cases with same days/miles but different receipts...")
found_duplicates = False
for key, case_list in duplicates.items():
    if len(case_list) > 1:
        found_duplicates = True
        print(f"\nDays={key[0]}, Miles={key[1]}:")
        for idx, case in case_list:
            r = case['input']['total_receipts_amount']
            o = case['expected_output']
            print(f"  Case {idx}: Receipts=${r:.2f}, Output=${o:.2f}")

if not found_duplicates:
    print("No exact duplicates found with different receipts.")

# Let's check if the formula is simpler
print("\n=== TESTING SIMPLER FORMULA ===")
# Maybe it's just per_diem * days + mileage * miles with specific rates

# Test different combinations
best_combo = None
best_exact = 0

for pd in [85, 86, 87, 88, 89, 90]:
    for mr in [0.54, 0.55, 0.56, 0.57, 0.58]:
        exact_matches = 0
        total_error = 0
        
        for case in cases:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            output = case['expected_output']
            
            predicted = pd * days + mr * miles
            error = abs(output - predicted)
            
            if error < 0.01:
                exact_matches += 1
            total_error += error
        
        if exact_matches > best_exact:
            best_exact = exact_matches
            best_combo = (pd, mr, total_error / len(cases))

print(f"Best simple formula: {best_combo[0]}*days + {best_combo[1]}*miles")
print(f"  Exact matches: {best_exact}/1000")
print(f"  Average error: ${best_combo[2]:.2f}")

# Check first few cases with this formula
print("\nTesting on first 10 cases:")
pd, mr, _ = best_combo
for i in range(10):
    case = cases[i]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    predicted = pd * days + mr * miles
    error = output - predicted
    
    print(f"  Case {i}: D={days}, M={miles}, R=${receipts:.2f}")
    print(f"    Expected=${output:.2f}, Predicted=${predicted:.2f}, Error=${error:.2f}")