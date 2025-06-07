#!/usr/bin/env python3
import json

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== ANALYZING RECEIPT EFFECT ON SAME TRIPS ===")

# Let's look at specific examples where same days/miles have different outputs
examples = [
    # (days, miles, [(receipts1, output1), (receipts2, output2)])
    (1, 140, [(22.71, 199.68), (255.99, 150.34)]),
    (3, 80, [(21.05, 366.87), (517.54, 457.49)]),
    (5, 477, [(704.42, 1045.96), (655.24, 935.38)]),
    (9, 524, [(474.75, 935.40), (2367.12, 1640.78), (136.46, 848.89)]),
]

for days, miles, cases_list in examples:
    print(f"\nDays={days}, Miles={miles}:")
    
    # Calculate base without receipts
    base_values = []
    for pd in range(85, 95):
        for mr in [0.54, 0.55, 0.56, 0.57, 0.58]:
            base = pd * days + mr * miles
            base_values.append((pd, mr, base))
    
    # Find which base is closest to outputs
    for receipts, output in cases_list:
        print(f"  Receipts=${receipts:.2f}, Output=${output:.2f}")
        
        # Find closest base
        best_diff = float('inf')
        best_base = None
        for pd, mr, base in base_values:
            diff = abs(output - base)
            if diff < best_diff:
                best_diff = diff
                best_base = (pd, mr, base)
        
        pd, mr, base = best_base
        print(f"    Closest base: {pd}*{days} + {mr:.2f}*{miles} = ${base:.2f} (diff=${output-base:.2f})")

# Let's test if receipts modify the per diem or mileage rate
print("\n\n=== TESTING RECEIPT EFFECT ON RATES ===")

# For case with 1 day, 140 miles
print("Case: 1 day, 140 miles")
print("  Low receipts ($22.71): Output=$199.68")
print("  High receipts ($255.99): Output=$150.34")
print("  Difference: $49.34")

# If per diem is constant, what mileage rates would explain this?
for pd in [85, 86, 87, 88, 89, 90]:
    mr_low = (199.68 - pd) / 140
    mr_high = (150.34 - pd) / 140
    print(f"\n  If per_diem=${pd}:")
    print(f"    Low receipts implies mileage rate: ${mr_low:.3f}/mi")
    print(f"    High receipts implies mileage rate: ${mr_high:.3f}/mi")
    print(f"    Difference: ${mr_low - mr_high:.3f}/mi")

# Look for a pattern in receipt effect
print("\n\n=== LOOKING FOR RECEIPT FORMULA ===")

# Collect all duplicate cases
duplicates = {}
for case in cases:
    key = (case['input']['trip_duration_days'], case['input']['miles_traveled'])
    if key not in duplicates:
        duplicates[key] = []
    duplicates[key].append((case['input']['total_receipts_amount'], case['expected_output']))

# Analyze cases with exactly 2 instances
print("Analyzing pairs with same days/miles:")
for (days, miles), instances in duplicates.items():
    if len(instances) == 2:
        r1, o1 = instances[0]
        r2, o2 = instances[1]
        
        # Order by receipt amount
        if r1 > r2:
            r1, o1, r2, o2 = r2, o2, r1, o1
        
        output_diff = o2 - o1
        receipt_diff = r2 - r1
        
        # Only analyze if significant receipt difference
        if receipt_diff > 100:
            ratio = output_diff / receipt_diff
            print(f"\n  Days={days}, Miles={miles}:")
            print(f"    Low: R=${r1:.2f}, O=${o1:.2f}")
            print(f"    High: R=${r2:.2f}, O=${o2:.2f}")
            print(f"    Output diff / Receipt diff = ${output_diff:.2f} / ${receipt_diff:.2f} = {ratio:.3f}")

# Test a specific hypothesis
print("\n\n=== TESTING HYPOTHESIS: RECEIPTS AFFECT OUTPUT NEGATIVELY ===")
print("Maybe high receipts REDUCE the reimbursement?")

# Check correlation
receipt_output_pairs = [(c['input']['total_receipts_amount'], c['expected_output']) for c in cases]
receipt_output_pairs.sort(key=lambda x: x[0])

# Sample every 100th case
print("\nSampling receipt vs output:")
for i in range(0, len(receipt_output_pairs), 100):
    r, o = receipt_output_pairs[i]
    print(f"  Receipts=${r:.2f}, Output=${o:.2f}")