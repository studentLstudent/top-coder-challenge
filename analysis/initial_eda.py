#!/usr/bin/env python3
import json
import statistics
import math

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

# Extract data
days = []
miles = []
receipts = []
outputs = []

for case in cases:
    days.append(case['input']['trip_duration_days'])
    miles.append(case['input']['miles_traveled'])
    receipts.append(case['input']['total_receipts_amount'])
    outputs.append(case['expected_output'])

print("=== BASIC STATISTICS ===")
print(f"Total cases: {len(cases)}")
print(f"\nDays: min={min(days)}, max={max(days)}, mean={statistics.mean(days):.2f}")
print(f"Miles: min={min(miles)}, max={max(miles)}, mean={statistics.mean(miles):.2f}")
print(f"Receipts: min=${min(receipts):.2f}, max=${max(receipts):.2f}, mean=${statistics.mean(receipts):.2f}")
print(f"Output: min=${min(outputs):.2f}, max=${max(outputs):.2f}, mean=${statistics.mean(outputs):.2f}")

# Look for simple linear relationships
print("\n=== SIMPLE LINEAR ANALYSIS ===")

# Try basic linear combination
total_squared_error = 0
for i in range(len(cases)):
    # Simple baseline: $100/day + $0.50/mile + receipts
    predicted = 100 * days[i] + 0.50 * miles[i] + receipts[i]
    error = outputs[i] - predicted
    total_squared_error += error * error
    
rmse = math.sqrt(total_squared_error / len(cases))
print(f"RMSE for baseline (100*days + 0.50*miles + receipts): ${rmse:.2f}")

# Check for obvious patterns by sorting
print("\n=== CHECKING FOR PATTERNS ===")

# Sort by days and check for per diem pattern
sorted_by_days = sorted(zip(days, outputs), key=lambda x: x[0])
print("\nPer diem analysis (first 10 cases sorted by days):")
for i in range(min(10, len(sorted_by_days))):
    d, out = sorted_by_days[i]
    print(f"  Days: {d}, Output: ${out:.2f}, Implied per diem: ${out/d:.2f}")

# Look for cases with same inputs
print("\n=== CHECKING FOR DETERMINISTIC BEHAVIOR ===")
input_map = {}
for i, case in enumerate(cases):
    key = (days[i], miles[i], receipts[i])
    if key in input_map:
        print(f"Found duplicate inputs: {key}")
        print(f"  Output 1: ${input_map[key]:.2f}")
        print(f"  Output 2: ${outputs[i]:.2f}")
    else:
        input_map[key] = outputs[i]

# Check for mileage tiers
print("\n=== MILEAGE ANALYSIS ===")
# Group by mileage ranges
mileage_ranges = [(0, 100), (100, 300), (300, 500), (500, 1000), (1000, 2000)]
for low, high in mileage_ranges:
    cases_in_range = [(m, o, d) for m, o, d in zip(miles, outputs, days) if low <= m < high]
    if cases_in_range:
        avg_miles = statistics.mean([c[0] for c in cases_in_range])
        avg_output = statistics.mean([c[1] for c in cases_in_range])
        avg_days = statistics.mean([c[2] for c in cases_in_range])
        # Subtract estimated per diem to isolate mileage component
        mileage_component = (avg_output - 100 * avg_days) / avg_miles if avg_miles > 0 else 0
        print(f"  Miles {low}-{high}: {len(cases_in_range)} cases, implied rate: ${mileage_component:.3f}/mile")

# Check for receipt thresholds
print("\n=== RECEIPT ANALYSIS ===")
receipt_ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000)]
for low, high in receipt_ranges:
    cases_in_range = [(r, o, d, m) for r, o, d, m in zip(receipts, outputs, days, miles) if low <= r < high]
    if cases_in_range:
        avg_receipts = statistics.mean([c[0] for c in cases_in_range])
        avg_output = statistics.mean([c[1] for c in cases_in_range])
        avg_days = statistics.mean([c[2] for c in cases_in_range])
        avg_miles = statistics.mean([c[3] for c in cases_in_range])
        # Subtract estimated per diem and mileage to see receipt treatment
        receipt_component = avg_output - (100 * avg_days + 0.50 * avg_miles)
        print(f"  Receipts ${low}-${high}: {len(cases_in_range)} cases")
        print(f"    Avg receipts: ${avg_receipts:.2f}, Receipt component: ${receipt_component:.2f}")
        print(f"    Ratio: {receipt_component/avg_receipts:.3f}" if avg_receipts > 0 else "    Ratio: N/A")