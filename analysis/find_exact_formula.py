#!/usr/bin/env python3
import json
import numpy as np

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

# Extract data
data = []
for case in cases:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    data.append((days, miles, receipts, output))

# Sort by output to look for patterns
data.sort(key=lambda x: x[3])

print("=== ANALYZING BASE PER DIEM ===")
# Look at cases with minimal miles and receipts
low_mile_cases = [d for d in data if d[1] < 50 and d[2] < 50]
if low_mile_cases:
    print("Cases with low miles and receipts:")
    for d, m, r, o in low_mile_cases[:10]:
        base = o - 0.55 * m - r  # Assuming some mileage rate
        per_diem = base / d
        print(f"  Days={d}, Miles={m:.1f}, Receipts=${r:.2f}, Output=${o:.2f}, Implied per diem=${per_diem:.2f}")

print("\n=== ANALYZING MILEAGE RATES ===")
# Look at single-day trips to isolate mileage effect
single_day = [(m, r, o) for d, m, r, o in data if d == 1]
single_day.sort(key=lambda x: x[0])  # Sort by miles

print("Single-day trips (first 20):")
for i, (m, r, o) in enumerate(single_day[:20]):
    # Assuming base per diem of ~100
    mileage_component = o - 100 - r
    rate = mileage_component / m if m > 0 else 0
    print(f"  Miles={m:.1f}, Receipts=${r:.2f}, Output=${o:.2f}, Implied rate=${rate:.3f}/mi")

# Look for mileage breakpoints more systematically
print("\n=== FINDING EXACT MILEAGE BREAKPOINTS ===")
# Group by mileage bands and calculate average rates
mileage_bands = {}
for d, m, r, o in data:
    band = int(m // 50) * 50  # 50-mile bands
    if band not in mileage_bands:
        mileage_bands[band] = []
    
    # Estimate mileage component (assuming $100/day per diem)
    estimated_base = 100 * d + r
    mileage_component = o - estimated_base
    implied_rate = mileage_component / m if m > 0 else 0
    mileage_bands[band].append((m, implied_rate, d, r, o))

print("Average mileage rates by band:")
for band in sorted(mileage_bands.keys()):
    if mileage_bands[band]:
        rates = [x[1] for x in mileage_bands[band]]
        avg_rate = np.mean(rates)
        print(f"  {band}-{band+50} miles: {len(rates)} cases, avg rate=${avg_rate:.3f}/mi")

print("\n=== ANALYZING RECEIPT HANDLING ===")
# Look at how receipts affect reimbursement
print("Looking for receipt threshold...")

# Group cases by receipt amount
receipt_groups = {}
for d, m, r, o in data:
    # Estimate non-receipt components
    base_estimate = 100 * d + 0.55 * m  # Rough estimate
    receipt_effect = o - base_estimate
    
    bucket = int(r // 100) * 100
    if bucket not in receipt_groups:
        receipt_groups[bucket] = []
    receipt_groups[bucket].append((r, receipt_effect, receipt_effect/r if r > 0 else 0))

print("\nReceipt effect by amount:")
for bucket in sorted(receipt_groups.keys()):
    if receipt_groups[bucket]:
        effects = [x[1] for x in receipt_groups[bucket]]
        ratios = [x[2] for x in receipt_groups[bucket]]
        avg_effect = np.mean(effects)
        avg_ratio = np.mean(ratios)
        print(f"  ${bucket}-${bucket+100}: avg effect=${avg_effect:.2f}, avg ratio={avg_ratio:.3f}")

# Look for specific thresholds
print("\n=== TESTING SPECIFIC HYPOTHESES ===")

# Test hypothesis: Base per diem is exactly $100
per_diem_test = []
for d, m, r, o in data:
    if m < 100 and r < 100:  # Low other components
        implied_per_diem = (o - 0.55 * m - r) / d
        per_diem_test.append(implied_per_diem)

if per_diem_test:
    print(f"Average implied per diem (low miles/receipts cases): ${np.mean(per_diem_test):.2f}")
    print(f"Std dev: ${np.std(per_diem_test):.2f}")

# Test for integer rounding patterns
print("\n=== CHECKING FOR ROUNDING PATTERNS ===")
output_decimals = [o - int(o) for _, _, _, o in data]
decimal_counts = {}
for dec in output_decimals:
    dec_str = f"{dec:.2f}"
    if dec_str not in decimal_counts:
        decimal_counts[dec_str] = 0
    decimal_counts[dec_str] += 1

print("Most common output decimal parts:")
for dec, count in sorted(decimal_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  .{dec[2:]}: {count} times")

# Look for specific quirks or bugs
print("\n=== LOOKING FOR QUIRKS/BUGS ===")
# Check if certain day counts have special treatment
day_averages = {}
for d, m, r, o in data:
    if d not in day_averages:
        day_averages[d] = []
    # Normalize by subtracting estimated components
    normalized = o - 0.55 * m - r
    day_averages[d].append(normalized / d)

print("Average per diem by trip length:")
for days in sorted(day_averages.keys()):
    if day_averages[days]:
        avg = np.mean(day_averages[days])
        count = len(day_averages[days])
        print(f"  {days} days: ${avg:.2f}/day (n={count})")