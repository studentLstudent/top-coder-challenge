#!/usr/bin/env python3
import json
import numpy as np

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== FINDING MAGIC NUMBERS PER THE BIG PLAN PART 2 ===")
print("Method: Sort unique values, compute first-differences, find sharp jumps\n")

# Extract data
data = []
for case in cases:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    data.append((days, miles, receipts, output))

# Function to find breakpoints for a given feature
def find_breakpoints(feature_idx, feature_name):
    print(f"\n=== ANALYZING {feature_name.upper()} ===")
    
    # Sort by feature value
    sorted_data = sorted(data, key=lambda x: x[feature_idx])
    
    # Group by unique feature values
    unique_values = {}
    for item in sorted_data:
        key = item[feature_idx]
        if key not in unique_values:
            unique_values[key] = []
        unique_values[key].append(item[3])  # output
    
    # Calculate average output for each unique value
    avg_outputs = []
    for key in sorted(unique_values.keys()):
        outputs = unique_values[key]
        avg = sum(outputs) / len(outputs)
        avg_outputs.append((key, avg, len(outputs)))
    
    # Compute first differences
    print(f"\nFirst differences in average output by {feature_name}:")
    diffs = []
    for i in range(1, len(avg_outputs)):
        val1, out1, count1 = avg_outputs[i-1]
        val2, out2, count2 = avg_outputs[i]
        
        diff = out2 - out1
        val_diff = val2 - val1
        normalized_diff = diff / val_diff if val_diff > 0 else 0
        
        diffs.append((val1, val2, diff, normalized_diff))
        
        # Print significant jumps
        if abs(normalized_diff) > 10:  # $10 per unit change
            print(f"  {feature_name} {val1}->{val2}: ${diff:.2f} total, ${normalized_diff:.2f}/unit")
    
    # Find the sharpest jumps
    sorted_diffs = sorted(diffs, key=lambda x: abs(x[3]), reverse=True)
    print(f"\nTop 5 sharpest jumps in {feature_name}:")
    for i in range(min(5, len(sorted_diffs))):
        val1, val2, diff, norm = sorted_diffs[i]
        print(f"  Between {val1} and {val2}: ${norm:.2f} per unit")

# Analyze each feature
find_breakpoints(0, "days")
find_breakpoints(1, "miles")
find_breakpoints(2, "receipts")

# Special analysis for receipts around known breakpoint
print("\n\n=== SPECIAL ANALYSIS: RECEIPTS AROUND $828.10 ===")
receipts_near_828 = []
for d, m, r, o in data:
    if 820 <= r <= 835:
        receipts_near_828.append((r, o, d, m))

receipts_near_828.sort()
print("Receipts near $828.10:")
for r, o, d, m in receipts_near_828:
    print(f"  R=${r:.2f}, Output=${o:.2f} (D={d}, M={m})")

# Analyze output patterns for specific feature values
print("\n\n=== ANALYZING OUTPUT PATTERNS ===")

# Check if there are "magic" day counts
print("\nOutputs by day count:")
days_outputs = {}
for d, m, r, o in data:
    if d not in days_outputs:
        days_outputs[d] = []
    days_outputs[d].append(o)

for day in sorted(days_outputs.keys()):
    outputs = days_outputs[day]
    avg = sum(outputs) / len(outputs)
    min_o = min(outputs)
    max_o = max(outputs)
    print(f"  {day} days: avg=${avg:.2f}, range=${min_o:.2f}-${max_o:.2f} (n={len(outputs)})")

# Look for specific mileage breakpoints
print("\n\nChecking common mileage breakpoints:")
common_breakpoints = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for bp in common_breakpoints:
    below = [o for d, m, r, o in data if bp-10 <= m < bp]
    above = [o for d, m, r, o in data if bp < m <= bp+10]
    
    if below and above:
        avg_below = sum(below) / len(below)
        avg_above = sum(above) / len(above)
        diff = avg_above - avg_below
        print(f"  Around {bp} miles: below=${avg_below:.2f}, above=${avg_above:.2f}, diff=${diff:.2f}")

# Analyze receipts more carefully
print("\n\nReceipt value distribution:")
receipt_values = sorted(set(r for _, _, r, _ in data))
print(f"Total unique receipt values: {len(receipt_values)}")
print(f"Min: ${min(receipt_values):.2f}, Max: ${max(receipt_values):.2f}")

# Check for patterns in receipt decimal parts
print("\nReceipt decimal parts analysis:")
decimal_counts = {}
for _, _, r, _ in data:
    decimal = int(round((r - int(r)) * 100))
    if decimal not in decimal_counts:
        decimal_counts[decimal] = 0
    decimal_counts[decimal] += 1

most_common = sorted(decimal_counts.items(), key=lambda x: -x[1])[:10]
print("Most common receipt decimal parts (cents):")
for cents, count in most_common:
    print(f"  .{cents:02d}: {count} times")