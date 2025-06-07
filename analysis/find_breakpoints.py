#!/usr/bin/env python3
import json
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import sys

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

# Extract data
X = []
y = []

for case in cases:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    
    X.append([days, miles, receipts])
    y.append(output)

X = np.array(X)
y = np.array(y)

# Build decision tree to find breakpoints
print("=== DECISION TREE ANALYSIS ===")
for max_depth in [2, 3, 4]:
    print(f"\nMax depth = {max_depth}:")
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=50, min_samples_leaf=20)
    tree.fit(X, y)
    
    # Print tree structure
    def print_tree_rules(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != -2 else "undefined!"
            for i in tree_.feature
        ]
        
        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != -2:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                print(f"{indent}if {name} <= {threshold:.2f}:")
                recurse(tree_.children_left[node], depth + 1)
                print(f"{indent}else:  # if {name} > {threshold:.2f}")
                recurse(tree_.children_right[node], depth + 1)
            else:
                print(f"{indent}value = {tree_.value[node][0][0]:.2f} (n={tree_.n_node_samples[node]})")
        
        recurse(0, 0)
    
    print_tree_rules(tree, ['days', 'miles', 'receipts'])
    
    # Calculate RMSE
    predictions = tree.predict(X)
    rmse = np.sqrt(np.mean((y - predictions) ** 2))
    print(f"RMSE: ${rmse:.2f}")

# Analyze mileage rates more precisely
print("\n=== DETAILED MILEAGE RATE ANALYSIS ===")
# Focus on single-day trips to isolate mileage effect
single_day_cases = [(m, r, o) for m, r, o, d in zip(X[:, 1], X[:, 2], y, X[:, 0]) if d == 1]
if single_day_cases:
    sorted_cases = sorted(single_day_cases, key=lambda x: x[0])
    
    # Look for rate changes
    print("Looking for mileage rate breakpoints...")
    window_size = 20
    for i in range(window_size, len(sorted_cases) - window_size):
        miles_center = sorted_cases[i][0]
        
        # Calculate rates before and after
        before_rates = []
        after_rates = []
        
        for j in range(i - window_size, i):
            m1, r1, o1 = sorted_cases[j]
            # Estimate mileage component by subtracting base per diem and receipts
            mileage_component = o1 - 100 - r1
            if m1 > 0:
                before_rates.append(mileage_component / m1)
        
        for j in range(i, i + window_size):
            m2, r2, o2 = sorted_cases[j]
            mileage_component = o2 - 100 - r2
            if m2 > 0:
                after_rates.append(mileage_component / m2)
        
        if before_rates and after_rates:
            avg_before = np.mean(before_rates)
            avg_after = np.mean(after_rates)
            
            # Check for significant rate change
            if abs(avg_before - avg_after) > 0.05:
                print(f"  Potential breakpoint at {miles_center:.0f} miles: ${avg_before:.3f}/mi -> ${avg_after:.3f}/mi")

# Analyze receipt patterns
print("\n=== RECEIPT PATTERN ANALYSIS ===")
# Look at receipt reimbursement ratio by amount
receipt_buckets = {}
for i in range(len(cases)):
    receipts = X[i][2]
    days = X[i][0]
    miles = X[i][1]
    output = y[i]
    
    # Bucket receipts by $50 intervals
    bucket = int(receipts // 50) * 50
    if bucket not in receipt_buckets:
        receipt_buckets[bucket] = []
    
    # Estimate receipt component
    base_component = 100 * days + 0.55 * miles  # Rough estimate
    receipt_component = output - base_component
    receipt_ratio = receipt_component / receipts if receipts > 0 else 0
    
    receipt_buckets[bucket].append((receipts, receipt_ratio))

# Print average ratios by bucket
print("\nReceipt reimbursement ratios by amount:")
for bucket in sorted(receipt_buckets.keys()):
    if receipt_buckets[bucket]:
        ratios = [r[1] for r in receipt_buckets[bucket]]
        avg_ratio = np.mean(ratios)
        print(f"  ${bucket}-${bucket+50}: avg ratio = {avg_ratio:.3f}")