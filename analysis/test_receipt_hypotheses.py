#!/usr/bin/env python3
import json
import numpy as np

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== TESTING RECEIPT HYPOTHESES PER THE BIG PLAN PART 2 ===")

# Extract data
data = []
for case in cases:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    output = case['expected_output']
    data.append((days, miles, receipts, output))

# Base formula parameters (from earlier analysis)
base_per_diem = 86
base_mileage_rate = 0.55

def calculate_base(days, miles):
    """Calculate base amount without receipts"""
    return base_per_diem * days + base_mileage_rate * miles

# Hypothesis 1: Cap then percentage: min(R, K) + α·max(0, R–K)
print("\n=== TESTING HYPOTHESIS 1: CAP THEN PERCENTAGE ===")
print("Formula: receipt_contribution = min(R, K) + α·max(0, R–K)")

best_k = None
best_alpha = None
best_error_h1 = float('inf')

# Grid search
k_values = range(0, 1000, 50)  # Cap values to test
alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for k in k_values:
    for alpha in alpha_values:
        total_error = 0
        
        for days, miles, receipts, expected in data:
            base = calculate_base(days, miles)
            
            # Apply hypothesis 1
            receipt_contribution = min(receipts, k) + alpha * max(0, receipts - k)
            predicted = base + receipt_contribution
            
            error = abs(predicted - expected)
            total_error += error
        
        avg_error = total_error / len(data)
        
        if avg_error < best_error_h1:
            best_error_h1 = avg_error
            best_k = k
            best_alpha = alpha

print(f"Best parameters: K=${best_k}, α={best_alpha}")
print(f"Average error: ${best_error_h1:.2f}")

# Test on some examples
print("\nTesting on sample cases:")
for i in range(5):
    days, miles, receipts, expected = data[i]
    base = calculate_base(days, miles)
    receipt_contribution = min(receipts, best_k) + best_alpha * max(0, receipts - best_k)
    predicted = base + receipt_contribution
    print(f"  Case {i}: R=${receipts:.2f}, Expected=${expected:.2f}, Predicted=${predicted:.2f}, Error=${abs(expected-predicted):.2f}")

# Hypothesis 2: Quadratic penalty for low receipts
print("\n\n=== TESTING HYPOTHESIS 2: QUADRATIC PENALTY FOR LOW RECEIPTS ===")
print("Formula: if R < L: receipt_contribution = R - β·(L–R)²")

best_l = None
best_beta = None
best_error_h2 = float('inf')

# Grid search
l_values = range(100, 1000, 50)  # Threshold values
beta_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]

for l in l_values:
    for beta in beta_values:
        total_error = 0
        
        for days, miles, receipts, expected in data:
            base = calculate_base(days, miles)
            
            # Apply hypothesis 2
            if receipts < l:
                receipt_contribution = receipts - beta * (l - receipts) ** 2
            else:
                receipt_contribution = receipts  # No penalty above threshold
            
            predicted = base + receipt_contribution
            error = abs(predicted - expected)
            total_error += error
        
        avg_error = total_error / len(data)
        
        if avg_error < best_error_h2:
            best_error_h2 = avg_error
            best_l = l
            best_beta = beta

print(f"Best parameters: L=${best_l}, β={best_beta}")
print(f"Average error: ${best_error_h2:.2f}")

# Test on some examples
print("\nTesting on sample cases:")
for i in range(5):
    days, miles, receipts, expected = data[i]
    base = calculate_base(days, miles)
    
    if receipts < best_l:
        receipt_contribution = receipts - best_beta * (best_l - receipts) ** 2
    else:
        receipt_contribution = receipts
    
    predicted = base + receipt_contribution
    print(f"  Case {i}: R=${receipts:.2f}, Expected=${expected:.2f}, Predicted=${predicted:.2f}, Error=${abs(expected-predicted):.2f}")

# Hypothesis 3: Combination - different handling for different receipt ranges
print("\n\n=== TESTING HYPOTHESIS 3: PIECEWISE RECEIPT HANDLING ===")

# Based on earlier analysis, test if receipts have different effects in different ranges
ranges = [(0, 100), (100, 500), (500, 1000), (1000, float('inf'))]

# Calculate average receipt effect for each range
for low, high in ranges:
    in_range = []
    for days, miles, receipts, expected in data:
        if low <= receipts < high:
            base = calculate_base(days, miles)
            receipt_effect = expected - base
            in_range.append((receipts, receipt_effect))
    
    if in_range:
        # Fit linear model for this range
        if len(in_range) > 1:
            receipts_arr = np.array([r for r, _ in in_range])
            effects_arr = np.array([e for _, e in in_range])
            
            # Linear regression
            A = np.vstack([receipts_arr, np.ones(len(receipts_arr))]).T
            slope, intercept = np.linalg.lstsq(A, effects_arr, rcond=None)[0]
            
            print(f"\nRange ${low}-${high}: {len(in_range)} cases")
            print(f"  Linear fit: effect = {slope:.4f} * receipts + {intercept:.2f}")
            
            # Calculate R²
            predicted = slope * receipts_arr + intercept
            ss_res = np.sum((effects_arr - predicted) ** 2)
            ss_tot = np.sum((effects_arr - np.mean(effects_arr)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            print(f"  R² = {r_squared:.3f}")

# Test a more complex formula based on observations
print("\n\n=== TESTING COMPLEX FORMULA ===")

def complex_receipt_function(receipts):
    """Complex receipt handling based on analysis"""
    if receipts < 100:
        # Heavy penalty for small receipts
        return -200 + 2 * receipts
    elif receipts < 500:
        # Gradual improvement
        return -50 + 0.7 * receipts
    elif receipts < 1000:
        # Near break-even
        return 0.9 * receipts - 100
    else:
        # Partial reimbursement for high amounts
        return 0.5 * receipts + 300

total_error = 0
exact_matches = 0

print("Testing complex formula on all cases...")
for i, (days, miles, receipts, expected) in enumerate(data):
    base = calculate_base(days, miles)
    receipt_contribution = complex_receipt_function(receipts)
    predicted = base + receipt_contribution
    
    error = abs(predicted - expected)
    total_error += error
    
    if error < 0.01:
        exact_matches += 1
    
    if i < 10:
        print(f"  Case {i}: R=${receipts:.2f}, Expected=${expected:.2f}, Predicted=${predicted:.2f}, Error=${error:.2f}")

avg_error = total_error / len(data)
print(f"\nAverage error: ${avg_error:.2f}")
print(f"Exact matches: {exact_matches}/{len(data)}")