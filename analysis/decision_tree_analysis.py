#!/usr/bin/env python3
import json
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

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

print("=== DECISION TREE ANALYSIS PER THE BIG PLAN ===")
print("Finding breakpoints for piecewise linear formula...\n")

# Build decision tree with max_depth=3 as specified in THE BIG PLAN
tree = DecisionTreeRegressor(max_depth=3, min_samples_split=20, min_samples_leaf=10)
tree.fit(X, y)

# Extract the tree structure
def extract_tree_rules(tree, feature_names):
    tree_ = tree.tree_
    rules = []
    
    def recurse(node, depth, conditions):
        if tree_.feature[node] != -2:  # Not a leaf
            feature = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            
            # Left branch
            left_conditions = conditions + [(feature, '<=', threshold)]
            recurse(tree_.children_left[node], depth + 1, left_conditions)
            
            # Right branch  
            right_conditions = conditions + [(feature, '>', threshold)]
            recurse(tree_.children_right[node], depth + 1, right_conditions)
        else:  # Leaf node
            value = tree_.value[node][0][0]
            n_samples = tree_.n_node_samples[node]
            rules.append((conditions, value, n_samples))
    
    recurse(0, 0, [])
    return rules

rules = extract_tree_rules(tree, ['days', 'miles', 'receipts'])

print("DISCOVERED REGIONS:")
for i, (conditions, value, n_samples) in enumerate(rules):
    print(f"\nRegion {i+1} (n={n_samples}):")
    for feature, op, threshold in conditions:
        print(f"  {feature} {op} {threshold:.2f}")
    print(f"  Average output: ${value:.2f}")

# Now fit linear regression in each region
print("\n\n=== COEFFICIENT SEARCH IN EACH REGION ===")

for i, (conditions, _, n_samples) in enumerate(rules):
    # Find cases in this region
    mask = np.ones(len(X), dtype=bool)
    for feature, op, threshold in conditions:
        if feature == 'days':
            col = 0
        elif feature == 'miles':
            col = 1
        else:  # receipts
            col = 2
            
        if op == '<=':
            mask = mask & (X[:, col] <= threshold)
        else:
            mask = mask & (X[:, col] > threshold)
    
    if np.sum(mask) > 10:  # Need enough samples
        X_region = X[mask]
        y_region = y[mask]
        
        # Fit linear model
        lr = LinearRegression()
        lr.fit(X_region, y_region)
        
        print(f"\nRegion {i+1} coefficients:")
        print(f"  Days coefficient: ${lr.coef_[0]:.2f}")
        print(f"  Miles coefficient: ${lr.coef_[1]:.4f}")
        print(f"  Receipts coefficient: ${lr.coef_[2]:.4f}")
        print(f"  Intercept: ${lr.intercept_:.2f}")
        
        # Calculate RMSE for this region
        predictions = lr.predict(X_region)
        rmse = np.sqrt(np.mean((y_region - predictions) ** 2))
        print(f"  Region RMSE: ${rmse:.2f}")

# Test financially plausible rounding
print("\n\n=== TESTING FINANCIALLY PLAUSIBLE COEFFICIENTS ===")

# Common financial rates to test
per_diem_rates = [85, 86, 87, 90, 100]
mileage_rates = [0.50, 0.55, 0.56, 0.57, 0.58, 0.60]

best_combo = None
best_error = float('inf')

for pd in per_diem_rates:
    for mr in mileage_rates:
        # Simple test without receipts first
        total_error = 0
        for j in range(len(X)):
            predicted = pd * X[j, 0] + mr * X[j, 1]
            total_error += (y[j] - predicted) ** 2
        
        rmse = np.sqrt(total_error / len(X))
        if rmse < best_error:
            best_error = rmse
            best_combo = (pd, mr)

print(f"Best simple combo (no receipts): ${best_combo[0]}/day, ${best_combo[1]}/mile")
print(f"RMSE: ${best_error:.2f}")