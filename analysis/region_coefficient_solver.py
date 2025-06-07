#!/usr/bin/env python3
"""
Region-based coefficient solver using integer least-squares
Following THE-BIG-PLAN-PART-2 strategy for finding exact coefficients
"""
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def load_public_cases():
    """Load all public test cases"""
    with open('public_cases.json', 'r') as f:
        return json.load(f)

def build_decision_tree(data):
    """Build decision tree to find regions"""
    X = np.array([[case['input']['trip_duration_days'], 
                   case['input']['miles_traveled'], 
                   case['input']['total_receipts_amount']] for case in data])
    y = np.array([case['expected_output'] for case in data])
    
    # Use same parameters as decision_tree_analysis.py
    tree = DecisionTreeRegressor(max_depth=3, min_samples_split=20, min_samples_leaf=10)
    tree.fit(X, y)
    
    return tree, X, y

def extract_regions(tree, X):
    """Extract regions from decision tree"""
    leaf_ids = tree.apply(X)
    unique_leaves = np.unique(leaf_ids)
    
    regions = {}
    for leaf_id in unique_leaves:
        mask = leaf_ids == leaf_id
        regions[leaf_id] = mask
    
    return regions

def solve_coefficients_for_region(X_region, y_region, round_to_finance=True):
    """
    Solve coefficients for a specific region
    X_region: [days, miles, receipts]
    y_region: expected outputs
    """
    # Add intercept column
    X_with_intercept = np.column_stack([X_region, np.ones(len(X_region))])
    
    # Solve using linear regression
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_with_intercept, y_region)
    
    coeffs = reg.coef_
    
    if round_to_finance:
        # Round to financially plausible values
        # Per diem: round to nearest dollar (85, 86, 87, etc.)
        coeffs[0] = round(coeffs[0])
        
        # Mileage: round to nearest 0.05 (0.55, 0.60, etc.)
        coeffs[1] = round(coeffs[1] * 20) / 20
        
        # Receipts: round to 0.01
        coeffs[2] = round(coeffs[2], 2)
        
        # Intercept: round to nearest cent
        coeffs[3] = round(coeffs[3], 2)
    
    return coeffs

def test_coefficients(data, regions, coefficients):
    """Test coefficients on all data and compute MAE"""
    X = np.array([[case['input']['trip_duration_days'], 
                   case['input']['miles_traveled'], 
                   case['input']['total_receipts_amount']] for case in data])
    y = np.array([case['expected_output'] for case in data])
    
    predictions = np.zeros(len(data))
    
    for leaf_id, mask in regions.items():
        if leaf_id not in coefficients:
            continue
            
        X_region = X[mask]
        coeffs = coefficients[leaf_id]
        
        # Predict: days*c0 + miles*c1 + receipts*c2 + c3
        pred = (X_region[:, 0] * coeffs[0] + 
                X_region[:, 1] * coeffs[1] + 
                X_region[:, 2] * coeffs[2] + 
                coeffs[3])
        
        predictions[mask] = pred
    
    # Compute MAE
    mae = np.mean(np.abs(predictions - y))
    
    # Find worst cases
    errors = np.abs(predictions - y)
    worst_indices = np.argsort(errors)[-10:]
    
    return mae, predictions, worst_indices

def main():
    print("Loading public cases...")
    data = load_public_cases()
    
    print(f"Building decision tree to find regions...")
    tree, X, y = build_decision_tree(data)
    
    print(f"Tree found {tree.get_n_leaves()} regions")
    
    regions = extract_regions(tree, X)
    
    print("\nSolving coefficients for each region:")
    coefficients = {}
    
    for leaf_id, mask in regions.items():
        X_region = X[mask]
        y_region = y[mask]
        
        print(f"\nRegion {leaf_id}: {mask.sum()} cases")
        
        # First try without rounding
        coeffs = solve_coefficients_for_region(X_region, y_region, round_to_finance=False)
        print(f"  Raw coefficients: days={coeffs[0]:.4f}, miles={coeffs[1]:.4f}, receipts={coeffs[2]:.4f}, intercept={coeffs[3]:.4f}")
        
        # Then round to finance values
        coeffs_rounded = solve_coefficients_for_region(X_region, y_region, round_to_finance=True)
        print(f"  Rounded coefficients: days={coeffs_rounded[0]:.2f}, miles={coeffs_rounded[1]:.2f}, receipts={coeffs_rounded[2]:.2f}, intercept={coeffs_rounded[3]:.2f}")
        
        coefficients[leaf_id] = coeffs_rounded
    
    print("\nTesting coefficients on full dataset...")
    mae, predictions, worst_indices = test_coefficients(data, regions, coefficients)
    
    print(f"\nOverall MAE: ${mae:.2f}")
    
    print("\nWorst 10 cases:")
    for idx in worst_indices:
        case = data[idx]
        pred = predictions[idx]
        error = abs(pred - case['expected_output'])
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        print(f"  Case {idx}: {days}d, {miles}mi, ${receipts:.2f}r -> Expected: ${case['expected_output']:.2f}, Predicted: ${pred:.2f}, Error: ${error:.2f}")
    
    # Save results
    results = {
        'regions': {str(k): v.tolist() for k, v in regions.items()},
        'coefficients': {str(k): v.tolist() for k, v in coefficients.items()},
        'mae': mae
    }
    
    with open('analysis/region_coefficients.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to analysis/region_coefficients.json")

if __name__ == '__main__':
    main()