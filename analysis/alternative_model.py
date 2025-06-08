#!/usr/bin/env python3
"""
Alternative model: What if high receipts actually REDUCE reimbursement?
Or what if there are multiple calculations and we take the minimum?
"""
import json
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def load_public_cases():
    """Load all public test cases"""
    with open('public_cases.json', 'r') as f:
        return json.load(f)

def analyze_receipt_effect(data):
    """Analyze the true effect of receipts on reimbursement"""
    # First, estimate base travel reimbursement (no receipts)
    no_receipt_cases = [c for c in data if c['input']['total_receipts_amount'] < 50]
    
    if len(no_receipt_cases) > 20:
        # Fit model for travel only
        X = []
        y = []
        for case in no_receipt_cases:
            X.append([case['input']['trip_duration_days'], 
                     case['input']['miles_traveled']])
            y.append(case['expected_output'] - case['input']['total_receipts_amount'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Simple linear regression
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(X, y)
        
        base_per_diem = reg.coef_[0]
        base_mileage = reg.coef_[1]
        base_intercept = reg.intercept_
        
        print(f"Base travel formula (from low-receipt cases):")
        print(f"  Per diem: ${base_per_diem:.2f}")
        print(f"  Mileage: ${base_mileage:.4f}")
        print(f"  Intercept: ${base_intercept:.2f}")
    else:
        # Fallback
        base_per_diem = 56
        base_mileage = 0.52
        base_intercept = 100
    
    # Now analyze receipt effect
    receipt_effects = []
    
    for case in data:
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        exp = case['expected_output']
        
        # Estimate base travel reimbursement
        base = base_per_diem * d + base_mileage * m + base_intercept
        
        # Receipt effect
        effect = exp - base
        
        receipt_effects.append({
            'receipts': r,
            'expected': exp,
            'base': base,
            'effect': effect,
            'effect_per_dollar': effect / r if r > 0 else 0,
            'days': d,
            'miles': m
        })
    
    # Sort by receipts
    receipt_effects.sort(key=lambda x: x['receipts'])
    
    # Plot receipt effect
    plt.figure(figsize=(12, 8))
    
    receipts = [e['receipts'] for e in receipt_effects]
    effects = [e['effect'] for e in receipt_effects]
    
    plt.subplot(2, 2, 1)
    plt.scatter(receipts, effects, alpha=0.5, s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Receipt Amount ($)')
    plt.ylabel('Receipt Effect on Total ($)')
    plt.title('Receipt Effect vs Receipt Amount')
    plt.grid(True, alpha=0.3)
    
    # Plot effect per dollar
    plt.subplot(2, 2, 2)
    effect_per_dollar = [e['effect_per_dollar'] for e in receipt_effects if e['receipts'] > 50]
    receipts_filtered = [e['receipts'] for e in receipt_effects if e['receipts'] > 50]
    
    plt.scatter(receipts_filtered, effect_per_dollar, alpha=0.5, s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Receipt Amount ($)')
    plt.ylabel('Receipt Effect per Dollar')
    plt.title('Receipt Effect Rate vs Receipt Amount')
    plt.ylim(-2, 2)
    plt.grid(True, alpha=0.3)
    
    # Plot expected vs base + receipts
    plt.subplot(2, 2, 3)
    base_plus_receipts = [e['base'] + e['receipts'] for e in receipt_effects]
    expected_vals = [e['expected'] for e in receipt_effects]
    
    plt.scatter(base_plus_receipts, expected_vals, alpha=0.5, s=10)
    plt.plot([0, 3000], [0, 3000], 'r--', label='y=x')
    plt.xlabel('Base + Receipts ($)')
    plt.ylabel('Expected Reimbursement ($)')
    plt.title('Expected vs (Base + Receipts)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histogram of receipt effects
    plt.subplot(2, 2, 4)
    plt.hist(effects, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Receipt Effect ($)')
    plt.ylabel('Count')
    plt.title('Distribution of Receipt Effects')
    
    plt.tight_layout()
    plt.savefig('analysis/receipt_effect_analysis.png')
    print("\nPlot saved to analysis/receipt_effect_analysis.png")
    
    return base_per_diem, base_mileage, base_intercept

def test_minimum_hypothesis(data, base_per_diem, base_mileage, base_intercept):
    """Test if reimbursement is the MINIMUM of multiple calculations"""
    print("\n\nTesting minimum hypothesis...")
    
    errors = []
    matches = 0
    
    for case in data:
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        exp = case['expected_output']
        
        # Different possible calculations
        calc1 = base_per_diem * d + base_mileage * m + base_intercept + r  # Full reimbursement
        calc2 = base_per_diem * d + base_mileage * m + base_intercept + 0.5 * r  # 50% receipts
        calc3 = base_per_diem * d + base_mileage * m + base_intercept  # No receipts
        calc4 = 100 * d + 0.58 * m  # Alternative formula from interviews
        calc5 = r  # Just receipts
        
        # Daily cap based on days
        daily_caps = {
            1: 1475, 2: 1550, 3: 1590, 4: 1700, 5: 1810,
            6: 1970, 7: 2070, 8: 1950, 9: 1950, 10: 2010,
            11: 2150, 12: 2160, 13: 2215, 14: 2340
        }
        
        cap = daily_caps.get(d, 99999)
        
        # Take minimum of various calculations
        options = [calc1, calc2, calc3, calc4, calc5, cap]
        predicted = min(options)
        
        error = abs(predicted - exp)
        errors.append(error)
        
        if error < 1:
            matches += 1
        
        if error > 500:  # Large errors
            print(f"  Case: {d}d, {m:.0f}mi, ${r:.2f}r")
            print(f"    Expected: ${exp:.2f}, Predicted: ${predicted:.2f}, Error: ${error:.2f}")
            print(f"    Options: {[f'${o:.2f}' for o in options]}")
    
    mae = np.mean(errors)
    print(f"\nMinimum hypothesis MAE: ${mae:.2f}")
    print(f"Matches within $1: {matches}")

def build_decision_tree_model(data):
    """Use decision tree to find exact rules"""
    print("\n\nBuilding deep decision tree...")
    
    X = []
    y = []
    
    for case in data:
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        
        # Add various features
        features = [
            d,  # days
            m,  # miles
            r,  # receipts
            d * m,  # interaction
            r / 100,  # receipt buckets
            1 if r > 828.10 else 0,  # threshold indicator
            min(r, 1000),  # capped receipts
            max(0, r - 1000),  # excess receipts
        ]
        
        X.append(features)
        y.append(case['expected_output'])
    
    X = np.array(X)
    y = np.array(y)
    
    # Build a deeper tree
    tree = DecisionTreeRegressor(max_depth=10, min_samples_split=10, min_samples_leaf=5)
    tree.fit(X, y)
    
    # Test
    predictions = tree.predict(X)
    mae = np.mean(np.abs(predictions - y))
    
    print(f"Decision tree MAE: ${mae:.2f}")
    print(f"Tree depth: {tree.get_depth()}")
    print(f"Number of leaves: {tree.get_n_leaves()}")
    
    # Find perfect predictions
    perfect = np.sum(np.abs(predictions - y) < 0.01)
    print(f"Perfect predictions: {perfect}/{len(data)}")
    
    return tree

def main():
    print("Loading public cases...")
    data = load_public_cases()
    
    print(f"\nAnalyzing alternative models for {len(data)} cases...")
    
    # Analyze receipt effects
    base_per_diem, base_mileage, base_intercept = analyze_receipt_effect(data)
    
    # Test minimum hypothesis
    test_minimum_hypothesis(data, base_per_diem, base_mileage, base_intercept)
    
    # Build decision tree
    tree = build_decision_tree_model(data)
    
    # Save tree for inspection
    from sklearn.tree import export_text
    tree_rules = export_text(tree, feature_names=[
        'days', 'miles', 'receipts', 'days*miles', 
        'receipt_bucket', 'above_828', 'capped_receipts', 'excess_receipts'
    ])
    
    with open('analysis/decision_tree_rules.txt', 'w') as f:
        f.write(tree_rules)
    
    print("\nDecision tree rules saved to analysis/decision_tree_rules.txt")

if __name__ == '__main__':
    main()