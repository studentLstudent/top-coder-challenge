#!/usr/bin/env python3
"""
Daily cap model - reimbursement is capped based on number of days
Key insight: High receipt cases show plateaus that depend on trip duration
"""
import json
import numpy as np
from sklearn.linear_model import LinearRegression

def load_public_cases():
    """Load all public test cases"""
    with open('public_cases.json', 'r') as f:
        return json.load(f)

def analyze_caps_by_days(data):
    """Find the cap pattern by analyzing high-receipt cases"""
    # Group by days
    by_days = {}
    for case in data:
        d = case['input']['trip_duration_days']
        r = case['input']['total_receipts_amount']
        exp = case['expected_output']
        
        if d not in by_days:
            by_days[d] = []
        by_days[d].append((r, exp))
    
    # Find caps for each day count
    caps = {}
    print("Analyzing caps by trip duration:")
    
    for days in sorted(by_days.keys()):
        cases = by_days[days]
        # Focus on high receipt cases (>$1500) to find the cap
        high_receipt_cases = [(r, exp) for r, exp in cases if r > 1500]
        
        if high_receipt_cases:
            # Find the maximum reimbursement for high receipt cases
            max_reimb = max(exp for r, exp in high_receipt_cases)
            avg_reimb = np.mean([exp for r, exp in high_receipt_cases])
            
            # Look at the 90th percentile to avoid outliers
            exps = sorted([exp for r, exp in high_receipt_cases])
            p90 = exps[int(len(exps) * 0.9)] if len(exps) > 10 else max_reimb
            
            print(f"  {days} days: {len(high_receipt_cases)} high-receipt cases")
            print(f"    Max: ${max_reimb:.2f}, Avg: ${avg_reimb:.2f}, 90th percentile: ${p90:.2f}")
            
            caps[days] = p90
    
    return caps

def build_capped_model(data, caps):
    """Build a model with daily caps"""
    # Extract features
    days = np.array([c['input']['trip_duration_days'] for c in data])
    miles = np.array([c['input']['miles_traveled'] for c in data])
    receipts = np.array([c['input']['total_receipts_amount'] for c in data])
    expected = np.array([c['expected_output'] for c in data])
    
    # First, let's find base rates for low-receipt cases
    low_receipt_mask = receipts < 500
    
    if low_receipt_mask.sum() > 50:
        # Build model for low receipt cases
        X_low = np.column_stack([days[low_receipt_mask], 
                                 miles[low_receipt_mask], 
                                 receipts[low_receipt_mask],
                                 np.ones(low_receipt_mask.sum())])
        y_low = expected[low_receipt_mask]
        
        reg_low = LinearRegression(fit_intercept=False)
        reg_low.fit(X_low, y_low)
        
        print("\nCoefficients from low-receipt cases:")
        print(f"  Per diem: ${reg_low.coef_[0]:.2f}")
        print(f"  Mileage: ${reg_low.coef_[1]:.4f}")
        print(f"  Receipt rate: {reg_low.coef_[2]:.4f}")
        print(f"  Intercept: ${reg_low.coef_[3]:.2f}")
        
        base_per_diem = reg_low.coef_[0]
        base_mileage = reg_low.coef_[1]
        base_receipt_rate = reg_low.coef_[2]
        base_intercept = reg_low.coef_[3]
    else:
        # Fallback values
        base_per_diem = 86.0
        base_mileage = 0.55
        base_receipt_rate = 0.5
        base_intercept = 0.0
    
    # Now compute predictions with caps
    predictions = np.zeros(len(data))
    
    for i, case in enumerate(data):
        d = days[i]
        m = miles[i]
        r = receipts[i]
        
        # Base calculation
        base = base_per_diem * d + base_mileage * m + base_receipt_rate * r + base_intercept
        
        # Apply cap if exists
        if d in caps:
            cap = caps[d]
            predictions[i] = min(base, cap)
        else:
            predictions[i] = base
    
    # Calculate MAE
    mae = np.mean(np.abs(predictions - expected))
    print(f"\nMAE with daily cap model: ${mae:.2f}")
    
    # Analyze residuals by receipt amount
    residuals = expected - predictions
    abs_residuals = np.abs(residuals)
    
    print("\nResidual analysis by receipt range:")
    receipt_ranges = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]
    
    for low, high in receipt_ranges:
        mask = (receipts >= low) & (receipts < high)
        if mask.sum() > 0:
            range_mae = np.mean(abs_residuals[mask])
            print(f"  ${low}-${high}: {mask.sum()} cases, MAE=${range_mae:.2f}")
    
    # Find worst predictions
    worst_idx = np.argsort(abs_residuals)[-20:]
    
    print("\nWorst 20 predictions:")
    for idx in worst_idx:
        case = data[idx]
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        exp = case['expected_output']
        pred = predictions[idx]
        error = abs_residuals[idx]
        
        print(f"  Case {idx}: {d}d, {m:.0f}mi, ${r:.2f}r")
        print(f"    Expected: ${exp:.2f}, Predicted: ${pred:.2f}, Error: ${error:.2f}")
        if d in caps and pred == caps[d]:
            print(f"    (Capped at ${caps[d]:.2f})")
    
    return {
        'base_per_diem': base_per_diem,
        'base_mileage': base_mileage,
        'base_receipt_rate': base_receipt_rate,
        'base_intercept': base_intercept,
        'caps': caps,
        'mae': mae
    }

def refine_model_with_exceptions(data, model):
    """Look for patterns in the residuals to refine the model"""
    print("\n\nRefining model by analyzing exceptions...")
    
    # Apply current model
    predictions = []
    for case in data:
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        
        base = (model['base_per_diem'] * d + 
                model['base_mileage'] * m + 
                model['base_receipt_rate'] * r + 
                model['base_intercept'])
        
        if d in model['caps']:
            pred = min(base, model['caps'][d])
        else:
            pred = base
            
        predictions.append(pred)
    
    predictions = np.array(predictions)
    expected = np.array([c['expected_output'] for c in data])
    residuals = expected - predictions
    abs_residuals = np.abs(residuals)
    
    # Look for patterns in large residuals
    large_error_idx = np.where(abs_residuals > 100)[0]
    
    print(f"Found {len(large_error_idx)} cases with error > $100")
    
    # Analyze these cases
    error_patterns = {}
    for idx in large_error_idx:
        case = data[idx]
        d = case['input']['trip_duration_days']
        r = case['input']['total_receipts_amount']
        
        key = f"{d}d_r{int(r/100)*100}"  # Group by days and receipt bucket
        if key not in error_patterns:
            error_patterns[key] = []
        error_patterns[key].append((idx, residuals[idx]))
    
    # Find consistent patterns
    print("\nConsistent error patterns:")
    for key, errors in error_patterns.items():
        if len(errors) >= 3:  # At least 3 cases with similar pattern
            avg_error = np.mean([e for _, e in errors])
            print(f"  {key}: {len(errors)} cases, avg error: ${avg_error:.2f}")

def main():
    print("Loading public cases...")
    data = load_public_cases()
    
    print(f"\nAnalyzing {len(data)} cases for daily caps...")
    caps = analyze_caps_by_days(data)
    
    print("\n" + "="*50)
    model = build_capped_model(data, caps)
    
    # Save model
    with open('analysis/daily_cap_model.json', 'w') as f:
        json.dump(model, f, indent=2)
    
    print(f"\nModel saved to analysis/daily_cap_model.json")
    
    # Try to refine further
    refine_model_with_exceptions(data, model)

if __name__ == '__main__':
    main()