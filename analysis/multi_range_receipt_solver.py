#!/usr/bin/env python3
"""
Multi-range receipt model based on observed patterns
Receipts have different effects in different ranges
"""
import json
import numpy as np
from sklearn.linear_model import LinearRegression

def load_public_cases():
    """Load all public test cases"""
    with open('public_cases.json', 'r') as f:
        return json.load(f)

def create_receipt_features(receipts):
    """
    Create piecewise linear features for receipts based on ranges
    Based on insights from CONTINUE_FROM_HERE.md
    """
    # Define receipt ranges
    ranges = [
        (0, 100),      # Heavy penalty
        (100, 500),    # Moderate penalty
        (500, 1000),   # Small positive
        (1000, 2000),  # Larger positive
        (2000, np.inf) # Decreasing
    ]
    
    n_samples = len(receipts)
    n_ranges = len(ranges)
    features = np.zeros((n_samples, n_ranges))
    
    for i, r in enumerate(receipts):
        for j, (low, high) in enumerate(ranges):
            if r <= low:
                # No contribution from this range
                continue
            elif r > high:
                # Full contribution from this range
                features[i, j] = high - low
            else:
                # Partial contribution
                features[i, j] = r - low
    
    return features, ranges

def solve_multi_range_model(data):
    """Solve for coefficients with multi-range receipt model"""
    # Extract data
    days = np.array([c['input']['trip_duration_days'] for c in data])
    miles = np.array([c['input']['miles_traveled'] for c in data])
    receipts = np.array([c['input']['total_receipts_amount'] for c in data])
    expected = np.array([c['expected_output'] for c in data])
    
    # Create receipt features
    receipt_features, ranges = create_receipt_features(receipts)
    
    print("Receipt ranges and case counts:")
    for i, (low, high) in enumerate(ranges):
        count = np.sum((receipts > low) & (receipts <= high if high != np.inf else True))
        print(f"  ${low}-${high if high != np.inf else 'inf'}: {count} cases")
    
    # Build feature matrix: [days, miles, receipt_features, intercept]
    X = np.column_stack([days, miles, receipt_features, np.ones(len(data))])
    
    # Solve linear regression
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, expected)
    
    coeffs = reg.coef_
    
    print(f"\nCoefficients:")
    print(f"  Per diem: ${coeffs[0]:.2f}")
    print(f"  Mileage: ${coeffs[1]:.4f}/mile")
    print(f"  Receipt effects by range:")
    for i, (low, high) in enumerate(ranges):
        print(f"    ${low}-${high if high != np.inf else 'inf'}: {coeffs[2+i]:.4f}")
    print(f"  Intercept: ${coeffs[-1]:.2f}")
    
    # Test model
    predictions = reg.predict(X)
    mae = np.mean(np.abs(predictions - expected))
    print(f"\nOverall MAE: ${mae:.2f}")
    
    # Analyze residuals
    residuals = expected - predictions
    abs_residuals = np.abs(residuals)
    
    print(f"\nResidual statistics:")
    print(f"  Mean absolute error: ${mae:.2f}")
    print(f"  Max absolute error: ${np.max(abs_residuals):.2f}")
    print(f"  Cases within $1: {np.sum(abs_residuals <= 1)}")
    print(f"  Cases within $5: {np.sum(abs_residuals <= 5)}")
    print(f"  Cases within $10: {np.sum(abs_residuals <= 10)}")
    
    # Find worst cases
    worst_indices = np.argsort(abs_residuals)[-20:]
    
    print("\nWorst 20 cases:")
    for idx in worst_indices:
        case = data[idx]
        pred = predictions[idx]
        error = abs_residuals[idx]
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        exp = case['expected_output']
        
        # Show breakdown
        base = coeffs[0] * d + coeffs[1] * m
        receipt_effect = 0
        for j in range(len(ranges)):
            if receipt_features[idx, j] > 0:
                receipt_effect += coeffs[2+j] * receipt_features[idx, j]
        
        print(f"  Case {idx}: {d}d, {m:.2f}mi, ${r:.2f}r")
        print(f"    Expected: ${exp:.2f}, Predicted: ${pred:.2f}, Error: ${error:.2f}")
        print(f"    Breakdown: base=${base:.2f}, receipts=${receipt_effect:.2f}, intercept=${coeffs[-1]:.2f}")
    
    return {
        'coefficients': coeffs.tolist(),
        'ranges': [(low, high if high != np.inf else None) for low, high in ranges],
        'mae': mae
    }

def test_integer_math(model):
    """Test the model with integer math as suggested in THE-BIG-PLAN-PART-2"""
    print("\nTesting with integer math (cents):")
    
    # Load test cases
    data = load_public_cases()
    
    errors = []
    for i, case in enumerate(data[:10]):  # Test first 10 cases
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        # Convert to cents
        r_cents = int(r * 100)
        expected_cents = int(expected * 100)
        
        # Compute with integer math
        # Note: We need to carefully handle the coefficient scaling
        per_diem_cents = int(model['coefficients'][0] * 100)
        mileage_cents = int(model['coefficients'][1] * 100)
        
        base_cents = per_diem_cents * d + (mileage_cents * m) // 100
        
        # Add receipt effects
        receipt_cents = 0
        r_remaining = r
        for j, (low, high) in enumerate(model['ranges']):
            if high is None:
                high = float('inf')
            
            if r_remaining <= 0:
                break
                
            if r > low:
                amount_in_range = min(r_remaining, high - low if r > high else r - low)
                effect_rate = int(model['coefficients'][2+j] * 100)
                receipt_cents += (effect_rate * int(amount_in_range * 100)) // 100
                r_remaining -= amount_in_range
        
        intercept_cents = int(model['coefficients'][-1] * 100)
        total_cents = base_cents + receipt_cents + intercept_cents
        
        # Convert back to dollars
        predicted = total_cents / 100.0
        error = abs(predicted - expected)
        errors.append(error)
        
        if error > 1.0:
            print(f"  Case {i}: Error ${error:.2f}")
            print(f"    {d}d, {m}mi, ${r:.2f}r -> Expected: ${expected:.2f}, Predicted: ${predicted:.2f}")
    
    print(f"  Average error in first 10 cases: ${np.mean(errors):.2f}")

def main():
    print("Loading public cases...")
    data = load_public_cases()
    
    print(f"\nAnalyzing {len(data)} cases with multi-range receipt model...")
    model = solve_multi_range_model(data)
    
    # Save model
    with open('analysis/multi_range_model.json', 'w') as f:
        json.dump(model, f, indent=2)
    
    print(f"\nModel saved to analysis/multi_range_model.json")
    
    # Test integer math
    test_integer_math(model)

if __name__ == '__main__':
    main()