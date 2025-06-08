#!/usr/bin/env python3
"""
Final formula solver using discovered patterns:
1. Receipt threshold at $828.10
2. Daily caps
3. Different receipt rates below/above threshold
"""
import json
import numpy as np
from sklearn.linear_model import LinearRegression

def load_public_cases():
    """Load all public test cases"""
    with open('public_cases.json', 'r') as f:
        return json.load(f)

def find_daily_caps(data):
    """Find precise daily caps from high-receipt cases"""
    by_days = {}
    
    for case in data:
        d = case['input']['trip_duration_days']
        r = case['input']['total_receipts_amount']
        exp = case['expected_output']
        
        if r > 1500:  # High receipt cases
            if d not in by_days:
                by_days[d] = []
            by_days[d].append(exp)
    
    caps = {}
    for days in sorted(by_days.keys()):
        if len(by_days[days]) >= 5:
            # Use 95th percentile to avoid outliers
            sorted_values = sorted(by_days[days])
            p95 = sorted_values[int(len(sorted_values) * 0.95)]
            caps[days] = p95
    
    return caps

def build_precise_model(data):
    """Build model with receipt threshold and caps"""
    RECEIPT_THRESHOLD = 828.10
    
    # First, find caps
    caps = find_daily_caps(data)
    print("Daily caps found:")
    for d, cap in sorted(caps.items()):
        print(f"  {d} days: ${cap:.2f}")
    
    # Split data by receipt threshold
    below_threshold = []
    above_threshold = []
    
    for case in data:
        r = case['input']['total_receipts_amount']
        if r <= RECEIPT_THRESHOLD:
            below_threshold.append(case)
        else:
            above_threshold.append(case)
    
    # Fit model for below threshold
    print(f"\nFitting model for receipts <= ${RECEIPT_THRESHOLD}...")
    X_below = []
    y_below = []
    
    for case in below_threshold:
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        X_below.append([d, m, r])
        y_below.append(case['expected_output'])
    
    X_below = np.array(X_below)
    y_below = np.array(y_below)
    
    # Add intercept
    X_below_with_intercept = np.column_stack([X_below, np.ones(len(X_below))])
    reg_below = LinearRegression(fit_intercept=False)
    reg_below.fit(X_below_with_intercept, y_below)
    
    # Round coefficients to sensible values
    coeffs_below = reg_below.coef_.copy()
    coeffs_below[0] = round(coeffs_below[0])  # Per diem
    coeffs_below[1] = round(coeffs_below[1] * 100) / 100  # Mileage to cents
    coeffs_below[2] = round(coeffs_below[2] * 100) / 100  # Receipt rate
    coeffs_below[3] = round(coeffs_below[3], 2)  # Intercept
    
    print(f"  Per diem: ${coeffs_below[0]:.0f}")
    print(f"  Mileage: ${coeffs_below[1]:.2f}")
    print(f"  Receipt rate: {coeffs_below[2]:.2f}")
    print(f"  Intercept: ${coeffs_below[3]:.2f}")
    
    # Fit model for above threshold
    print(f"\nFitting model for receipts > ${RECEIPT_THRESHOLD}...")
    X_above = []
    y_above = []
    
    for case in above_threshold:
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        
        # Apply cap if exists
        if d in caps and case['expected_output'] < caps[d]:
            X_above.append([d, m, r])
            y_above.append(case['expected_output'])
    
    X_above = np.array(X_above)
    y_above = np.array(y_above)
    
    # Add intercept
    X_above_with_intercept = np.column_stack([X_above, np.ones(len(X_above))])
    reg_above = LinearRegression(fit_intercept=False)
    reg_above.fit(X_above_with_intercept, y_above)
    
    # Round coefficients
    coeffs_above = reg_above.coef_.copy()
    coeffs_above[0] = round(coeffs_above[0])
    coeffs_above[1] = round(coeffs_above[1] * 100) / 100
    coeffs_above[2] = round(coeffs_above[2] * 100) / 100
    coeffs_above[3] = round(coeffs_above[3], 2)
    
    print(f"  Per diem: ${coeffs_above[0]:.0f}")
    print(f"  Mileage: ${coeffs_above[1]:.2f}")
    print(f"  Receipt rate: {coeffs_above[2]:.2f}")
    print(f"  Intercept: ${coeffs_above[3]:.2f}")
    
    return {
        'threshold': RECEIPT_THRESHOLD,
        'coeffs_below': coeffs_below,
        'coeffs_above': coeffs_above,
        'caps': caps
    }

def test_model(data, model):
    """Test the model on all data"""
    predictions = []
    
    for case in data:
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        
        # Choose coefficients based on receipt threshold
        if r <= model['threshold']:
            coeffs = model['coeffs_below']
        else:
            coeffs = model['coeffs_above']
        
        # Calculate base reimbursement
        reimbursement = coeffs[0] * d + coeffs[1] * m + coeffs[2] * r + coeffs[3]
        
        # Apply cap if exists
        if d in model['caps']:
            reimbursement = min(reimbursement, model['caps'][d])
        
        predictions.append(reimbursement)
    
    predictions = np.array(predictions)
    expected = np.array([c['expected_output'] for c in data])
    
    # Calculate metrics
    residuals = expected - predictions
    abs_residuals = np.abs(residuals)
    mae = np.mean(abs_residuals)
    
    print(f"\nModel performance:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  Within $1: {np.sum(abs_residuals <= 1)}")
    print(f"  Within $5: {np.sum(abs_residuals <= 5)}")
    print(f"  Within $10: {np.sum(abs_residuals <= 10)}")
    
    # Show worst cases
    worst_idx = np.argsort(abs_residuals)[-10:]
    print("\nWorst 10 cases:")
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
        
        # Show which formula was used
        if r <= model['threshold']:
            print(f"    Used below-threshold formula")
        else:
            print(f"    Used above-threshold formula")
            if d in model['caps'] and pred == model['caps'][d]:
                print(f"    Hit cap of ${model['caps'][d]:.2f}")
    
    return predictions, abs_residuals

def generate_run_sh(model):
    """Generate the final run.sh script"""
    script = f"""#!/usr/bin/env bash
set -euo pipefail

days=$1
miles=$2
receipts=$3

# Receipt threshold
THRESHOLD=828.10

# Function to compare floats
compare_float() {{
    echo "$1 <= $2" | bc -l
}}

# Determine which formula to use
if [ $(compare_float "$receipts" "$THRESHOLD") -eq 1 ]; then
    # Below threshold formula
    per_diem={int(model['coeffs_below'][0])}
    mileage={model['coeffs_below'][1]}
    receipt_rate={model['coeffs_below'][2]}
    intercept={model['coeffs_below'][3]}
else
    # Above threshold formula
    per_diem={int(model['coeffs_above'][0])}
    mileage={model['coeffs_above'][1]}
    receipt_rate={model['coeffs_above'][2]}
    intercept={model['coeffs_above'][3]}
fi

# Calculate base reimbursement
base=$(echo "scale=2; $per_diem * $days + $mileage * $miles + $receipt_rate * $receipts + $intercept" | bc -l)

# Apply daily caps
case $days in
"""
    
    # Add cap cases
    for days, cap in sorted(model['caps'].items()):
        script += f"    {days}) cap={cap:.2f} ;;\n"
    
    script += """    *) cap=999999 ;;  # No cap
esac

# Apply cap if needed
if [ $(echo "$base > $cap" | bc -l) -eq 1 ]; then
    result=$cap
else
    result=$base
fi

# Output with proper formatting
printf "%.2f\\n" "$result"
"""
    
    with open('run_final.sh', 'w') as f:
        f.write(script)
    
    print("\nGenerated run_final.sh")

def main():
    print("Loading public cases...")
    data = load_public_cases()
    
    print(f"\nBuilding precise model for {len(data)} cases...")
    model = build_precise_model(data)
    
    predictions, abs_residuals = test_model(data, model)
    
    # Save model
    model_json = {
        'threshold': model['threshold'],
        'coeffs_below': model['coeffs_below'].tolist(),
        'coeffs_above': model['coeffs_above'].tolist(),
        'caps': model['caps']
    }
    
    with open('analysis/final_model.json', 'w') as f:
        json.dump(model_json, f, indent=2)
    
    print("\nModel saved to analysis/final_model.json")
    
    # Generate run.sh
    generate_run_sh(model)
    
    # If MAE is still high, suggest next steps
    mae = np.mean(abs_residuals)
    if mae > 10:
        print(f"\nMAE is still ${mae:.2f}. Consider:")
        print("1. More granular receipt ranges")
        print("2. Day-specific formulas")
        print("3. Non-linear receipt transformation")
        print("4. Integer math and rounding issues")

if __name__ == '__main__':
    main()