#!/usr/bin/env python3
"""
Solve coefficients with specific receipt breakpoint at $828.10
Following insights from decision tree analysis and THE-BIG-PLAN
"""
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def load_public_cases():
    """Load all public test cases"""
    with open('public_cases.json', 'r') as f:
        return json.load(f)

def solve_piecewise_receipts(data):
    """
    Solve for coefficients with receipt breakpoint at $828.10
    Model: output = per_diem * days + mileage_rate * miles + receipt_func(receipts)
    """
    # Extract data
    days = np.array([c['input']['trip_duration_days'] for c in data])
    miles = np.array([c['input']['miles_traveled'] for c in data])
    receipts = np.array([c['input']['total_receipts_amount'] for c in data])
    expected = np.array([c['expected_output'] for c in data])
    
    # Split by receipt threshold
    receipt_threshold = 828.10
    low_mask = receipts <= receipt_threshold
    high_mask = receipts > receipt_threshold
    
    print(f"Cases with receipts <= ${receipt_threshold}: {low_mask.sum()}")
    print(f"Cases with receipts > ${receipt_threshold}: {high_mask.sum()}")
    
    # First, estimate base coefficients (per_diem and mileage) from low receipt cases
    # where receipt effect might be simpler
    low_receipt_mask = receipts < 100  # Use very low receipt cases
    if low_receipt_mask.sum() > 20:
        X_base = np.column_stack([days[low_receipt_mask], miles[low_receipt_mask]])
        y_base = expected[low_receipt_mask] - receipts[low_receipt_mask]  # Subtract receipts to isolate base
        
        reg_base = LinearRegression()
        reg_base.fit(X_base, y_base)
        
        per_diem_estimate = reg_base.coef_[0]
        mileage_estimate = reg_base.coef_[1]
        
        print(f"\nBase estimates from low receipt cases:")
        print(f"  Per diem: ${per_diem_estimate:.2f}")
        print(f"  Mileage: ${mileage_estimate:.2f}")
    else:
        per_diem_estimate = 86.0
        mileage_estimate = 0.55
    
    # Now solve for receipt coefficients in each region
    results = {}
    
    for region_name, mask in [("low", low_mask), ("high", high_mask)]:
        if mask.sum() < 10:
            continue
            
        print(f"\nSolving for {region_name} receipt region ({mask.sum()} cases):")
        
        # Subtract estimated base amounts to isolate receipt effect
        base_amount = per_diem_estimate * days[mask] + mileage_estimate * miles[mask]
        receipt_effect = expected[mask] - base_amount
        
        # Model receipt effect as linear function of receipts
        X_receipt = receipts[mask].reshape(-1, 1)
        reg_receipt = LinearRegression()
        reg_receipt.fit(X_receipt, receipt_effect)
        
        receipt_coef = reg_receipt.coef_[0]
        receipt_intercept = reg_receipt.intercept_
        
        print(f"  Receipt coefficient: {receipt_coef:.4f}")
        print(f"  Receipt intercept: ${receipt_intercept:.2f}")
        
        # Compute residuals
        predicted_effect = reg_receipt.predict(X_receipt)
        residuals = receipt_effect - predicted_effect
        mae = np.mean(np.abs(residuals))
        print(f"  MAE for receipt effect: ${mae:.2f}")
        
        results[region_name] = {
            'receipt_coef': receipt_coef,
            'receipt_intercept': receipt_intercept,
            'mask': mask
        }
    
    # Now refine all coefficients together
    print("\nRefining all coefficients together:")
    
    # Build piecewise receipt feature
    receipt_feature = np.zeros_like(receipts)
    
    if 'low' in results:
        low_mask = results['low']['mask']
        receipt_feature[low_mask] = (results['low']['receipt_coef'] * receipts[low_mask] + 
                                     results['low']['receipt_intercept'])
    
    if 'high' in results:
        high_mask = results['high']['mask']
        receipt_feature[high_mask] = (results['high']['receipt_coef'] * receipts[high_mask] + 
                                      results['high']['receipt_intercept'])
    
    # Solve for final coefficients
    X_final = np.column_stack([days, miles, receipt_feature, np.ones(len(data))])
    reg_final = LinearRegression(fit_intercept=False)
    reg_final.fit(X_final, expected)
    
    final_coeffs = reg_final.coef_
    
    print(f"\nFinal coefficients:")
    print(f"  Per diem: ${final_coeffs[0]:.2f}")
    print(f"  Mileage: ${final_coeffs[1]:.4f}")
    print(f"  Receipt multiplier: {final_coeffs[2]:.4f}")
    print(f"  Intercept: ${final_coeffs[3]:.2f}")
    
    # Test on full dataset
    predictions = reg_final.predict(X_final)
    mae = np.mean(np.abs(predictions - expected))
    print(f"\nOverall MAE: ${mae:.2f}")
    
    # Find worst cases
    errors = np.abs(predictions - expected)
    worst_indices = np.argsort(errors)[-10:]
    
    print("\nWorst 10 cases:")
    for idx in worst_indices:
        case = data[idx]
        pred = predictions[idx]
        error = errors[idx]
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        print(f"  Case {idx}: {d}d, {m}mi, ${r:.2f}r -> Expected: ${case['expected_output']:.2f}, Predicted: ${pred:.2f}, Error: ${error:.2f}")
    
    # Save results
    return {
        'per_diem': final_coeffs[0],
        'mileage': final_coeffs[1],
        'receipt_threshold': receipt_threshold,
        'low_receipt_coef': results.get('low', {}).get('receipt_coef', 0),
        'low_receipt_intercept': results.get('low', {}).get('receipt_intercept', 0),
        'high_receipt_coef': results.get('high', {}).get('receipt_coef', 0),
        'high_receipt_intercept': results.get('high', {}).get('receipt_intercept', 0),
        'receipt_multiplier': final_coeffs[2],
        'intercept': final_coeffs[3],
        'mae': mae
    }

def plot_receipt_effect(data, model):
    """Plot the receipt effect to visualize the breakpoint"""
    receipts = np.array([c['input']['total_receipts_amount'] for c in data])
    days = np.array([c['input']['trip_duration_days'] for c in data])
    miles = np.array([c['input']['miles_traveled'] for c in data])
    expected = np.array([c['expected_output'] for c in data])
    
    # Calculate base amount
    base_amount = model['per_diem'] * days + model['mileage'] * miles
    receipt_effect = expected - base_amount
    
    # Sort by receipts for plotting
    sort_idx = np.argsort(receipts)
    receipts_sorted = receipts[sort_idx]
    effect_sorted = receipt_effect[sort_idx]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(receipts_sorted, effect_sorted, alpha=0.5, s=10)
    plt.axvline(x=model['receipt_threshold'], color='red', linestyle='--', label=f"Threshold: ${model['receipt_threshold']}")
    plt.xlabel('Receipt Amount ($)')
    plt.ylabel('Receipt Effect on Reimbursement ($)')
    plt.title('Receipt Effect vs Receipt Amount')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('analysis/receipt_effect_plot.png')
    print("\nPlot saved to analysis/receipt_effect_plot.png")

def main():
    print("Loading public cases...")
    data = load_public_cases()
    
    print(f"Analyzing {len(data)} cases...")
    model = solve_piecewise_receipts(data)
    
    # Save model
    with open('analysis/receipt_breakpoint_model.json', 'w') as f:
        json.dump(model, f, indent=2)
    
    print(f"\nModel saved to analysis/receipt_breakpoint_model.json")
    
    # Plot receipt effect
    plot_receipt_effect(data, model)

if __name__ == '__main__':
    main()