#!/usr/bin/env python3
"""
Shrink-the-Diff Solver: Iteratively find all patterns until zero error.
"""

import json
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict, Counter
import pandas as pd


def load_data():
    """Load public cases"""
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    data = []
    for case in cases:
        d = int(case['input']['trip_duration_days'])
        m = float(case['input']['miles_traveled'])
        r = float(case['input']['total_receipts_amount'])
        output = float(case['expected_output'])
        data.append({'days': d, 'miles': m, 'receipts': r, 'expected': output})
    
    return pd.DataFrame(data)


def fit_piecewise_model(df):
    """Fit the two-regime piecewise model"""
    BREAKPOINT = 828.10
    
    # Split data
    low_df = df[df['receipts'] <= BREAKPOINT]
    high_df = df[df['receipts'] > BREAKPOINT]
    
    # Fit low regime
    X_low = low_df[['days', 'miles', 'receipts']].values
    y_low = low_df['expected'].values
    model_low = LinearRegression()
    model_low.fit(X_low, y_low)
    
    # Fit high regime  
    X_high = high_df[['days', 'miles', 'receipts']].values
    y_high = high_df['expected'].values
    model_high = LinearRegression()
    model_high.fit(X_high, y_high)
    
    # Round coefficients to financially plausible values
    low_params = {
        'per_diem': round(model_low.coef_[0]),
        'mileage': round(model_low.coef_[1], 2),
        'receipt_rate': round(model_low.coef_[2], 2),
        'intercept': round(model_low.intercept_, 2)
    }
    
    high_params = {
        'per_diem': round(model_high.coef_[0]),
        'mileage': round(model_high.coef_[1], 2),
        'receipt_rate': round(model_high.coef_[2], 2),
        'intercept': round(model_high.intercept_, 2)
    }
    
    return low_params, high_params, BREAKPOINT


def find_daily_caps(df):
    """Find daily caps from high-receipt cases"""
    # Focus on high receipt cases
    high_receipt_df = df[df['receipts'] > 1800]
    
    caps = {}
    for days in range(1, 11):
        day_df = high_receipt_df[high_receipt_df['days'] == days]
        if len(day_df) > 0:
            caps[days] = day_df['expected'].max()
    
    return caps


def apply_model(row, low_params, high_params, breakpoint, caps):
    """Apply the current model to calculate reimbursement"""
    d = row['days']
    m = row['miles']
    r = row['receipts']
    
    # Piecewise calculation
    if r <= breakpoint:
        amount = (low_params['per_diem'] * d + 
                  low_params['mileage'] * m + 
                  low_params['receipt_rate'] * r + 
                  low_params['intercept'])
    else:
        amount = (high_params['per_diem'] * d + 
                  high_params['mileage'] * m + 
                  high_params['receipt_rate'] * r + 
                  high_params['intercept'])
    
    # Apply cap
    if d in caps:
        amount = min(amount, caps[d])
    
    return amount


def analyze_residuals(df, predictions):
    """Analyze residuals to find patterns"""
    df['predicted'] = predictions
    df['error'] = df['expected'] - df['predicted']
    df['abs_error'] = abs(df['error'])
    
    # Sort by absolute error
    worst_cases = df.nlargest(20, 'abs_error')
    
    print("\n=== Top 20 Worst Predictions ===")
    for _, row in worst_cases.iterrows():
        print(f"Days: {row['days']}, Miles: {row['miles']:.0f}, "
              f"Receipts: ${row['receipts']:.2f}, "
              f"Expected: ${row['expected']:.2f}, "
              f"Predicted: ${row['predicted']:.2f}, "
              f"Error: ${row['error']:.2f}")
    
    # Analyze patterns
    patterns = {
        '5_day_trips': df[df['days'] == 5]['error'].describe(),
        'receipt_49_cents': df[df['receipts'] % 1 == 0.49]['error'].describe(),
        'receipt_99_cents': df[df['receipts'] % 1 == 0.99]['error'].describe(),
        'high_efficiency': df[df['miles'] / df['days'] > 200]['error'].describe(),
        'by_duration': df.groupby('days')['error'].agg(['mean', 'std', 'count'])
    }
    
    return patterns, worst_cases


def main():
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} cases")
    
    # Step 1: Fit piecewise model
    print("\n=== Fitting Piecewise Model ===")
    low_params, high_params, breakpoint = fit_piecewise_model(df)
    print(f"Low regime (receipts <= ${breakpoint}):")
    print(f"  {low_params['per_diem']}*days + {low_params['mileage']}*miles + "
          f"{low_params['receipt_rate']}*receipts + {low_params['intercept']}")
    print(f"High regime (receipts > ${breakpoint}):")
    print(f"  {high_params['per_diem']}*days + {high_params['mileage']}*miles + "
          f"{high_params['receipt_rate']}*receipts + {high_params['intercept']}")
    
    # Step 2: Find daily caps
    print("\n=== Finding Daily Caps ===")
    caps = find_daily_caps(df)
    for days, cap in sorted(caps.items()):
        print(f"  {days} days: ${cap:.2f}")
    
    # Step 3: Apply model and calculate residuals
    print("\n=== Applying Model ===")
    predictions = df.apply(lambda row: apply_model(row, low_params, high_params, breakpoint, caps), axis=1)
    
    # Step 4: Analyze residuals
    patterns, worst_cases = analyze_residuals(df, predictions)
    
    print("\n=== Pattern Analysis ===")
    print("\n5-day trips error stats:")
    print(patterns['5_day_trips'])
    
    print("\nReceipts ending in .49:")
    print(patterns['receipt_49_cents'])
    
    print("\nReceipts ending in .99:")
    print(patterns['receipt_99_cents'])
    
    print("\nHigh efficiency trips (>200 miles/day):")
    print(patterns['high_efficiency'])
    
    print("\nErrors by trip duration:")
    print(patterns['by_duration'])
    
    # Calculate overall accuracy
    total_error = abs(df['expected'] - predictions).sum()
    avg_error = total_error / len(df)
    perfect_predictions = sum(abs(df['expected'] - predictions) < 0.01)
    
    print(f"\n=== Overall Accuracy ===")
    print(f"Total absolute error: ${total_error:.2f}")
    print(f"Average error per case: ${avg_error:.2f}")
    print(f"Perfect predictions: {perfect_predictions}/{len(df)}")
    
    # Save current model state
    model_state = {
        'breakpoint': breakpoint,
        'low_regime': low_params,
        'high_regime': high_params,
        'caps': {str(k): v for k, v in caps.items()},
        'total_error': total_error,
        'avg_error': avg_error
    }
    
    with open('model_iteration_1.json', 'w') as f:
        json.dump(model_state, f, indent=2)
    
    print("\nModel state saved to model_iteration_1.json")


if __name__ == "__main__":
    main()