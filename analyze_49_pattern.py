#!/usr/bin/env python3
"""
Analyze the .49 receipt pattern - seems like they use a different formula!
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

data = []
for case in cases:
    d = int(case['input']['trip_duration_days'])
    m = float(case['input']['miles_traveled'])
    r = float(case['input']['total_receipts_amount'])
    output = float(case['expected_output'])
    
    cents = round((r % 1) * 100)
    
    data.append({
        'days': d, 
        'miles': m, 
        'receipts': r, 
        'expected': output,
        'cents': cents,
        'ends_49': cents == 49,
        'ends_99': cents == 99
    })

df = pd.DataFrame(data)

# Separate .49 cases
df_49 = df[df['ends_49']]
df_normal = df[~df['ends_49'] & ~df['ends_99']]
df_99 = df[df['ends_99']]

print(f"Total cases: {len(df)}")
print(f".49 cases: {len(df_49)}")
print(f".99 cases: {len(df_99)}")
print(f"Normal cases: {len(df_normal)}")

# Analyze .49 cases
print("\n=== All .49 Receipt Cases ===")
for _, row in df_49.sort_values('receipts').iterrows():
    print(f"Days: {row['days']}, Miles: {row['miles']:.0f}, Receipts: ${row['receipts']:.2f}, Expected: ${row['expected']:.2f}")

# Try to fit a formula for .49 cases
print("\n=== Fitting Formula for .49 Cases ===")

# Maybe it's just based on miles and days?
X_49 = df_49[['days', 'miles']].values
y_49 = df_49['expected'].values

model_49_no_receipts = LinearRegression()
model_49_no_receipts.fit(X_49, y_49)

print(f"\nFormula without receipts:")
print(f"  {model_49_no_receipts.coef_[0]:.2f}*days + {model_49_no_receipts.coef_[1]:.2f}*miles + {model_49_no_receipts.intercept_:.2f}")

# Check predictions
df_49['pred_no_receipts'] = model_49_no_receipts.predict(X_49)
df_49['error_no_receipts'] = abs(df_49['expected'] - df_49['pred_no_receipts'])

print(f"\nMax error: ${df_49['error_no_receipts'].max():.2f}")
print(f"Mean error: ${df_49['error_no_receipts'].mean():.2f}")

# Try including receipts with small coefficient
X_49_full = df_49[['days', 'miles', 'receipts']].values
model_49_full = LinearRegression()
model_49_full.fit(X_49_full, y_49)

print(f"\nFormula with receipts:")
print(f"  {model_49_full.coef_[0]:.2f}*days + {model_49_full.coef_[1]:.2f}*miles + {model_49_full.coef_[2]:.4f}*receipts + {model_49_full.intercept_:.2f}")

df_49['pred_full'] = model_49_full.predict(X_49_full)
df_49['error_full'] = abs(df_49['expected'] - df_49['pred_full'])

print(f"\nMax error: ${df_49['error_full'].max():.2f}")
print(f"Mean error: ${df_49['error_full'].mean():.2f}")

# Show cases with largest errors
print("\n=== Cases with Largest Errors (full model) ===")
for _, row in df_49.nlargest(5, 'error_full').iterrows():
    print(f"Days: {row['days']}, Miles: {row['miles']:.0f}, Receipts: ${row['receipts']:.2f}")
    print(f"  Expected: ${row['expected']:.2f}, Predicted: ${row['pred_full']:.2f}, Error: ${row['error_full']:.2f}")

# Now analyze .99 cases
print("\n\n=== All .99 Receipt Cases ===")
for _, row in df_99.sort_values('receipts').iterrows():
    # Apply normal formula to see difference
    BREAKPOINT = 828.10
    if row['receipts'] <= BREAKPOINT:
        normal_calc = 56*row['days'] + 0.52*row['miles'] + 0.51*row['receipts'] + 32.49
    else:
        normal_calc = 43*row['days'] + 0.39*row['miles'] + 0.08*row['receipts'] + 907.66
    
    diff = row['expected'] - normal_calc
    print(f"Days: {row['days']}, Miles: {row['miles']:.0f}, Receipts: ${row['receipts']:.2f}, Expected: ${row['expected']:.2f}")
    print(f"  Normal formula would give: ${normal_calc:.2f}, Difference: ${diff:.2f}")