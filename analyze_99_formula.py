#!/usr/bin/env python3
"""
Find the formula for .99 receipt cases
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
        'ends_99': cents == 99
    })

df = pd.DataFrame(data)
df_99 = df[df['ends_99']]

print(f"Analyzing {len(df_99)} cases ending in .99")

# Try fitting without receipts first (like .49 cases)
X_99 = df_99[['days', 'miles']].values
y_99 = df_99['expected'].values

model_99_no_receipts = LinearRegression()
model_99_no_receipts.fit(X_99, y_99)

print(f"\nFormula without receipts:")
print(f"  {model_99_no_receipts.coef_[0]:.2f}*days + {model_99_no_receipts.coef_[1]:.2f}*miles + {model_99_no_receipts.intercept_:.2f}")

# Check predictions
df_99['pred_no_receipts'] = model_99_no_receipts.predict(X_99)
df_99['error_no_receipts'] = abs(df_99['expected'] - df_99['pred_no_receipts'])

print(f"\nMax error: ${df_99['error_no_receipts'].max():.2f}")
print(f"Mean error: ${df_99['error_no_receipts'].mean():.2f}")

# Try with receipts
X_99_full = df_99[['days', 'miles', 'receipts']].values
model_99_full = LinearRegression()
model_99_full.fit(X_99_full, y_99)

print(f"\nFormula with receipts:")
print(f"  {model_99_full.coef_[0]:.2f}*days + {model_99_full.coef_[1]:.2f}*miles + {model_99_full.coef_[2]:.4f}*receipts + {model_99_full.intercept_:.2f}")

df_99['pred_full'] = model_99_full.predict(X_99_full)
df_99['error_full'] = abs(df_99['expected'] - df_99['pred_full'])

print(f"\nMax error: ${df_99['error_full'].max():.2f}")
print(f"Mean error: ${df_99['error_full'].mean():.2f}")

# Show all cases with predictions
print("\n=== All .99 Cases with Predictions ===")
for _, row in df_99.sort_values('days').iterrows():
    print(f"Days: {row['days']}, Miles: {row['miles']:.0f}, Receipts: ${row['receipts']:.2f}")
    print(f"  Expected: ${row['expected']:.2f}")
    print(f"  Predicted (no receipts): ${row['pred_no_receipts']:.2f}, Error: ${row['error_no_receipts']:.2f}")
    print(f"  Predicted (with receipts): ${row['pred_full']:.2f}, Error: ${row['error_full']:.2f}")
    print()

# Try integer coefficients
print("\n=== Trying Integer Coefficients ===")
# Round to nearest sensible values
days_coef = round(model_99_full.coef_[0])
miles_coef = round(model_99_full.coef_[1], 2)
receipts_coef = round(model_99_full.coef_[2], 2)
intercept = round(model_99_full.intercept_)

print(f"Rounded formula: {days_coef}*days + {miles_coef}*miles + {receipts_coef}*receipts + {intercept}")

# Test rounded formula
df_99['pred_rounded'] = days_coef * df_99['days'] + miles_coef * df_99['miles'] + receipts_coef * df_99['receipts'] + intercept
df_99['error_rounded'] = abs(df_99['expected'] - df_99['pred_rounded'])

print(f"Max error with rounded: ${df_99['error_rounded'].max():.2f}")
print(f"Mean error with rounded: ${df_99['error_rounded'].mean():.2f}")