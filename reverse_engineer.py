import json
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
import sys

def load_data(filename='public_cases.json'):
    """Load test cases from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def analyze_base_components():
    """Try to identify base components of the formula."""
    data = load_data()
    
    print("=== ANALYZING BASE COMPONENTS ===\n")
    
    # Look at simple cases first
    print("Simple 1-day trips with low receipts:")
    for case in data[:100]:
        inp = case['input']
        if inp['trip_duration_days'] == 1 and inp['total_receipts_amount'] < 20:
            days = inp['trip_duration_days']
            miles = inp['miles_traveled']
            receipts = inp['total_receipts_amount']
            output = case['expected_output']
            
            # Try base calculation
            base = 100  # Per diem
            mileage = miles * 0.58
            total_simple = base + mileage + receipts
            
            print(f"D={days}, M={miles:.0f}, R=${receipts:.2f}")
            print(f"  Output: ${output:.2f}")
            print(f"  Simple calc (100 + {miles:.0f}*0.58 + {receipts:.2f}): ${total_simple:.2f}")
            print(f"  Difference: ${output - total_simple:.2f}")
            print()

def test_linear_regression():
    """Use linear regression to find coefficients."""
    data = load_data()
    
    print("\n=== LINEAR REGRESSION ANALYSIS ===\n")
    
    # Prepare data
    X = []
    y = []
    for case in data:
        inp = case['input']
        X.append([
            inp['trip_duration_days'],
            inp['miles_traveled'],
            inp['total_receipts_amount'],
            inp['miles_traveled'] / inp['trip_duration_days'],  # MPD
            inp['trip_duration_days'] == 5,  # 5-day bonus
            inp['total_receipts_amount'] < 50,  # Small receipt penalty
        ])
        y.append(case['expected_output'])
    
    X = np.array(X)
    y = np.array(y)
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    print("Coefficients:")
    print(f"  Per day: ${model.coef_[0]:.2f}")
    print(f"  Per mile: ${model.coef_[1]:.2f}")
    print(f"  Per receipt dollar: ${model.coef_[2]:.2f}")
    print(f"  Per MPD: ${model.coef_[3]:.2f}")
    print(f"  5-day bonus: ${model.coef_[4]:.2f}")
    print(f"  Small receipt penalty: ${model.coef_[5]:.2f}")
    print(f"  Intercept: ${model.intercept_:.2f}")
    
    # Test accuracy
    predictions = model.predict(X)
    errors = np.abs(predictions - y)
    print(f"\nAverage error: ${np.mean(errors):.2f}")
    print(f"Max error: ${np.max(errors):.2f}")

def analyze_receipt_patterns():
    """Deep dive into receipt handling."""
    data = load_data()
    
    print("\n=== RECEIPT PATTERN ANALYSIS ===\n")
    
    # Group by receipt ranges and analyze
    receipt_ranges = [
        (0, 10), (10, 50), (50, 100), (100, 200), 
        (200, 500), (500, 1000), (1000, 2000), (2000, 3000)
    ]
    
    for low, high in receipt_ranges:
        cases_in_range = []
        for case in data:
            inp = case['input']
            if low <= inp['total_receipts_amount'] < high:
                # Calculate receipt contribution
                # Estimate base + mileage
                base_mileage = inp['trip_duration_days'] * 100 + inp['miles_traveled'] * 0.58
                receipt_contrib = case['expected_output'] - base_mileage
                receipt_rate = receipt_contrib / inp['total_receipts_amount'] if inp['total_receipts_amount'] > 0 else 0
                
                cases_in_range.append({
                    'days': inp['trip_duration_days'],
                    'miles': inp['miles_traveled'],
                    'receipts': inp['total_receipts_amount'],
                    'output': case['expected_output'],
                    'receipt_contrib': receipt_contrib,
                    'receipt_rate': receipt_rate
                })
        
        if cases_in_range:
            avg_rate = np.mean([c['receipt_rate'] for c in cases_in_range])
            std_rate = np.std([c['receipt_rate'] for c in cases_in_range])
            print(f"Receipts ${low}-${high}: avg_rate={avg_rate:.2f}, std={std_rate:.2f}, count={len(cases_in_range)}")

def test_specific_hypothesis():
    """Test specific calculation hypothesis."""
    data = load_data()
    
    print("\n=== TESTING SPECIFIC FORMULA ===\n")
    
    errors = []
    for i, case in enumerate(data[:20]):  # Test first 20
        inp = case['input']
        days = inp['trip_duration_days']
        miles = inp['miles_traveled']
        receipts = inp['total_receipts_amount']
        expected = case['expected_output']
        
        # New hypothesis
        # Base per diem
        base = days * 100
        
        # Mileage - tiered
        if miles <= 100:
            mileage = miles * 0.58
        else:
            mileage = 100 * 0.58 + (miles - 100) * 0.50
        
        # Receipts - complex transformation
        if receipts == 0:
            receipt_part = 0
        elif receipts < 50:
            receipt_part = receipts * -0.5  # Penalty!
        elif receipts < 200:
            receipt_part = receipts * 0.8
        elif receipts < 1000:
            receipt_part = receipts * 0.9
        else:
            receipt_part = 900 + (receipts - 1000) * 0.5
        
        # Efficiency bonus
        mpd = miles / days
        if 150 <= mpd <= 250:
            efficiency_mult = 1.1
        else:
            efficiency_mult = 1.0
        
        # Calculate
        total = (base + mileage + receipt_part) * efficiency_mult
        
        # 5-day bonus
        if days == 5:
            total *= 1.05
        
        error = abs(total - expected)
        errors.append(error)
        
        if i < 10:  # Show first 10
            print(f"Case {i}: D={days}, M={miles:.0f}, R=${receipts:.2f}")
            print(f"  Expected: ${expected:.2f}, Calculated: ${total:.2f}, Error: ${error:.2f}")
    
    print(f"\nAverage error: ${np.mean(errors):.2f}")

if __name__ == "__main__":
    analyze_base_components()
    test_linear_regression()
    analyze_receipt_patterns()
    test_specific_hypothesis()