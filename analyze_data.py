import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_data(filename='public_cases.json'):
    """Load test cases from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    days = []
    miles = []
    receipts = []
    outputs = []
    
    for case in data:
        inp = case['input']
        days.append(inp['trip_duration_days'])
        miles.append(inp['miles_traveled'])
        receipts.append(inp['total_receipts_amount'])
        outputs.append(case['expected_output'])
    
    return np.array(days), np.array(miles), np.array(receipts), np.array(outputs)

def analyze_patterns():
    """Analyze patterns in the data."""
    days, miles, receipts, outputs = load_data()
    
    print("=== DATA SUMMARY ===")
    print(f"Days: min={days.min()}, max={days.max()}, mean={days.mean():.2f}")
    print(f"Miles: min={miles.min()}, max={miles.max()}, mean={miles.mean():.2f}")
    print(f"Receipts: min=${receipts.min():.2f}, max=${receipts.max():.2f}, mean=${receipts.mean():.2f}")
    print(f"Outputs: min=${outputs.min():.2f}, max=${outputs.max():.2f}, mean=${outputs.mean():.2f}")
    
    # Analyze per-day rates
    print("\n=== PER-DAY ANALYSIS ===")
    for d in sorted(set(days)):
        mask = days == d
        if mask.sum() > 5:  # Only show if enough samples
            avg_out = outputs[mask].mean()
            avg_receipts = receipts[mask].mean()
            avg_miles = miles[mask].mean()
            print(f"Days={d}: avg_output=${avg_out:.2f}, avg_receipts=${avg_receipts:.2f}, avg_miles={avg_miles:.2f}, count={mask.sum()}")
    
    # Analyze by receipt ranges
    print("\n=== RECEIPT RANGES ===")
    receipt_ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 3000)]
    for low, high in receipt_ranges:
        mask = (receipts >= low) & (receipts < high)
        if mask.sum() > 0:
            avg_out = outputs[mask].mean()
            avg_days = days[mask].mean()
            receipt_ratio = (outputs[mask] / receipts[mask]).mean()
            print(f"Receipts ${low}-${high}: avg_output=${avg_out:.2f}, avg_days={avg_days:.2f}, receipt_ratio={receipt_ratio:.2f}, count={mask.sum()}")
    
    # Analyze receipt transformation
    print("\n=== RECEIPT TRANSFORMATION ===")
    # Look at cases with similar days/miles but different receipts
    for d in [1, 3, 5]:
        for m_range in [(0, 100), (100, 300), (300, 600)]:
            mask = (days == d) & (miles >= m_range[0]) & (miles < m_range[1])
            if mask.sum() > 10:
                r = receipts[mask]
                o = outputs[mask]
                if len(set(r)) > 5:  # Enough variation
                    # Fit polynomial to see receipt->output relationship
                    poly = PolynomialFeatures(degree=2)
                    r_poly = poly.fit_transform(r.reshape(-1, 1))
                    model = LinearRegression()
                    model.fit(r_poly, o)
                    score = model.score(r_poly, o)
                    print(f"Days={d}, Miles={m_range}: R²={score:.3f}, samples={mask.sum()}")
    
    # Analyze efficiency bonuses
    print("\n=== EFFICIENCY ANALYSIS ===")
    mpd = miles / days
    efficiency_ranges = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 500)]
    for low, high in efficiency_ranges:
        mask = (mpd >= low) & (mpd < high)
        if mask.sum() > 0:
            avg_out = outputs[mask].mean()
            avg_days = days[mask].mean()
            avg_receipts = receipts[mask].mean()
            print(f"MPD {low}-{high}: avg_output=${avg_out:.2f}, avg_days={avg_days:.2f}, avg_receipts=${avg_receipts:.2f}, count={mask.sum()}")
    
    # Check for .49/.99 receipt endings
    print("\n=== RECEIPT ENDING ANALYSIS ===")
    cents = np.round((receipts * 100) % 100).astype(int)
    special_endings = [49, 99]
    for ending in special_endings:
        mask = cents == ending
        if mask.sum() > 0:
            avg_out = outputs[mask].mean()
            avg_receipts = receipts[mask].mean()
            print(f"Receipts ending in .{ending}: avg_output=${avg_out:.2f}, avg_receipts=${avg_receipts:.2f}, count={mask.sum()}")
    
    # Simple regression to find base rates
    print("\n=== BASE RATE ESTIMATION ===")
    # For short trips with low receipts to estimate per diem
    mask = (days <= 3) & (receipts < 50) & (miles < 100)
    if mask.sum() > 0:
        X = np.column_stack([days[mask], miles[mask], receipts[mask]])
        y = outputs[mask]
        model = LinearRegression()
        model.fit(X, y)
        print(f"Base rates (low receipts): per_day=${model.coef_[0]:.2f}, per_mile=${model.coef_[1]:.2f}, per_receipt_dollar=${model.coef_[2]:.2f}")
        print(f"Intercept: ${model.intercept_:.2f}")

def find_receipt_pattern():
    """Specifically analyze receipt handling."""
    days, miles, receipts, outputs = load_data()
    
    print("\n=== DETAILED RECEIPT ANALYSIS ===")
    
    # For trips of specific lengths, see how receipts affect output
    for d in [1, 3, 5]:
        mask = (days == d) & (miles < 200)  # Control for miles
        if mask.sum() > 20:
            r = receipts[mask]
            o = outputs[mask]
            
            # Sort by receipts
            sort_idx = np.argsort(r)
            r_sorted = r[sort_idx]
            o_sorted = o[sort_idx]
            
            print(f"\nDays={d} (miles<200):")
            # Show samples across receipt range
            indices = np.linspace(0, len(r_sorted)-1, min(10, len(r_sorted)), dtype=int)
            for i in indices:
                print(f"  Receipts=${r_sorted[i]:.2f} -> Output=${o_sorted[i]:.2f}")

if __name__ == "__main__":
    analyze_patterns()
    find_receipt_pattern()