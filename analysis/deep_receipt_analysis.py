#!/usr/bin/env python3
"""
Deep dive into receipt transformation
Focus on understanding why high receipts lead to low reimbursements
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_public_cases():
    """Load all public test cases"""
    with open('public_cases.json', 'r') as f:
        return json.load(f)

def analyze_receipt_penalty_cases(data):
    """Analyze cases where expected < receipts (receipt penalty)"""
    penalty_cases = []
    
    for i, case in enumerate(data):
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        exp = case['expected_output']
        
        if exp < r:  # Receipt penalty case
            # Estimate base (without receipts)
            base_estimate = 50 * d + 0.5 * m  # rough estimate
            
            penalty_cases.append({
                'index': i,
                'days': d,
                'miles': m,
                'receipts': r,
                'expected': exp,
                'base_estimate': base_estimate,
                'receipt_penalty': r - exp,  # How much less than receipts
                'penalty_rate': (r - exp) / r if r > 0 else 0  # Penalty as fraction of receipts
            })
    
    # Sort by receipts
    penalty_cases.sort(key=lambda x: x['receipts'])
    
    print(f"Found {len(penalty_cases)} cases where expected < receipts")
    print("\nAnalyzing penalty patterns...")
    
    # Group by receipt ranges
    ranges = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000)]
    
    for low, high in ranges:
        range_cases = [c for c in penalty_cases if low <= c['receipts'] < high]
        if range_cases:
            avg_penalty_rate = np.mean([c['penalty_rate'] for c in range_cases])
            avg_expected = np.mean([c['expected'] for c in range_cases])
            avg_receipts = np.mean([c['receipts'] for c in range_cases])
            
            print(f"\nReceipts ${low}-${high}: {len(range_cases)} penalty cases")
            print(f"  Avg receipt amount: ${avg_receipts:.2f}")
            print(f"  Avg expected: ${avg_expected:.2f}")
            print(f"  Avg penalty rate: {avg_penalty_rate:.2%}")
            
            # Show examples
            for c in range_cases[:2]:
                print(f"  Example: {c['days']}d, {c['miles']:.0f}mi, ${c['receipts']:.2f}r -> ${c['expected']:.2f}")

def test_receipt_cap_hypothesis(data):
    """Test if there's a cap on total reimbursement based on days"""
    print("\n\nTesting receipt cap hypothesis...")
    
    # Group by days
    by_days = {}
    for case in data:
        d = case['input']['trip_duration_days']
        if d not in by_days:
            by_days[d] = []
        by_days[d].append({
            'miles': case['input']['miles_traveled'],
            'receipts': case['input']['total_receipts_amount'],
            'expected': case['expected_output']
        })
    
    # For each day count, find the apparent cap
    caps = {}
    
    for days in sorted(by_days.keys()):
        cases = by_days[days]
        
        # Look at high-receipt cases
        high_receipt_cases = [c for c in cases if c['receipts'] > 1000]
        
        if len(high_receipt_cases) >= 5:
            # Sort by expected output
            sorted_cases = sorted(high_receipt_cases, key=lambda x: x['expected'])
            
            # Look at top 10%
            top_10_pct = sorted_cases[int(len(sorted_cases) * 0.9):]
            
            if top_10_pct:
                avg_top = np.mean([c['expected'] for c in top_10_pct])
                max_val = max(c['expected'] for c in sorted_cases)
                
                # Check if values cluster near a cap
                std_top = np.std([c['expected'] for c in top_10_pct]) if len(top_10_pct) > 1 else 0
                
                print(f"\n{days} days:")
                print(f"  High-receipt cases: {len(high_receipt_cases)}")
                print(f"  Max reimbursement: ${max_val:.2f}")
                print(f"  Top 10% average: ${avg_top:.2f} (std: ${std_top:.2f})")
                
                # If std is low, might be a cap
                if std_top < 50:
                    caps[days] = max_val
                    print(f"  Possible cap: ${max_val:.2f}")

def analyze_base_formula(data):
    """Try to find the base formula by looking at low-receipt cases"""
    print("\n\nAnalyzing base formula from low-receipt cases...")
    
    # Focus on cases with receipts < $100
    low_receipt_cases = []
    
    for case in data:
        if case['input']['total_receipts_amount'] < 100:
            low_receipt_cases.append(case)
    
    print(f"Found {len(low_receipt_cases)} cases with receipts < $100")
    
    # Extract features
    X = []
    y = []
    
    for case in low_receipt_cases:
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        
        X.append([d, m, r])
        y.append(case['expected_output'])
    
    X = np.array(X)
    y = np.array(y)
    
    # Try different models
    # Model 1: Linear with receipts
    X1 = np.column_stack([X, np.ones(len(X))])
    reg1 = LinearRegression(fit_intercept=False)
    reg1.fit(X1, y)
    
    print("\nModel 1 (with receipts):")
    print(f"  Per diem: ${reg1.coef_[0]:.2f}")
    print(f"  Mileage: ${reg1.coef_[1]:.4f}")
    print(f"  Receipt rate: {reg1.coef_[2]:.4f}")
    print(f"  Intercept: ${reg1.coef_[3]:.2f}")
    
    pred1 = reg1.predict(X1)
    mae1 = np.mean(np.abs(y - pred1))
    print(f"  MAE: ${mae1:.2f}")
    
    # Model 2: Subtract receipts first (maybe receipts reduce reimbursement)
    y_minus_r = y - X[:, 2]  # Expected minus receipts
    X2 = np.column_stack([X[:, :2], np.ones(len(X))])  # Just days and miles
    reg2 = LinearRegression(fit_intercept=False)
    reg2.fit(X2, y_minus_r)
    
    print("\nModel 2 (expected - receipts):")
    print(f"  Per diem: ${reg2.coef_[0]:.2f}")
    print(f"  Mileage: ${reg2.coef_[1]:.4f}")
    print(f"  Intercept: ${reg2.coef_[2]:.2f}")
    
    # This would mean: expected = per_diem * days + mileage * miles + intercept + receipts
    pred2 = reg2.predict(X2) + X[:, 2]
    mae2 = np.mean(np.abs(y - pred2))
    print(f"  MAE: ${mae2:.2f}")

def test_receipt_threshold_formula(data):
    """Test if there's a specific threshold where receipt handling changes"""
    print("\n\nTesting receipt threshold formulas...")
    
    # The famous $828.10 threshold
    threshold = 828.10
    
    # Split data
    below_threshold = []
    above_threshold = []
    
    for case in data:
        r = case['input']['total_receipts_amount']
        if r <= threshold:
            below_threshold.append(case)
        else:
            above_threshold.append(case)
    
    print(f"Cases below ${threshold}: {len(below_threshold)}")
    print(f"Cases above ${threshold}: {len(above_threshold)}")
    
    # Analyze each group separately
    for group_name, group in [("Below", below_threshold), ("Above", above_threshold)]:
        print(f"\n{group_name} threshold:")
        
        X = []
        y = []
        
        for case in group:
            d = case['input']['trip_duration_days']
            m = case['input']['miles_traveled']
            r = case['input']['total_receipts_amount']
            
            X.append([d, m, r])
            y.append(case['expected_output'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Fit model
        X_with_intercept = np.column_stack([X, np.ones(len(X))])
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_with_intercept, y)
        
        print(f"  Per diem: ${reg.coef_[0]:.2f}")
        print(f"  Mileage: ${reg.coef_[1]:.4f}")
        print(f"  Receipt rate: {reg.coef_[2]:.4f}")
        print(f"  Intercept: ${reg.coef_[3]:.2f}")
        
        pred = reg.predict(X_with_intercept)
        mae = np.mean(np.abs(y - pred))
        print(f"  MAE: ${mae:.2f}")

def visualize_receipt_patterns(data):
    """Create visualizations to understand receipt patterns"""
    # Extract data
    receipts = [c['input']['total_receipts_amount'] for c in data]
    expected = [c['expected_output'] for c in data]
    days = [c['input']['trip_duration_days'] for c in data]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Expected vs Receipts colored by days
    ax1 = axes[0, 0]
    scatter = ax1.scatter(receipts, expected, c=days, cmap='viridis', alpha=0.6, s=20)
    ax1.plot([0, 3000], [0, 3000], 'r--', alpha=0.5, label='y=x')
    ax1.set_xlabel('Receipt Amount ($)')
    ax1.set_ylabel('Expected Reimbursement ($)')
    ax1.set_title('Expected vs Receipts (colored by days)')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Days')
    
    # Plot 2: Expected/Receipts ratio vs Receipts
    ax2 = axes[0, 1]
    ratios = [e/r if r > 50 else None for e, r in zip(expected, receipts)]
    valid_receipts = [r for r, ratio in zip(receipts, ratios) if ratio is not None]
    valid_ratios = [ratio for ratio in ratios if ratio is not None]
    
    ax2.scatter(valid_receipts, valid_ratios, alpha=0.5, s=10)
    ax2.axhline(y=1, color='r', linestyle='--', label='ratio=1')
    ax2.set_xlabel('Receipt Amount ($)')
    ax2.set_ylabel('Expected / Receipts Ratio')
    ax2.set_title('Reimbursement Rate vs Receipt Amount')
    ax2.set_ylim(0, 3)
    ax2.legend()
    
    # Plot 3: Expected vs Receipts for different day ranges
    ax3 = axes[1, 0]
    day_ranges = [(1, 3), (4, 7), (8, 11), (12, 14)]
    colors = ['blue', 'green', 'orange', 'red']
    
    for (d_min, d_max), color in zip(day_ranges, colors):
        mask = [(d >= d_min and d <= d_max) for d in days]
        r_subset = [r for r, m in zip(receipts, mask) if m]
        e_subset = [e for e, m in zip(expected, mask) if m]
        ax3.scatter(r_subset, e_subset, alpha=0.5, s=10, color=color, label=f'{d_min}-{d_max} days')
    
    ax3.set_xlabel('Receipt Amount ($)')
    ax3.set_ylabel('Expected Reimbursement ($)')
    ax3.set_title('Expected vs Receipts by Day Ranges')
    ax3.legend()
    
    # Plot 4: Histogram of receipt penalty cases
    ax4 = axes[1, 1]
    penalties = [r - e for r, e in zip(receipts, expected) if r > e]
    ax4.hist(penalties, bins=50, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Receipt Penalty (receipts - expected) ($)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Distribution of Receipt Penalties ({len(penalties)} cases)')
    
    plt.tight_layout()
    plt.savefig('analysis/deep_receipt_patterns.png')
    print("\nVisualization saved to analysis/deep_receipt_patterns.png")

def main():
    print("Loading public cases...")
    data = load_public_cases()
    
    print(f"Deep diving into receipt transformation for {len(data)} cases...")
    
    # Run analyses
    analyze_receipt_penalty_cases(data)
    test_receipt_cap_hypothesis(data)
    analyze_base_formula(data)
    test_receipt_threshold_formula(data)
    visualize_receipt_patterns(data)

if __name__ == '__main__':
    main()