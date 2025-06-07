#!/usr/bin/env python3
"""
Analyze cases with high receipts to understand the pattern
Focus on cases where expected < receipts
"""
import json
import numpy as np
import matplotlib.pyplot as plt

def load_public_cases():
    """Load all public test cases"""
    with open('public_cases.json', 'r') as f:
        return json.load(f)

def analyze_high_receipt_cases(data):
    """Analyze cases where receipts are high"""
    # Extract data
    cases_info = []
    for i, case in enumerate(data):
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        exp = case['expected_output']
        
        # Calculate a simple base (without receipts)
        base_estimate = 50 * d + 0.5 * m  # rough estimate
        
        cases_info.append({
            'index': i,
            'days': d,
            'miles': m,
            'receipts': r,
            'expected': exp,
            'base_estimate': base_estimate,
            'receipt_contribution': exp - base_estimate,
            'expected_minus_receipts': exp - r
        })
    
    # Sort by receipts
    cases_info.sort(key=lambda x: x['receipts'])
    
    print("Cases where expected < receipts (i.e., receipts seem to be ignored or penalized):")
    print("="*100)
    
    negative_cases = [c for c in cases_info if c['expected'] < c['receipts']]
    print(f"Found {len(negative_cases)} cases where expected < receipts")
    
    # Analyze by receipt ranges
    ranges = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]
    
    for low, high in ranges:
        range_cases = [c for c in negative_cases if low <= c['receipts'] < high]
        if range_cases:
            print(f"\nReceipts ${low}-${high}: {len(range_cases)} cases")
            # Show first few examples
            for c in range_cases[:3]:
                print(f"  Case {c['index']}: {c['days']}d, {c['miles']:.0f}mi, ${c['receipts']:.2f}r")
                print(f"    Expected: ${c['expected']:.2f} (receipts - expected = ${c['receipts'] - c['expected']:.2f})")
    
    # Plot expected vs receipts for all cases
    plt.figure(figsize=(12, 8))
    
    receipts = [c['receipts'] for c in cases_info]
    expected = [c['expected'] for c in cases_info]
    
    plt.subplot(2, 2, 1)
    plt.scatter(receipts, expected, alpha=0.5, s=10)
    plt.plot([0, 3000], [0, 3000], 'r--', label='y=x (if receipts were added 1:1)')
    plt.xlabel('Receipt Amount ($)')
    plt.ylabel('Expected Reimbursement ($)')
    plt.title('Expected vs Receipts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot receipt contribution (expected - base)
    plt.subplot(2, 2, 2)
    receipt_contrib = [c['receipt_contribution'] for c in cases_info]
    plt.scatter(receipts, receipt_contrib, alpha=0.5, s=10)
    plt.xlabel('Receipt Amount ($)')
    plt.ylabel('Receipt Contribution ($)')
    plt.title('Receipt Contribution to Total')
    plt.grid(True, alpha=0.3)
    
    # Focus on high receipt cases
    high_receipt_cases = [c for c in cases_info if c['receipts'] > 1000]
    
    plt.subplot(2, 2, 3)
    hr_receipts = [c['receipts'] for c in high_receipt_cases]
    hr_expected = [c['expected'] for c in high_receipt_cases]
    plt.scatter(hr_receipts, hr_expected, alpha=0.5, s=10)
    plt.xlabel('Receipt Amount ($)')
    plt.ylabel('Expected Reimbursement ($)')
    plt.title('High Receipt Cases (>$1000)')
    plt.grid(True, alpha=0.3)
    
    # Analyze ratio of expected/receipts
    plt.subplot(2, 2, 4)
    ratios = [c['expected'] / c['receipts'] if c['receipts'] > 0 else 0 for c in cases_info if c['receipts'] > 100]
    receipts_for_ratio = [c['receipts'] for c in cases_info if c['receipts'] > 100]
    plt.scatter(receipts_for_ratio, ratios, alpha=0.5, s=10)
    plt.xlabel('Receipt Amount ($)')
    plt.ylabel('Expected / Receipts Ratio')
    plt.title('Reimbursement Rate vs Receipt Amount')
    plt.ylim(0, 2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/high_receipts_analysis.png')
    print("\nPlot saved to analysis/high_receipts_analysis.png")
    
    # Look for patterns in the expected/receipts ratio
    print("\nAnalyzing reimbursement patterns:")
    
    # Group by days and analyze
    for days in range(1, 15):
        day_cases = [c for c in cases_info if c['days'] == days]
        if len(day_cases) > 10:
            receipts_array = np.array([c['receipts'] for c in day_cases])
            expected_array = np.array([c['expected'] for c in day_cases])
            
            # Find cases with similar base (days/miles) but different receipts
            miles_array = np.array([c['miles'] for c in day_cases])
            
            # Look for receipt threshold
            high_r = receipts_array > 1000
            if high_r.sum() > 5 and (~high_r).sum() > 5:
                avg_high = expected_array[high_r].mean()
                avg_low = expected_array[~high_r].mean()
                print(f"\n{days} days: avg reimbursement for receipts >$1000: ${avg_high:.2f}, <=$1000: ${avg_low:.2f}")

def find_receipt_threshold(data):
    """Try to find the exact receipt threshold where behavior changes"""
    print("\n\nSearching for receipt threshold...")
    
    # Create sorted list by receipts
    sorted_cases = []
    for i, case in enumerate(data):
        sorted_cases.append({
            'receipts': case['input']['total_receipts_amount'],
            'expected': case['expected_output'],
            'days': case['input']['trip_duration_days'],
            'miles': case['input']['miles_traveled']
        })
    
    sorted_cases.sort(key=lambda x: x['receipts'])
    
    # Look for sharp changes in expected/receipts ratio
    for i in range(1, len(sorted_cases)-1):
        curr = sorted_cases[i]
        prev = sorted_cases[i-1]
        next = sorted_cases[i+1]
        
        if curr['receipts'] > 100:  # Ignore very low receipts
            curr_ratio = curr['expected'] / curr['receipts']
            prev_ratio = prev['expected'] / prev['receipts'] if prev['receipts'] > 0 else 0
            next_ratio = next['expected'] / next['receipts'] if next['receipts'] > 0 else 0
            
            # Look for sharp drops in ratio
            if prev_ratio > 0 and curr_ratio < prev_ratio * 0.5:
                print(f"  Sharp drop at ${curr['receipts']:.2f}: ratio {prev_ratio:.3f} -> {curr_ratio:.3f}")

def main():
    print("Loading public cases...")
    data = load_public_cases()
    
    print(f"Analyzing {len(data)} cases focusing on high receipts...")
    analyze_high_receipt_cases(data)
    
    find_receipt_threshold(data)

if __name__ == '__main__':
    main()