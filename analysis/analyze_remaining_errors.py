#!/usr/bin/env python3
"""
Analyze remaining high-error cases after applying receipt threshold model
"""
import json
import numpy as np
import subprocess
import os

def load_public_cases():
    """Load all public test cases"""
    with open('public_cases.json', 'r') as f:
        return json.load(f)

def test_current_solution(data):
    """Test current run.sh and analyze errors"""
    errors = []
    
    print("Testing current solution...")
    
    for i, case in enumerate(data):
        d = case['input']['trip_duration_days']
        m = case['input']['miles_traveled']
        r = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        # Run the script
        try:
            result = subprocess.check_output(['./run.sh', str(d), str(m), str(r)])
            predicted = float(result.strip())
        except:
            predicted = 0.0
        
        error = abs(predicted - expected)
        
        errors.append({
            'index': i,
            'days': d,
            'miles': m,
            'receipts': r,
            'expected': expected,
            'predicted': predicted,
            'error': error
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/1000 cases...")
    
    # Sort by error
    errors.sort(key=lambda x: x['error'], reverse=True)
    
    # Calculate statistics
    mae = np.mean([e['error'] for e in errors])
    perfect = sum(1 for e in errors if e['error'] < 0.01)
    within_1 = sum(1 for e in errors if e['error'] <= 1.0)
    
    print(f"\nResults:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  Perfect matches: {perfect}")
    print(f"  Within $1: {within_1}")
    
    return errors

def analyze_error_patterns(errors):
    """Analyze patterns in high-error cases"""
    print("\n\nAnalyzing error patterns...")
    
    # Group high errors by characteristics
    high_errors = [e for e in errors if e['error'] > 100]
    
    print(f"\nFound {len(high_errors)} cases with error > $100")
    
    # Analyze by receipt ranges
    receipt_ranges = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]
    
    for low, high in receipt_ranges:
        range_errors = [e for e in high_errors if low <= e['receipts'] < high]
        if range_errors:
            avg_error = np.mean([e['error'] for e in range_errors])
            print(f"\nReceipts ${low}-${high}: {len(range_errors)} high-error cases, avg error: ${avg_error:.2f}")
            
            # Show examples
            for e in range_errors[:3]:
                print(f"  {e['days']}d, {e['miles']:.0f}mi, ${e['receipts']:.2f}r -> " +
                      f"Expected: ${e['expected']:.2f}, Got: ${e['predicted']:.2f}, Error: ${e['error']:.2f}")
    
    # Look for specific patterns
    print("\n\nLooking for specific patterns...")
    
    # Pattern 1: Cases where expected < 1000 but predicted > 1500
    pattern1 = [e for e in high_errors if e['expected'] < 1000 and e['predicted'] > 1500]
    if pattern1:
        print(f"\nPattern 1: Expected < $1000 but predicted > $1500 ({len(pattern1)} cases)")
        for e in pattern1[:5]:
            print(f"  {e['days']}d, {e['miles']:.0f}mi, ${e['receipts']:.2f}r -> " +
                  f"Expected: ${e['expected']:.2f}, Got: ${e['predicted']:.2f}")
    
    # Pattern 2: High days (>10) with errors
    pattern2 = [e for e in high_errors if e['days'] > 10]
    if pattern2:
        print(f"\nPattern 2: High days (>10) with errors ({len(pattern2)} cases)")
        for e in pattern2[:5]:
            print(f"  {e['days']}d, {e['miles']:.0f}mi, ${e['receipts']:.2f}r -> " +
                  f"Expected: ${e['expected']:.2f}, Got: ${e['predicted']:.2f}")
    
    # Pattern 3: Look at expected values that seem capped
    print("\n\nAnalyzing potential caps by looking at maximum expected values per day:")
    
    by_days = {}
    for e in errors:
        d = e['days']
        if d not in by_days:
            by_days[d] = []
        by_days[d].append(e['expected'])
    
    for days in sorted(by_days.keys()):
        values = by_days[days]
        max_val = max(values)
        # Count how many are close to max
        near_max = sum(1 for v in values if v > max_val * 0.95)
        print(f"  {days} days: max=${max_val:.2f}, {near_max} cases within 5% of max")

def suggest_improvements(errors):
    """Suggest formula improvements based on error analysis"""
    print("\n\nSuggested improvements:")
    
    # Find cases where our formula significantly overestimates
    overestimates = [e for e in errors if e['predicted'] - e['expected'] > 100]
    
    if overestimates:
        print(f"\n1. Overestimation issue: {len(overestimates)} cases")
        
        # Check if these have common characteristics
        avg_days = np.mean([e['days'] for e in overestimates])
        avg_miles = np.mean([e['miles'] for e in overestimates])
        avg_receipts = np.mean([e['receipts'] for e in overestimates])
        
        print(f"   Average profile: {avg_days:.1f} days, {avg_miles:.0f} miles, ${avg_receipts:.0f} receipts")
        
        # Check if they're mostly high-receipt cases
        high_receipt_count = sum(1 for e in overestimates if e['receipts'] > 1000)
        print(f"   High receipt (>$1000): {high_receipt_count}/{len(overestimates)}")
    
    # Find perfect matches to understand what works
    perfect = [e for e in errors if e['error'] < 0.01]
    if perfect:
        print(f"\n2. Perfect matches: {len(perfect)} cases")
        print("   Examples:")
        for e in perfect[:5]:
            print(f"   {e['days']}d, {e['miles']:.0f}mi, ${e['receipts']:.2f}r -> ${e['expected']:.2f}")

def main():
    print("Loading public cases...")
    data = load_public_cases()
    
    # Test current solution
    errors = test_current_solution(data)
    
    # Analyze patterns
    analyze_error_patterns(errors)
    
    # Suggest improvements
    suggest_improvements(errors)
    
    # Save detailed error report
    print("\n\nSaving detailed error report...")
    
    with open('analysis/error_report.json', 'w') as f:
        json.dump({
            'total_cases': len(errors),
            'mae': float(np.mean([e['error'] for e in errors])),
            'perfect_matches': sum(1 for e in errors if e['error'] < 0.01),
            'worst_10': errors[:10]
        }, f, indent=2)
    
    print("Report saved to analysis/error_report.json")

if __name__ == '__main__':
    main()