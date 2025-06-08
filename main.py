#!/usr/bin/env python3
"""
ACME Corp Reimbursement Calculator - Final Version
Based on detailed cluster and breakpoint analysis
"""

import sys


def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """Calculate reimbursement using discovered formula with receipt breakpoint."""
    
    # Convert inputs
    days = int(trip_duration_days)
    miles = float(miles_traveled)
    receipts = float(total_receipts_amount)
    
    # Key insight: There's a critical receipt threshold at $828.10
    RECEIPT_THRESHOLD = 828.10
    
    # Different formulas for below/above threshold
    if receipts <= RECEIPT_THRESHOLD:
        # Below threshold formula (from final_model.json)
        per_diem = 56.0
        mileage_rate = 0.52
        receipt_rate = 0.51
        base_intercept = 32.49
        
        reimbursement = (per_diem * days + 
                        mileage_rate * miles + 
                        receipt_rate * receipts + 
                        base_intercept)
    else:
        # Above threshold formula
        per_diem = 41.0
        mileage_rate = 0.32
        receipt_rate = 0.11
        base_intercept = 899.94
        
        reimbursement = (per_diem * days + 
                        mileage_rate * miles + 
                        receipt_rate * receipts + 
                        base_intercept)
    
    # Apply daily caps based on trip duration
    daily_caps = {
        1: 1466.95,
        2: 1549.54,
        3: 1586.21,
        4: 1698.94,
        5: 1796.70,
        6: 1972.88,
        7: 2063.98,
        8: 1902.37,
        9: 1913.87,
        10: 1897.37,
        11: 2050.62,
        12: 1944.88,
        13: 2097.69,
        14: 2080.00
    }
    
    # Apply cap if trip duration is in the caps dictionary
    if days in daily_caps:
        reimbursement = min(reimbursement, daily_caps[days])
    
    # Final rounding to 2 decimal places
    return round(reimbursement, 2)


def main():
    if len(sys.argv) == 4:
        # Command line mode for eval.sh
        try:
            days = float(sys.argv[1])
            miles = float(sys.argv[2])
            receipts = float(sys.argv[3])
            result = calculate_reimbursement(days, miles, receipts)
            print(f"{result:.2f}")
        except Exception:
            sys.exit(1)
    else:
        # Test mode
        print("Usage: python main.py <days> <miles> <receipts>")
        sys.exit(1)


if __name__ == "__main__":
    main()