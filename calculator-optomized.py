#!/usr/bin/env python3
"""
ACME Corp Reimbursement Calculator - Optimized Version
Auto-generated with machine learning optimized parameters
"""

import random


def calculate_reimbursement(trip_duration_days: int, miles_traveled: float, 
                           total_receipts_amount: float) -> float:
    """Calculate reimbursement using optimized parameters."""
    
    # Optimized parameters from ML
    BASE_PER_DIEM = 80.00
    FIVE_DAY_BONUS = 0.103
    
    MILEAGE_TIERS = [
        (106, 0.454),
        (377, 0.405),
        (771, 0.410),
        (1135, 0.373),
        (float('inf'), 0.296)
    ]
    
    EFFICIENCY_BONUS = 0.020
    EFFICIENCY_RANGE = (170, 200)
    
    SMALL_RECEIPT_THRESH = 53
    SMALL_RECEIPT_PENALTY = 0.712
    MEDIUM_RECEIPT_RANGE = (43, 688)
    MEDIUM_RECEIPT_COVERAGE = 0.809
    HIGH_RECEIPT_COVERAGE = 0.935
    
    CLUSTER_MULTIPLIERS = [
        0.933,  # Road warrior
        1.002,  # Optimal business
        0.821,  # Extended low
        0.900,  # Standard
        0.914,  # Local
        0.959   # High spend
    ]
    
    # Classify trip
    if trip_duration_days == 0:
        cluster = 3
    else:
        mpd = miles_traveled / trip_duration_days
        rpd = total_receipts_amount / trip_duration_days
        
        if mpd > 350 and trip_duration_days <= 3:
            cluster = 0
        elif trip_duration_days == 5 and 150 <= mpd <= 250 and rpd < 120:
            cluster = 1
        elif trip_duration_days >= 7 and mpd < 150:
            cluster = 2
        elif rpd > 130:
            cluster = 5
        elif miles_traveled < 80 and trip_duration_days <= 2:
            cluster = 4
        else:
            cluster = 3
    
    # Per diem
    per_diem = BASE_PER_DIEM * trip_duration_days
    if trip_duration_days == 5:
        per_diem *= (1 + FIVE_DAY_BONUS)
    
    # Mileage
    mileage = 0.0
    remaining = miles_traveled
    prev_tier = 0
    
    for tier_limit, rate in MILEAGE_TIERS:
        if remaining <= 0:
            break
        tier_miles = min(remaining, tier_limit - prev_tier)
        mileage += tier_miles * rate
        remaining -= tier_miles
        prev_tier = tier_limit
    
    # Efficiency bonus
    efficiency_bonus = 0.0
    if trip_duration_days > 0:
        mpd = miles_traveled / trip_duration_days
        if EFFICIENCY_RANGE[0] <= mpd <= EFFICIENCY_RANGE[1]:
            efficiency_bonus = (per_diem + mileage) * EFFICIENCY_BONUS
    
    # Receipt reimbursement
    if total_receipts_amount == 0:
        receipt_reimb = 0.0
    elif total_receipts_amount < SMALL_RECEIPT_THRESH:
        receipt_reimb = total_receipts_amount * SMALL_RECEIPT_PENALTY
    elif MEDIUM_RECEIPT_RANGE[0] <= total_receipts_amount <= MEDIUM_RECEIPT_RANGE[1]:
        receipt_reimb = total_receipts_amount * MEDIUM_RECEIPT_COVERAGE
    else:
        base = MEDIUM_RECEIPT_RANGE[1] * HIGH_RECEIPT_COVERAGE
        excess = total_receipts_amount - MEDIUM_RECEIPT_RANGE[1]
        excess_rate = 0.6 * (1 - excess / 3000)
        excess_rate = max(0.2, excess_rate)
        receipt_reimb = base + (excess * excess_rate)
    
    # Total with cluster adjustment
    total = per_diem + mileage + efficiency_bonus + receipt_reimb
    total *= CLUSTER_MULTIPLIERS[cluster]
    
    # Add noise
    noise_seed = int((trip_duration_days * 1000 + miles_traveled * 10 + total_receipts_amount * 100) % 1000)
    random.seed(noise_seed)
    noise = random.uniform(-0.01, 0.01)
    total *= (1 + noise)
    
    # Magic cents
    cents = int(round(total_receipts_amount * 100)) % 100
    if cents in [49, 99]:
        total += random.uniform(2, 5)
    
    return round(total, 2)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: reimbursement_calculator_optimized.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        trip_duration = int(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        result = calculate_reimbursement(trip_duration, miles, receipts)
        print(result)
        
    except (ValueError, TypeError) as e:
        print(f"Error: Invalid input - {e}", file=sys.stderr)
        sys.exit(1)
